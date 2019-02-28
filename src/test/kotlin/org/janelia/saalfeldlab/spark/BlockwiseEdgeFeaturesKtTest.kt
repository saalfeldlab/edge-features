package org.janelia.saalfeldlab.spark

import net.imglib2.FinalInterval
import net.imglib2.algorithm.util.Grids
import net.imglib2.img.array.ArrayImgs
import net.imglib2.type.numeric.integer.UnsignedLongType
import net.imglib2.type.numeric.real.FloatType
import net.imglib2.util.Intervals
import net.imglib2.view.Views
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.janelia.saalfeldlab.edge.feature.Histogram
import org.janelia.saalfeldlab.n5.DataType
import org.janelia.saalfeldlab.n5.GzipCompression
import org.janelia.saalfeldlab.n5.N5FSWriter
import org.janelia.saalfeldlab.n5.imglib2.N5Utils
import org.janelia.saalfeldlab.util.N5TestUtil
import org.junit.Test
import org.slf4j.LoggerFactory
import java.lang.invoke.MethodHandles

class BlockwiseEdgeFeaturesKtTest {

    @Test
    fun testFromImage() {
        val dims = longArrayOf(10, 20)
        val blockSize = intArrayOf(4, 7)
        val labels = ArrayImgs.unsignedLongs(*dims)
        val weights = ArrayImgs.floats(*dims)
        Views.interval(labels, FinalInterval(10, 10)).forEach(UnsignedLongType::setOne)
        Views.interval(labels, Intervals.translate(FinalInterval(10, 10), 0, 10L)).forEach {it.setInteger(3L)}
        Views.interval(labels, Intervals.createMinMax(3, 4, 6, 12)).forEach {it.setInteger(2L)}
        weights.forEach(FloatType::setOne)

        val tmpDir = N5TestUtil.tmpDir(!LOG.isDebugEnabled).toAbsolutePath().toString()
        LOG.debug("Created tmp dir at {}", tmpDir)
        N5FSWriter(tmpDir).let {
            N5Utils.save(labels, it, "labels", blockSize, GzipCompression())
            N5Utils.save(weights, it, "weights", blockSize, GzipCompression())
            it.createDataset("feature-blocks", dims, blockSize, DataType.INT8, GzipCompression())
        }

        val superBlocks = Grids.collectAllContainedIntervals(dims, blockSize.map { it * 2 }.toIntArray())

        val conf = SparkConf().setAppName(MethodHandles.lookup().lookupClass().simpleName)
        val sc = JavaSparkContext(conf)
        val n5io = {
            N5IO(
                featureBlockContainer = N5FSWriter(tmpDir),
                mergedFeaturesDataset = "edge-features",
                    edgesDataset = "edges",
                    featureBlockDataset = "feature-blocks",
                    labelsDataset = "labels",
                    weightsDataset = "labels"
            )}

        val featureRequests = arrayOf({ Histogram(nBins = 10) })

        sc.use {
            BlockwiseEdgeFeatures.updateFeatureBlocks(it, n5io, superBlocks, dims, blockSize, *featureRequests, numEdgesPerBlock = 1)
            BlockwiseEdgeFeatures.mergeFeatures(it, n5io, *featureRequests, numEdgesPerBlock = 1)
        }


    }


    companion object {

        private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())

    }
}