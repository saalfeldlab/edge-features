package org.janelia.saalfeldlab.spark

import gnu.trove.map.hash.TLongObjectHashMap
import gnu.trove.set.TLongSet
import gnu.trove.set.hash.TLongHashSet
import net.imglib2.FinalInterval
import net.imglib2.RandomAccess
import net.imglib2.algorithm.util.Grids
import net.imglib2.img.array.ArrayImgs
import net.imglib2.type.numeric.integer.UnsignedLongType
import net.imglib2.type.numeric.real.DoubleType
import net.imglib2.type.numeric.real.FloatType
import net.imglib2.util.Intervals
import net.imglib2.view.Views
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.janelia.saalfeldlab.edge.feature.DoubleStatisticsFeature
import org.janelia.saalfeldlab.edge.feature.Histogram
import org.janelia.saalfeldlab.n5.DataType
import org.janelia.saalfeldlab.n5.GzipCompression
import org.janelia.saalfeldlab.n5.N5FSWriter
import org.janelia.saalfeldlab.n5.imglib2.N5Utils
import org.janelia.saalfeldlab.util.N5TestUtil
import org.janelia.saalfeldlab.util.computeIfAbsent
import org.junit.Assert
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
        val n5io = { N5IO(
                    featureBlockContainer = N5FSWriter(tmpDir),
                    mergedFeaturesDataset = "edge-features",
                    edgesDataset = "edges",
                    featureBlockDataset = "feature-blocks",
                    labelsDataset = "labels",
                    weightsDataset = "weights"
            )}

        val featureRequests = arrayOf({ Histogram(nBins = 10) })
        val numDoubleEntries = featureRequests.map { it().packedSizeInDoubles() }.sum()
        val numFeatureBytes = featureRequests.map { it().numBytes() }.sum()

        sc.use {
            BlockwiseEdgeFeatures.updateFeatureBlocks(it, n5io, superBlocks, dims, blockSize, *featureRequests)
            BlockwiseEdgeFeatures.findEdges(it, n5io, numFeatureBytes, numEdgesPerBlock = 1)
            BlockwiseEdgeFeatures.mergeFeaturesWithTreeAggregate(it, n5io, *featureRequests, numEdgesPerBlock = 1)
//            BlockwiseEdgeFeatures.mergeFeatures(it, n5io, *featureRequests, numEdgesPerBlock = 1)
        }

        n5io().let {n5 ->
            Assert.assertTrue(n5.weightsContainer.datasetExists(n5.weightsDataset))
            Assert.assertTrue(n5.labelsContainer.datasetExists(n5.labelsDataset))
            Assert.assertTrue(n5.featureBlockContainer.datasetExists(n5.featureBlockDataset))
            Assert.assertTrue(n5.edgesContainer.datasetExists(n5.edgesDataset))
            Assert.assertTrue(n5.mergedFeaturesContainer.datasetExists(n5.mergedFeaturesDataset))

            Assert.assertEquals(DataType.INT8, n5.featureBlockContainer.getDatasetAttributes(n5.featureBlockDataset).dataType)
            Assert.assertEquals(DataType.UINT64, n5.edgesContainer.getDatasetAttributes(n5.edgesDataset).dataType)
            Assert.assertEquals(DataType.FLOAT64, n5.mergedFeaturesContainer.getDatasetAttributes(n5.mergedFeaturesDataset).dataType)

            N5Utils.open<UnsignedLongType>(n5.edgesContainer, n5.edgesDataset).let {
                Assert.assertArrayEquals(longArrayOf(2, 3), Intervals.dimensionsAsLongArray(it))
                val edgeMap = TLongObjectHashMap<TLongSet>()
                val edges = (0 .. 2).map { idx -> it.randomAccess().let{ ra -> Pair(ra.get(0, idx.toLong()).integerLong, ra.get(1, idx.toLong()).integerLong) } }
                edges.forEach { edgeMap.computeIfAbsent(it.first) {TLongHashSet()}.add(it.second) }
                LOG.debug("Extracted edge map {}", edgeMap)
                Assert.assertEquals(TLongHashSet(longArrayOf(1, 2)), edgeMap.keySet())
                Assert.assertEquals(TLongHashSet(longArrayOf(2, 3)), edgeMap[1])
                Assert.assertEquals(TLongHashSet(longArrayOf(3)), edgeMap[2])

                val groundTruthFeatures = DoubleStatisticsFeature.addAll(Views.extendValue(weights, FloatType(Float.NaN)), Views.extendZero(labels), weights, features = *featureRequests)
                N5Utils.open<DoubleType>(n5.mergedFeaturesContainer, n5.mergedFeaturesDataset).let {
                    Assert.assertArrayEquals(longArrayOf(numDoubleEntries.toLong(), 3), Intervals.dimensionsAsLongArray(it))
                    for (index in edges.indices) {
                        val edge = edges[index]
                        val actual = Views.flatIterable(Views.hyperSlice(it, 1, index.toLong())).cursor()
                        val expected = ArrayImgs.doubles(numDoubleEntries.toLong())
                        Views.flatIterable(expected).cursor().let {
                            for (expectedFeature in groundTruthFeatures[edge.first][edge.second])
                                expectedFeature.pack().serializeInto(it)
                        }
                        Views.flatIterable(expected).cursor().let {
                            while (it.hasNext())
                                Assert.assertEquals(it.next(), actual.next())
                        }
                    }
                }

            }
        }


    }


    companion object {

        private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())

        private fun <T> RandomAccess<T>.get(vararg pos: Long): T {
            this.setPosition(pos)
            return get()
        }

    }
}