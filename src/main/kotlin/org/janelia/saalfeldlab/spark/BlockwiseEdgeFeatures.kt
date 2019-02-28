package org.janelia.saalfeldlab.spark

import gnu.trove.map.hash.TLongObjectHashMap
import gnu.trove.set.TLongSet
import gnu.trove.set.hash.TLongHashSet
import net.imglib2.FinalInterval
import net.imglib2.Interval
import net.imglib2.algorithm.util.Grids
import net.imglib2.img.array.ArrayImgs
import net.imglib2.img.cell.CellGrid
import net.imglib2.type.numeric.integer.UnsignedLongType
import net.imglib2.type.numeric.real.FloatType
import net.imglib2.util.Intervals
import net.imglib2.view.Views
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.janelia.saalfeldlab.edge.feature.DoubleStatisticsFeature
import org.janelia.saalfeldlab.edge.feature.Histogram
import org.janelia.saalfeldlab.labels.Label
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock
import org.janelia.saalfeldlab.n5.DataType
import org.janelia.saalfeldlab.n5.GzipCompression
import org.janelia.saalfeldlab.n5.N5FSReader
import org.janelia.saalfeldlab.n5.N5FSWriter
import org.janelia.saalfeldlab.n5.N5Reader
import org.janelia.saalfeldlab.n5.N5Writer
import org.janelia.saalfeldlab.n5.imglib2.N5Utils
import org.janelia.saalfeldlab.util.computeIfAbsent
import org.slf4j.LoggerFactory
import scala.Tuple2
import scala.Tuple3
import java.lang.invoke.MethodHandles
import java.nio.ByteBuffer
import kotlin.random.Random

private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())

class N5IO(
        val featureBlockContainer: N5Writer,
        val weightsContainer: N5Reader = featureBlockContainer,
        val labelsContainer: N5Reader = featureBlockContainer,
        val edgesContainer: N5Writer = featureBlockContainer,
        val weightsDataset: String,
        val labelsDataset: String,
        val featureBlockDataset: String,
        val edgesDataset: String)

class FeatureAndName(val feature: () -> DoubleStatisticsFeature<*>, val name: String)

fun updateFeatureBlocks(
        sc: JavaSparkContext,
        n5io: () -> N5IO,
        superBlocks: List<Interval>,
        dimensions: LongArray,
        blockSize: IntArray,
        vararg features: () -> DoubleStatisticsFeature<*>,
        numEdgesPerBlock: Int = 1 shl 16) {

    val edges = TLongObjectHashMap<TLongSet>()
    val perBlockEdges = sc
            .parallelize(superBlocks.map{ Tuple2(Intervals.minAsLongArray(it), Intervals.maxAsLongArray(it)) })
            .map {

                val grid = CellGrid(dimensions, blockSize)
                val blockPos = LongArray(grid.numDimensions())
                val edgeMap = TLongObjectHashMap<TLongHashSet>()
                LOG.info("Super block ({} {}) for block size {}", it._1(), it._2(), blockSize)

                Grids.collectAllContainedIntervals(it._1(), it._2(), blockSize).forEach {block ->
                    LOG.info("Extracting features for block {} inside superblock ({} {})", block, it._1(), it._2())
                    block.min(blockPos)
                    grid.getCellPosition(blockPos, blockPos)
                    val weights = Views.extendValue(N5Utils.open<FloatType>(n5io().weightsContainer, n5io().weightsDataset), FloatType(Float.NaN))
                    val labels = Views.extendValue(N5Utils.open<UnsignedLongType>(n5io().labelsContainer, n5io().labelsDataset), UnsignedLongType(Label.INVALID))
                    val edgeFeatures = DoubleStatisticsFeature.addAll(weights, labels, block, features = *features)
                    edgeFeatures.forEachEntry { key, values -> edgeMap.computeIfAbsent(key) { TLongHashSet() }.addAll(values.keySet()); true }
                    LOG.info("Edge features: {}", edgeFeatures)
                    // TLongObjectHashMap$KeyView does not have closing curly brace in string representation!!
                    LOG.info("Edge map: {}", edgeMap)

                    var requiredSize = 0
                    edgeFeatures.forEachEntry { e1, set -> set.forEachEntry { e2, features -> requiredSize += 2 * java.lang.Long.BYTES + features.map(DoubleStatisticsFeature<*>::numBytes).sum(); true }; true }
                    val blockData = ByteArray(requiredSize)
                    val buffer = ByteBuffer.wrap(blockData)
                    edgeFeatures.forEachEntry { e1, set -> set.forEachEntry { e2, features -> buffer.putLong(e1); buffer.putLong(e2); features.forEach { it.serializeInto(buffer) }; true }; true }
                    buffer.rewind()
                    val dataBlock = ByteArrayDataBlock(Intervals.dimensionsAsIntArray(block), blockPos, blockData)
                    n5io().featureBlockContainer.writeBlock(n5io().featureBlockDataset, n5io().featureBlockContainer.getDatasetAttributes(n5io().featureBlockDataset), dataBlock)
                }

                edgeMap
            }
            .collect()
    perBlockEdges.forEach { it.forEachEntry { a, b -> edges.computeIfAbsent(a, {TLongHashSet()}).addAll(b); true } }
    val edgesContainer = n5io().edgesContainer
    val edgesDataset = n5io().edgesDataset
    if (edgesContainer.datasetExists(edgesDataset)) {
        val cursor = Views.flatIterable(N5Utils.open<UnsignedLongType>(edgesContainer, edgesDataset)).cursor()
        while (cursor.hasNext())
            edges.computeIfAbsent(cursor.next().integerLong, {TLongHashSet()}).add(cursor.next().integerLong)
    }

    val edgeCount = edges.valueCollection().map { it.size() }.reduce { i1, i2 -> i1 + i2}
    val edgesRai = ArrayImgs.unsignedLongs(2L, edgeCount.toLong())
    val edgesRaiCursor = Views.flatIterable(edgesRai).cursor()
    edges.forEachEntry { e1, value -> value.forEach { edgesRaiCursor.next().setInteger(e1); edgesRaiCursor.next().setInteger(it); true }; true }

    N5Utils.save(edgesRai, edgesContainer, edgesDataset, intArrayOf(2, numEdgesPerBlock), GzipCompression())

}


}

fun main(args: Array<String>) {

    val dims = longArrayOf(30, 40, 35)
    val blockSize = dims.map { it.toInt() }.toIntArray()
    blockSize[0] /= 2
    val rng = Random(100L)
    val randomLabels = ArrayImgs.unsignedLongs(*dims)
    val randomEdges = ArrayImgs.floats(*dims)
    randomLabels.forEach { it.setInteger(rng.nextLong(1, 11)) }
    randomEdges.forEach { it.setReal(rng.nextFloat()) }

    val path = "${System.getProperty("user.home")}/.local/tmp/edge-features.n5"
    N5Utils.save(randomLabels, N5FSWriter(path), "labels", blockSize, GzipCompression())
    N5Utils.save(randomEdges, N5FSWriter(path), "affinities", blockSize, GzipCompression())

    N5FSWriter(path).createDataset("feature-blocks", dims, blockSize, DataType.INT8, GzipCompression())

    val conf = SparkConf().setAppName(MethodHandles.lookup().lookupClass().simpleName)
    val sc = JavaSparkContext(conf)
    val n5io = {N5IO(
            weightsContainer = N5FSReader(path),
            weightsDataset = "affinities",
            labelsDataset = "labels",
            featureBlockContainer = N5FSWriter(path),
            featureBlockDataset = "feature-blocks",
            edgesDataset = "edges"
    )}
    val blocks = listOf(
            FinalInterval(*LongArray(blockSize.size, {blockSize[it].toLong()})),
            FinalInterval(LongArray(blockSize.size, {if (it == 0) blockSize[it].toLong() else 0L}), dims.map {it - 1}.toLongArray()))

    sc.use {
        updateFeatureBlocks(it, n5io, blocks, dims, blockSize, {Histogram(5,max=1.0001)})
    }


    val featureMap = TLongObjectHashMap<TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>>()
    for (index in 0 .. 1) {
        val dataBlock = N5FSReader(path).let { it.readBlock("feature-blocks", it.getDatasetAttributes("feature-blocks"), longArrayOf(index.toLong(), 0, 0)) as ByteArrayDataBlock }
        val localFeatureMap = TLongObjectHashMap<TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>>()
        val buffer = ByteBuffer.wrap(dataBlock.data)
        while (buffer.hasRemaining()) {
            val k1 = buffer.long
            val k2 = buffer.long
            val h = Histogram(5, max = 1.0001).let { it.deserializeFrom(buffer); it}
            localFeatureMap.computeIfAbsent(k1) { TLongObjectHashMap() }.put(k2, listOf(h))
            featureMap.computeIfAbsent(k1) { TLongObjectHashMap() }.let { it.put(k2, listOf(it[k2]?.get(0)?.plusUnsafe(h) ?: h)) }
        }
        LOG.info("Block edge features: {}", localFeatureMap)
    }

    LOG.info("Edge features: {}", featureMap)

}