package org.janelia.saalfeldlab.spark

import gnu.trove.map.TLongObjectMap
import gnu.trove.map.hash.TLongLongHashMap
import gnu.trove.map.hash.TLongObjectHashMap
import gnu.trove.set.TLongSet
import gnu.trove.set.hash.TLongHashSet
import net.imglib2.FinalInterval
import net.imglib2.Interval
import net.imglib2.algorithm.util.Grids
import net.imglib2.get
import net.imglib2.img.array.ArrayImgs
import net.imglib2.img.cell.CellGrid
import net.imglib2.type.numeric.integer.UnsignedLongType
import net.imglib2.type.numeric.real.FloatType
import net.imglib2.util.Intervals
import net.imglib2.view.Views
import org.apache.commons.lang3.builder.ToStringBuilder
import org.apache.commons.lang3.builder.ToStringStyle
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.FlatMapFunction
import org.apache.spark.api.java.function.Function
import org.apache.spark.api.java.function.VoidFunction
import org.apache.spark.broadcast.Broadcast
import org.janelia.saalfeldlab.edge.feature.DoubleStatisticsFeature
import org.janelia.saalfeldlab.edge.feature.Histogram
import org.janelia.saalfeldlab.labels.Label
import org.janelia.saalfeldlab.n5.*
import org.janelia.saalfeldlab.n5.imglib2.N5Utils
import org.janelia.saalfeldlab.util.computeIfAbsent
import org.slf4j.LoggerFactory
import scala.Tuple2
import java.lang.invoke.MethodHandles
import java.nio.ByteBuffer
import java.util.*
import java.util.concurrent.Executors
import kotlin.collections.HashMap
import kotlin.math.min

private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())

private typealias MinMaxInterval = Tuple2<LongArray, LongArray>

private fun MinMaxInterval.min() = _1()
private fun MinMaxInterval.max() = _2()
private fun MinMaxInterval.interval() = FinalInterval(min(), max())
private fun MinMaxInterval.size() = Intervals.dimensionsAsLongArray(interval())

private fun asMinMaxInterval(interval: Interval) = MinMaxInterval(Intervals.minAsLongArray(interval), Intervals.maxAsLongArray(interval))

class N5IO(
        val featureBlockContainer: N5Writer,
        val weightsContainer: N5Reader = featureBlockContainer,
        val labelsContainer: N5Reader = featureBlockContainer,
        val edgesContainer: N5Writer = featureBlockContainer,
        val mergedFeaturesContainer: N5Writer = featureBlockContainer,
        val weightsDataset: String,
        val labelsDataset: String,
        val featureBlockDataset: String,
        val edgesDataset: String,
        val mergedFeaturesDataset: String)

class FeatureAndName(val feature: () -> DoubleStatisticsFeature<*>, val name: String)

class BlockwiseEdgeFeatures {

    companion object {

        private class WriteBlock<T>(
                val dataBlock: DataBlock<T>,
                val container: N5Writer,
                val dataset: String,
                val datasetAttributes: DatasetAttributes? = null): Runnable {
            override fun run() {
                LOG.debug("Writing block {}", dataBlock)
                container.writeBlock(dataset, datasetAttributes ?: container.getDatasetAttributes(dataset), dataBlock)
            }

            override fun toString(): String {
                return ToStringBuilder(this, ToStringStyle.SHORT_PREFIX_STYLE)
                        .append("dataBlock", dataBlock)
                        .append("container", container)
                        .append("dataset", dataset)
                        .append("datasetAttributes", datasetAttributes)
                        .toString()
            }

        }

        fun updateFeatureBlocks(
                sc: JavaSparkContext,
                n5io: () -> N5IO,
                superBlocks: List<Interval>,
                dimensions: LongArray,
                blockSize: IntArray,
                vararg features: () -> DoubleStatisticsFeature<*>,
                numWriterThreads: Int = 1) {

            val edges = TLongObjectHashMap<TLongSet>()
            val perBlockEdges = sc
                    .parallelize(superBlocks.map { MinMaxInterval(Intervals.minAsLongArray(it), Intervals.maxAsLongArray(it)) })
                    .map {

                        val ioExecutors = Executors.newFixedThreadPool(numWriterThreads) {val t = Thread(it); t.name = "io-executor-$it"; t.isDaemon = true; LOG.debug("Creating new thread {}", t); t}
                        val grid = CellGrid(dimensions, blockSize)
                        val edgeMap = TLongObjectHashMap<TLongHashSet>()
                        val weights = Views.extendValue(N5Utils.open<FloatType>(n5io().weightsContainer, n5io().weightsDataset), FloatType(Float.NaN))
                        val labels = Views.extendValue(N5Utils.open<UnsignedLongType>(n5io().labelsContainer, n5io().labelsDataset), UnsignedLongType(Label.INVALID))
                        LOG.debug("Super block ({} {}) for block size {}", it.min(), it.max(), blockSize)

                        Grids.collectAllContainedIntervals(it._1(), it._2(), blockSize).forEach { block ->
                            val blockPos = LongArray(grid.numDimensions())
                            LOG.debug("Extracting features for block {} inside super-block ({} {})", block, it.min(), it.max())
                            block.min(blockPos)
                            grid.getCellPosition(blockPos, blockPos)
                            val edgeFeatures = DoubleStatisticsFeature.addAll(weights, labels, block, features = *features)
                            edgeFeatures.forEachEntry { key, values -> edgeMap.computeIfAbsent(key) { TLongHashSet() }.addAll(values.keySet()); true }
                            LOG.debug("Edge features {} in block {}", edgeFeatures, block)
                            // TLongObjectHashMap$KeyView does not have closing curly brace in string representation!!
                            LOG.debug("Edge map: {}", edgeMap)

                            var requiredSize = 0
                            edgeFeatures.forEachEntry { e1, set -> set.forEachEntry { e2, features -> requiredSize += 2 * java.lang.Long.BYTES + features.map(DoubleStatisticsFeature<*>::numBytes).sum(); true }; true }
                            val blockData = ByteArray(requiredSize)
                            val buffer = ByteBuffer.wrap(blockData)
                            edgeFeatures.forEachEntry { e1, set -> set.forEachEntry { e2, features -> buffer.putLong(e1); buffer.putLong(e2); features.forEach { it.serializeInto(buffer) }; true }; true }
                            buffer.rewind()
                            val dataBlock = ByteArrayDataBlock(Intervals.dimensionsAsIntArray(block), blockPos, blockData)
                            val task = n5io().let { n5 -> WriteBlock(dataBlock, n5.featureBlockContainer, n5.featureBlockDataset) }
                            LOG.debug("Submitting task {}", task)
                            ioExecutors.submit(task)
                        }

                        ioExecutors.shutdown()
                        LOG.debug("Finished processing super block {}", it)
                        edgeMap
                    }
                    .count()
        }

        fun findEdges(
                sc: JavaSparkContext,
                n5io: () -> N5IO,
                numFeatureBytes: Int,
                vararg blocksPerSuperBlock: Int,
                numEdgesPerBlock: Int = 1 shl 16) {
            n5io().let { n5 ->
                val featureBlockAttributes = n5.featureBlockContainer.getDatasetAttributes(n5.featureBlockDataset)
                val actualBlocksPerSuperBlock = if (blocksPerSuperBlock.isEmpty()) IntArray(featureBlockAttributes.numDimensions, {1}) else blocksPerSuperBlock
                require(actualBlocksPerSuperBlock.size == featureBlockAttributes.numDimensions)
                {"Super block size has wrong number of dimensions. Required: ${featureBlockAttributes.numDimensions} Found: ${Arrays.toString(blocksPerSuperBlock)}"}

                val superBlockSize = (actualBlocksPerSuperBlock zip featureBlockAttributes.blockSize).map { it.first * it.second }.toIntArray()

                val superBlocks = Grids
                        .collectAllContainedIntervals(featureBlockAttributes.dimensions, superBlockSize)
                        .map {MinMaxInterval(Intervals.minAsLongArray(it), Intervals.maxAsLongArray(it))}

                val edges = TLongObjectHashMap<TLongSet>()
                val perBlockEdges = sc
                        .parallelize(superBlocks)
                        .map(ExtractEdges(n5io = n5io, numFeatureBytes = numFeatureBytes))
                        .collect()
                perBlockEdges.forEach { it.forEachEntry { a, b -> edges.computeIfAbsent(a, { TLongHashSet() }).addAll(b); true } }
                val edgesContainer = n5io().edgesContainer
                val edgesDataset = n5io().edgesDataset
                if (edgesContainer.datasetExists(edgesDataset)) {
                    val cursor = Views.flatIterable(N5Utils.open<UnsignedLongType>(edgesContainer, edgesDataset)).cursor()
                    while (cursor.hasNext())
                        edges.computeIfAbsent(cursor.next().integerLong, { TLongHashSet() }).add(cursor.next().integerLong)
                }

                val edgeCount = edges.valueCollection().map { it.size() }.reduce { i1, i2 -> i1 + i2 }
                val edgesRai = ArrayImgs.unsignedLongs(2L, edgeCount.toLong())
                val edgesRaiCursor = Views.flatIterable(edgesRai).cursor()
                edges.forEachEntry { e1, value -> value.forEach { edgesRaiCursor.next().setInteger(e1); edgesRaiCursor.next().setInteger(it); true }; true }

                N5Utils.save(edgesRai, edgesContainer, edgesDataset, intArrayOf(2, numEdgesPerBlock), GzipCompression())
            }
        }

        fun mergeFeaturesWithReduceByKey(
                sc: JavaSparkContext,
                n5io: () -> N5IO,
                vararg features: () -> DoubleStatisticsFeature<*>,
                blocksPerSuperBlock: IntArray? = null,
                numEdgesPerBlock: Int = 1 shl 16) {
            n5io().let { n5 ->
                val featureBlockAttributes = n5.featureBlockContainer.getDatasetAttributes(n5.featureBlockDataset)
                blocksPerSuperBlock?.let { require(featureBlockAttributes.numDimensions == it.size) { "Invalid blocks per super blocks ${Arrays.toString(blocksPerSuperBlock)} for ${featureBlockAttributes.numDimensions} dimensions" } }
                val superBlockSize = ((blocksPerSuperBlock
                        ?: IntArray(featureBlockAttributes.numDimensions) { 1 }) zip featureBlockAttributes.blockSize).map { it.first * it.second }.toIntArray()
                val edgeDatasetAttributes = n5.edgesContainer.getDatasetAttributes(n5.edgesDataset)
                val numFeaturesDoubles = features.map { it().packedSizeInDoubles() }.sum()
                val numEdges = edgeDatasetAttributes.dimensions[1]
                n5.mergedFeaturesContainer.createDataset(n5.mergedFeaturesDataset, longArrayOf(numFeaturesDoubles.toLong(), numEdges), intArrayOf(numFeaturesDoubles, numEdgesPerBlock), DataType.FLOAT64, GzipCompression())

                val blockSize = featureBlockAttributes.blockSize
                val edgeIndexLookupBC = sc.broadcast(Views.flatIterable(N5Utils.open<UnsignedLongType>(n5.edgesContainer, n5.edgesDataset)).cursor().let { edgeCursor ->
                    val edgeIndexLookup = TLongObjectHashMap<TLongLongHashMap>()
                    var index = 0L
                    while (edgeCursor.hasNext()) {
                        edgeIndexLookup.computeIfAbsent(edgeCursor.next().integerLong, { TLongLongHashMap() }).put(edgeCursor.next().integerLong, index)
                        ++index
                    }
                    edgeIndexLookup
                })

                val numFeaturesBytes = features.map { it().numBytes() }.sum()

                val superBlocks = Grids.collectAllContainedIntervals(featureBlockAttributes.dimensions, superBlockSize).map { asMinMaxInterval(it) }
                sc
                        .parallelize(superBlocks)
                        .map(ReadFeatureBlocksIntoMap(n5io, edgeIndexLookupBC, *features))
                        .flatMap(FlatMapEdgeFeatureMap(numEdgesPerBlock))
                        .mapToPair { entry -> Tuple2(entry.key, entry.value) }
                        .reduceByKey { statsMap1, statsMap2 ->
                            val statsMap = TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>()
                            statsMap1.forEachEntry { k, v -> statsMap.put(k, v.map { it.copy() }); true }
                            statsMap2.forEachEntry { k, v ->
                                val v1 = statsMap[k]
                                if (v1 == null)
                                    statsMap.put(k, v.map { it.copy() })
                                else
                                    (v1 zip v).forEach { it.first.plusUnsafeAssign(it.second) }
                                true
                            }
                            LOG.debug("Returning stats map {}", statsMap)
                            statsMap
                        }
                        .foreach(WriteEdgeFeatures(n5io, numEdgesPerBlock, numEdges, *features))
            }
        }


        @Deprecated(message = "This seems to be slow", replaceWith = ReplaceWith("mergeFeaturesWithReduceByKey(sc, n5io, features, numEdgesPerBlock)"))
        fun mergeFeatures(
                sc: JavaSparkContext,
                n5io: () -> N5IO,
                vararg features: () -> DoubleStatisticsFeature<*>,
                numEdgesPerBlock: Int = 1 shl 16) {

            n5io().let {
                val edgeDatasetAttributes = it.edgesContainer.getDatasetAttributes(it.edgesDataset)
                val numDoubles = features.map { it().packedSizeInDoubles() }.reduce { i1, i2 -> i1 + i2 }
                val numEdges = edgeDatasetAttributes.dimensions[1]
                it.mergedFeaturesContainer.createDataset(it.mergedFeaturesDataset, longArrayOf(numDoubles.toLong(), numEdges), intArrayOf(numDoubles, numEdgesPerBlock), DataType.FLOAT64, GzipCompression())
                val edgeFeatureBlocks = Grids
                        .collectAllContainedIntervals(longArrayOf(numDoubles.toLong(), numEdges), intArrayOf(numDoubles, numEdgesPerBlock))
                        .map { MinMaxInterval(Intervals.minAsLongArray(it), Intervals.maxAsLongArray(it)) }
                sc
                        .parallelize(edgeFeatureBlocks)
                        .foreach(MergeEdgeFeatures(n5io, numEdgesPerBlock, *features))
            }

        }
    }
}

fun main() {
    val readPath = "/nrs/saalfeld/hanslovskyp/experiments/quasi-isotropic-predictions/affinities-glia/neuron_ids_noglia/predictions/lauritzen/02/workspace.n5"
    val weightsDataset = "volumes/predictions/quasi-isotropic-predictions/affinities-glia/neuron_ids_noglia/0/142000/affinities-averaged"
    val labelsDataset = "volumes/predictions/quasi-isotropic-predictions/affinities-glia/neuron_ids_noglia/0/142000/watersheds/merge_threshold=0.75_seed_threshold=0.5/merged"
    val path = "${System.getProperty("user.home")}/.local/tmp/edge-features.n5"
//    val readPath = "${System.getProperty("user.home")}/.local/tmp/slice.n5"
//    val weightsDataset = "weights"
//    val labelsDataset = "labels"
//    val path = "${System.getProperty("user.home")}/.local/tmp/edge-features-smaller-block.n5"
    val attributes = N5FSReader(readPath).getDatasetAttributes(weightsDataset)
    val dims = attributes.dimensions

    N5FSWriter(path).createDataset("feature-blocks", dims, attributes.blockSize, DataType.INT8, GzipCompression())

    val features = arrayOf({Histogram(100,max=1.0001)})

    val conf = SparkConf().setAppName(MethodHandles.lookup().lookupClass().simpleName)
    val sc = JavaSparkContext(conf)
    val n5io = {N5IO(
            weightsContainer = N5FSReader(readPath),
            labelsContainer = N5FSReader(readPath),
            weightsDataset = weightsDataset,
            labelsDataset = labelsDataset,
            featureBlockContainer = N5FSWriter(path),
            featureBlockDataset = "feature-blocks",
            edgesDataset = "edges",
            mergedFeaturesDataset = "edge-features"
    )}
    val blocks = Grids.collectAllContainedIntervals(attributes.dimensions, attributes.blockSize.map { 3 * it }.toIntArray())
    val numEdgesPerBlock = 1 shl 12

    sc.use {
        BlockwiseEdgeFeatures.updateFeatureBlocks(it, n5io, blocks, dims, attributes.blockSize, *features)
        BlockwiseEdgeFeatures.findEdges(it, n5io, features.map { it().numBytes() }.sum(), 3, 3, 3, numEdgesPerBlock = numEdgesPerBlock)
        BlockwiseEdgeFeatures.mergeFeaturesWithReduceByKey(it, n5io, *features, blocksPerSuperBlock = intArrayOf(3, 3, 3), numEdgesPerBlock = numEdgesPerBlock)
    }

}

class ExtractEdges(
        val n5io: () -> N5IO,
        val numFeatureBytes: Int
): Function<MinMaxInterval, TLongObjectHashMap<TLongHashSet>> {

    val dimensions: LongArray
    val blockSize: IntArray

    init {
        n5io().let {
            val attrs = it.featureBlockContainer.getDatasetAttributes(it.featureBlockDataset)
            dimensions = attrs.dimensions
            blockSize = attrs.blockSize
        }
    }

    override fun call(superBlock: MinMaxInterval): TLongObjectHashMap<TLongHashSet> {
        return n5io().let {
            val edgesInBlock = TLongObjectHashMap<TLongHashSet>()
            val grid = CellGrid(dimensions, blockSize)
            val blockPos = superBlock.min().clone()
            val attributes = it.featureBlockContainer.getDatasetAttributes(it.featureBlockDataset)
            Grids.forEachOffset(superBlock.min(), superBlock.max(), blockSize) { block ->
                grid.getCellPosition(block, blockPos)
                val dataBlock = it.featureBlockContainer.readBlock(it.featureBlockDataset, attributes, blockPos) as ByteArrayDataBlock
                val buffer = ByteBuffer.wrap(dataBlock.data)
                while (buffer.hasRemaining()) {
                    edgesInBlock.computeIfAbsent(buffer.long) {TLongHashSet()}.add(buffer.long)
                    buffer.position(buffer.position() + numFeatureBytes)
                }
            }
            edgesInBlock
        }
    }

}

class ReadFeatureBlocksIntoMap(
        private val n5io: () -> N5IO,
        private val edgeIndexLookupBC: Broadcast<TLongObjectHashMap<TLongLongHashMap>>,
        private vararg val features: () -> DoubleStatisticsFeature<*>) : Function<MinMaxInterval, TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>> {

    private val numFeaturesBytes: Int = this.features.map { it().numBytes() }.sum()


    override fun call(superBlock: MinMaxInterval): TLongObjectHashMap<List<DoubleStatisticsFeature<*>>> {
        val blockPos = superBlock.min().clone()
        val edgeIndexLookup = edgeIndexLookupBC.value
        return n5io().let { n5 ->
            val attributes = n5.featureBlockContainer.getDatasetAttributes(n5.featureBlockDataset)
            LOG.debug("Attributes: {}", attributes)
            val grid = CellGrid(attributes.dimensions, attributes.blockSize)
            val edgeFeatureMap = TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>()
            LOG.debug("Iterating over all blocks of size {} in ({} {})", attributes.blockSize, superBlock.min(), superBlock.max())
            Grids.forEachOffset(superBlock.min(), superBlock.max(), attributes.blockSize) { block ->
                grid.getCellPosition(block, blockPos)
                val dataBlock = n5.featureBlockContainer.readBlock(n5.featureBlockDataset, attributes, blockPos) as ByteArrayDataBlock
                val buffer = ByteBuffer.wrap(dataBlock.data)
                while (buffer.hasRemaining()) {
                    val e1 = buffer.long
                    val e2 = buffer.long
                    LOG.debug("Looking up ({} {}) in {}", e1, e2, edgeIndexLookup)
                    val edgeIndex = edgeIndexLookup[e1][e2]
                    val deserializedFeatures = features.map { it().let {it.deserializeFrom(buffer); it} }
                    LOG.debug("De-serialized features into {} for edge ({} {}) in block {}", deserializedFeatures, e1, e2, block)
                    edgeFeatureMap.put(edgeIndex, deserializedFeatures)
                }
            }
            edgeFeatureMap
        }
    }

    companion object {
        private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())
    }

}

class FlatMapEdgeFeatureMap(private val numEdgesPerBlock: Int) : FlatMapFunction<TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>, Map.Entry<Long, TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>>> {
    override fun call(edgeFeatureMap: TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>): Iterator<Map.Entry<Long, TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>>> {
        val blockMinToEdgeFeatureMapping = HashMap<Long, TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>>()
        edgeFeatureMap.forEachEntry { edgeIndex, features ->
            val blockMin = edgeIndex / numEdgesPerBlock * numEdgesPerBlock
            val offsetWithinBlock = edgeIndex - blockMin
            blockMinToEdgeFeatureMapping.computeIfAbsent(blockMin) {TLongObjectHashMap()}.put(offsetWithinBlock, features)
            true
        }
        return blockMinToEdgeFeatureMapping.iterator()
    }


    companion object {
        private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())
    }


}

class WriteEdgeFeatures(
        private val n5io: () -> N5IO,
        private val numEdgesPerBlock: Int,
        private val numEdges: Long,
        vararg features: () -> DoubleStatisticsFeature<*>) : VoidFunction<Tuple2<Long, TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>>> {

    private val numFeaturesDoubles = features.map { it().packedSizeInDoubles() }.sum()

    override fun call(blockMinAndData: Tuple2<Long, TLongObjectHashMap<List<DoubleStatisticsFeature<*>>>>) {
        LOG.debug("Writing {} features for block position {}", blockMinAndData._2().size(), blockMinAndData._1())
        val blockMin = blockMinAndData._1()
        val blockMax = min(blockMin + numEdgesPerBlock, numEdges) - 1
        val featureMap = blockMinAndData._2()
        val featureRai = ArrayImgs.doubles(numFeaturesDoubles.toLong(), blockMax - blockMin + 1)
        LOG.debug("Created store {} for edge features of size {} {}", featureRai, numFeaturesDoubles, blockMax - blockMin + 1)
        require(featureMap.size().toLong() == blockMax - blockMin + 1) {"Expected ${blockMax - blockMin + 1} features but got ${featureMap.size()}"}
        val validRange = 0..featureRai.max(1)
        featureMap.forEachEntry { key, localFeatures ->
            require(key in validRange) {"Edge index $key not in valid range $validRange"};
            val target = Views.flatIterable(Views.hyperSlice(featureRai, 1, key)).cursor()
            localFeatures.forEach { it.pack().serializeInto(target) }
            LOG.debug("Serialized {} to doubles", localFeatures)
            true }
        n5io().let { N5Utils.saveBlock(featureRai, it.mergedFeaturesContainer, it.mergedFeaturesDataset, longArrayOf(0, blockMin / numEdgesPerBlock)) }
    }
}

class MergeEdgeFeatures(
        val n5io: () -> N5IO,
        val numEdgesPerBlock: Int,
        vararg val features: () -> DoubleStatisticsFeature<*>) : VoidFunction<MinMaxInterval> {

    val numDoubles: Int
    val numEdges: Long

    init {
        n5io().let {
            val edgeDatasetAttributes = it.edgesContainer.getDatasetAttributes(it.edgesDataset)
            numDoubles = features.map { it().packedSizeInDoubles() }.reduce { i1, i2 -> i1 + i2 }
            numEdges = edgeDatasetAttributes.dimensions[1]
        }
    }


    override fun call(edgeFeatureBlock: MinMaxInterval) {
        n5io().let {
            val relevantEdges    = Views.zeroMin(Views.interval(N5Utils.open<UnsignedLongType>(it.edgesContainer, it.edgesDataset), FinalInterval(longArrayOf(0, edgeFeatureBlock.min()[1]), longArrayOf(1, edgeFeatureBlock.max()[1]))))
            val edgeFeatures     = ArrayImgs.doubles(*Intervals.dimensionsAsLongArray(FinalInterval(edgeFeatureBlock.min(), edgeFeatureBlock.max())))
            val edgeFeaturesList = List(edgeFeatures.dimension(1).toInt()) {features.map { it() }}
            val featureBlockDatasetAttributes = it.featureBlockContainer.getDatasetAttributes(it.featureBlockDataset)
            val featureBlockGrid = CellGrid(featureBlockDatasetAttributes.dimensions, featureBlockDatasetAttributes.blockSize)
            LOG.debug("Feature block grid is {}", featureBlockGrid)


            Grids.forEachOffset(LongArray(featureBlockGrid.numDimensions(), {0}), featureBlockGrid.gridDimensions.map { it - 1 }.toLongArray(), IntArray(featureBlockGrid.numDimensions()) {1}) { blockPos ->
                LOG.debug("Loading block data for block position {}", blockPos)
                val blockData = it.featureBlockContainer.readBlock(it.featureBlockDataset, featureBlockDatasetAttributes, blockPos) as ByteArrayDataBlock
                val buffer = ByteBuffer.wrap(blockData.data)
                val featureMap = TLongObjectHashMap<TLongObjectMap<List<DoubleStatisticsFeature<*>>>>()
                while (buffer.hasRemaining()) {
                    val k1 = buffer.long
                    val k2 = buffer.long
                    featureMap.computeIfAbsent(k1) { TLongObjectHashMap() }.put(k2, features.map { val  f = it(); f.deserializeFrom(buffer); f })
                }

                for (index in edgeFeaturesList.indices) {

                    val e1 = relevantEdges[0L, index.toLong()].integerLong
                    featureMap[e1]?.let { m1 ->
                        val e2 = relevantEdges[1L, index.toLong()].integerLong
                        m1[e2]?.let { features ->
                            val featuresFor = edgeFeaturesList[index]
                            for (featureIndex in featuresFor.indices) {
                                featuresFor[featureIndex].plusUnsafeAssign(features[featureIndex])
                            }
                        }
                    }

                }
            }

            for (index in edgeFeaturesList.indices) {
                val featuresFor = edgeFeaturesList[index]
                val target = Views.flatIterable(Views.hyperSlice(edgeFeatures, 1, index.toLong())).cursor()
                LOG.debug("Packing features {}", featuresFor)
                featuresFor.forEach { f -> LOG.debug("Packing feature {}", f); f.pack().serializeInto(target) }
            }

            val targetAttributes = DatasetAttributes(longArrayOf(numDoubles.toLong(), numEdges), intArrayOf(numDoubles, numEdgesPerBlock), DataType.FLOAT64, GzipCompression())
            val targetGrid = CellGrid(targetAttributes.dimensions, targetAttributes.blockSize)
            val blockPos = edgeFeatureBlock.min().clone()
            targetGrid.getCellPosition(blockPos, blockPos)
            N5Utils.saveBlock(edgeFeatures, it.mergedFeaturesContainer, it.mergedFeaturesDataset, targetAttributes, blockPos)


        }
    }

    companion object {
        private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())
    }

}