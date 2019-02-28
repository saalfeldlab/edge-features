package org.janelia.saalfeldlab.edge.feature

import gnu.trove.map.TLongObjectMap
import gnu.trove.map.hash.TLongObjectHashMap
import net.imglib2.Interval
import net.imglib2.RandomAccessible
import net.imglib2.type.numeric.IntegerType
import net.imglib2.type.numeric.RealType
import net.imglib2.util.Intervals
import net.imglib2.view.Views
import org.janelia.saalfeldlab.labels.Label
import org.scijava.plugin.SciJavaPlugin
import java.util.function.BiPredicate
import java.util.function.DoublePredicate

interface Feature<F: Feature<F>> : SciJavaPlugin {


    @Throws(IncompatibleFeaturesException::class) operator fun plus(other: F): F
    @Throws(IncompatibleFeaturesException::class) operator fun plusAssign(other:F)

    fun numDoubles(): Int
    fun <T: RealType<T>> serializeInto(target: Iterator<T>)

    companion object {
        fun <F: Feature<F>> combine(f1: F, f2: F): F {
            return f1 + f2
        }
    }

}

interface DoubleStatisticsFeature<F: DoubleStatisticsFeature<F>> : Feature<F> {

    fun addValue(value: Double)

    fun addValues(vararg values: Double)

    companion object {

        // corner pixels will be added twice like this. Maybe use neighborhoods instead, but: less efficient!
        fun <E: RealType<E>, L: IntegerType<L>> addAll(
                edges: RandomAccessible<E>,
                fragments: RandomAccessible<L>,
                block: Interval,
                isEdge: BiPredicate<L, L> = BiPredicate { t, u -> val tp = t.integerLong; val up = u.integerLong; tp != up && tp != Label.INVALID && tp != 0L && up != Label.INVALID && up != 0L},
                isValidWeight: DoublePredicate = DoublePredicate { !it.isNaN() },
                vararg features: () -> DoubleStatisticsFeature<*>
        ): TLongObjectMap<TLongObjectMap<List<DoubleStatisticsFeature<*>>>> {
            val nDim = edges.numDimensions()
            val fragmentBlock1 = Views.interval(fragments, block)
            val edgeBlock1 = Views.interval(edges, block)

            val edgeIndexToIndexToFeatureListMapping = TLongObjectHashMap<TLongObjectMap<List<DoubleStatisticsFeature<*>>>>()

            for (dim in 0 until nDim) {
                val fragmentBlock2 = Views.interval(fragments, Intervals.translate(block, 1L, dim))
                val edgeBlock2 = Views.interval(fragments, Intervals.translate(block, 1L, dim))

                val fc1 = Views.flatIterable(fragmentBlock1).cursor()
                val fc2 = Views.flatIterable(fragmentBlock2).cursor()
                val ec1 = Views.flatIterable(edgeBlock1).cursor()
                val ec2 = Views.flatIterable(edgeBlock2).cursor()

                while (fc1.hasNext()) {
                    var f1t = fc1.next()
                    var f2t = fc2.next()
                    val e1  = ec1.next().realDouble
                    val e2  = ec2.next().realDouble

                    if (!isEdge.test(f1t, f2t))
                        continue

                    var f1 = f1t.integerLong
                    var f2 = f2t.integerLong
                    if (f2 < f1) {
                        val tmp = f2
                        f2 = f1
                        f1 = tmp
                    }


                    val features = edgeIndexToIndexToFeatureListMapping.computeIfAbsent(f1) {TLongObjectHashMap()}.computeIfAbsent(f2) {features.map { it() }}
                    if (isValidWeight.test(e1)) features.forEach { it.addValue(e1) }
                    if (isValidWeight.test(e2)) features.forEach { it.addValue(e2) }
                }
            }

            return edgeIndexToIndexToFeatureListMapping

        }


        private inline fun <T> TLongObjectMap<T>.computeIfAbsent(key: Long, mappingFunction: (Long) -> T) = this[key] ?: putAndReturn(key, mappingFunction(key))

        private fun <T> TLongObjectMap<T>.putAndReturn(key: Long, value: T): T {
            this.put(key, value)
            return value
        }

    }

}

class IncompatibleFeaturesException(val f1: Feature<*>, val f2: Feature<*>, message: String? = "Features not compatible: $f1 and $f2") : Exception(message)