package org.janelia.saalfeldlab.edge.feature

interface Feature<F: Feature<F>> {


    @Throws(IncompatibleFeaturesException::class) operator fun plus(other: F): F
    @Throws(IncompatibleFeaturesException::class) operator fun plusAssign(other:F)

    companion object {
        fun <F: Feature<F>> combine(f1: F, f2: F): F {
            return f1 + f2
        }
    }

}

interface DoubleStatisticsFeature<F: DoubleStatisticsFeature<F>> : Feature<F> {

    fun addValue(value: Double)

    fun addValues(vararg values: Double)

}

class IncompatibleFeaturesException(val f1: Feature<*>, val f2: Feature<*>, message: String? = "Features not compatible: $f1 and $f2") : Exception(message)