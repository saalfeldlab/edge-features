package org.janelia.saalfeldlab.edge.feature

import net.imglib2.type.operators.ValueEquals
import org.apache.commons.lang3.builder.ToStringBuilder
import java.util.Arrays
import kotlin.math.floor

// TODO use experimental kotlin unsigned data types? Currently, no division defined between double and unsigned types
class Histogram @JvmOverloads constructor(
        val nBins: Int,
        val min: Double = 0.0,
        val max: Double = 1.0,
        vararg initialValues: Double) : DoubleStatisticsFeature<Histogram>, ValueEquals<Histogram> {

    private val bins: LongArray
    val range: Double
    val binWidth: Double
    val maxBinIndex: Int
    var count = 0L
    var overflow = 0L
    var underflow = 0L

    init {
        assert(min.isFinite(), {"min must be final but got $min"})
        assert(max.isFinite(), {"max must be final but got $max"})
        assert(max > min, {"$max > $min"})
        assert(nBins > 0, {"Need at least one bin in the histogram"})

        this.bins = LongArray(this.nBins) {0}
        this.range = this.max - this.min
        this.binWidth = this.range / this.nBins
        this.maxBinIndex = this.nBins - 1

        addValues(*initialValues)

    }

    fun copy(): Histogram {
        val that = Histogram(nBins = this.nBins, min = this.min, max = this.max)
        that.count = this.count
        that.underflow = this.underflow
        that.overflow = this.overflow
        System.arraycopy(this.bins, 0, that.bins, 0, this.bins.size)
        return that
    }

    private fun mapToBin(value: Double): Int {
        assert(value.isFinite(), {"Expected finite value but got $value"})
        // do min and max before mapping into int space to avoid overflows
        return          if (value < this.min)
        -1         else if (value >= this.max)
        this.nBins else
        floor((value - this.min) / this.binWidth).toInt()
    }


    override fun addValue(value: Double) {
        val binIndex = mapToBin(value)
        if (binIndex < 0) {
            ++this.bins[0]
            ++underflow
        } else if (binIndex > this.maxBinIndex) {
            ++this.bins[this.maxBinIndex]
            ++overflow
        } else
            ++this.bins[binIndex]
        ++count
    }

    override fun addValues(vararg values: Double) {
        values.forEach(::addValue)
    }

    override fun plus(that: Histogram): Histogram {
        if (this.min != that.min || this.max != that.max || this.nBins != that.nBins)
            throw IncompatibleFeaturesException(this, that, "Histograms disagree in min, max, or nBins: $this $that")
        val newHist = this.copy()
        newHist += that
        return newHist
    }

    override fun plusAssign(that: Histogram) {
        if (this.min != that.min || this.max != that.max || this.nBins != that.nBins)
            throw IncompatibleFeaturesException(this, that, "Histograms disagree in min, max, or nBins: $this $that")
        that.bins.forEachIndexed { index, binCount -> this.bins[index] += binCount }
        this.count += that.count
        this.underflow += that.underflow
        this.overflow += that.overflow
    }

    override fun toString(): String {
        return ToStringBuilder(this)
                .append("nBins", nBins)
                .append("min", min)
                .append("max", max)
                .append("count", count)
                .append("underflow", underflow)
                .append("overflow", overflow)
                .append("bins", Arrays.toString(bins))
                .toString()
    }

    operator fun get(index: Int): Long {
        return this.bins[index]
    }

    override fun valueEquals(that: Histogram): Boolean {
        return that.min == this.min
                && that.max == this.max
                && that.nBins == this.nBins
                && that.underflow == this.underflow
                && that.overflow == this.overflow
                && Arrays.equals(that.bins, this.bins)
    }

    override fun equals(other: Any?): Boolean = if (other is Histogram) this.valueEquals(other) else false

    fun normalized() = NormalizedHistogram(
            min = this.min,
            max = this.max,
            count=this.count,
            overflow = this.overflow.toDouble() / count,
            underflow = this.underflow.toDouble() / count,
            normalizedValues = normalizeCounts(this.count, this.bins))

    companion object {
        private fun normalizeCounts(count: Long, bins: LongArray) = DoubleArray(bins.size) {bins[it].toDouble() / count}
    }

    fun toDoubleArray() = normalized().toDoubleArray()

}

class NormalizedHistogram(
        val min: Double,
        val max: Double,
        private val normalizedValues: DoubleArray,
        val count: Long,
        val overflow: Double,
        val underflow: Double) {

    operator fun get(index: Int): Double {
        return this.normalizedValues[index]
    }

    fun toDoubleArray(): DoubleArray = normalizedValues + doubleArrayOf(min, max, count.toDouble(), underflow, overflow)
}