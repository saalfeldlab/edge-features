package org.janelia.saalfeldlab.edge.feature

import net.imglib2.type.numeric.real.DoubleType
import net.imglib2.type.operators.ValueEquals
import org.apache.commons.lang3.builder.ToStringBuilder
import org.apache.commons.lang3.builder.ToStringStyle
import java.nio.ByteBuffer
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
        require(min.isFinite(), {"min must be final but got $min"})
        require(max.isFinite(), {"max must be final but got $max"})
        require(max > min, {"$max > $min"})
        require(nBins > 0, {"Need at least one bin in the histogram"})

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

    override fun plusAssign(other: Histogram) {
        if (this.min != other.min || this.max != other.max || this.nBins != other.nBins)
            throw IncompatibleFeaturesException(this, other, "Histograms disagree in min, max, or nBins: $this $other")
        other.bins.forEachIndexed { index, binCount -> this.bins[index] += binCount }
        this.count += other.count
        this.underflow += other.underflow
        this.overflow += other.overflow
    }
    override fun plusUnsafe(other: Feature<*>): Histogram {
        if (!(other is Histogram))
            throw IncompatibleFeaturesException(this, other, "Cannot add feature of type ${other::class} to Histogram: $this $other")
        return plus(other)
    }

    override fun plusUnsafeAssign(other: Feature<*>) {
        if (!(other is Histogram))
            throw IncompatibleFeaturesException(this, other, "Cannot add feature of type ${other::class} to Histogram: $this $other")
        plusAssign(other)
    }

    override fun toString(): String {
        return ToStringBuilder(this, ToStringStyle.SHORT_PREFIX_STYLE)
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

    override fun pack() = normalized()

    // 5: min, max, count, overflow, underflow
    override fun packedSizeInDoubles(): Int = nBins + 5

    // nbins, bins, min, max, count, underflow, overflow
    override fun serializeInto(target: ByteBuffer) {
        target.putInt(this.nBins)
        bins.forEach {target.putLong(it)}
        target.putDouble(min)
        target.putDouble(max)
        target.putLong(count)
        target.putLong(underflow)
        target.putLong(overflow)
    }

    // nbins, bins, min, max, count, underflow, overflow
    override fun deserializeFrom(source: ByteBuffer) {
        val nBins = source.int
        require(nBins == this.nBins) {"Inconsistent number of bins: $nBins != ${this.nBins}"}
        bins.indices.forEach { bins[it] = source.getLong() }
        val min = source.double
        val max = source.double
        require(min == this.min) {"Inconsistent min: $min != ${this.max}"}
        require(max == this.max) {"Inconsistent max: $max != ${this.max}"}
        count = source.getLong()
        underflow = source.getLong()
        overflow = source.getLong()
    }


    // pre-pend nbins
    override fun numBytes(): Int = packedSizeInDoubles() * java.lang.Double.BYTES + java.lang.Integer.BYTES



}

class NormalizedHistogram(
        val min: Double,
        val max: Double,
        private val normalizedValues: DoubleArray,
        val count: Long,
        val overflow: Double,
        val underflow: Double) : DoubleSerializable {

    operator fun get(index: Int): Double {
        return this.normalizedValues[index]
    }

    fun toDoubleArray(): DoubleArray = normalizedValues + doubleArrayOf(min, max, count.toDouble(), underflow, overflow)

    override fun numDoubles(): Int = normalizedValues.size + 5 // 5: min, max, count, overflow, underflow, cf: toDoubleArray().size

    override fun serializeInto(target: Iterator<DoubleType>) = toDoubleArray().forEach { target.next().setReal(it) }
}