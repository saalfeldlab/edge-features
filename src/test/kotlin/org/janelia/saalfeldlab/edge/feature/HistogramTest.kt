package org.janelia.saalfeldlab.edge.feature

import gnu.trove.set.hash.TLongHashSet
import net.imglib2.FinalInterval
import net.imglib2.img.array.ArrayImgs
import net.imglib2.type.numeric.integer.UnsignedLongType
import net.imglib2.type.numeric.real.FloatType
import net.imglib2.util.Intervals
import net.imglib2.view.Views
import org.junit.Assert
import org.junit.Test
import org.slf4j.LoggerFactory
import java.lang.invoke.MethodHandles

class HistogramTest {

    @Test
    fun testHistogram() {

        val data = doubleArrayOf(-0.9, -0.8, 0.3999999, 0.7, 0.8, 1.0)
        val histogram = Histogram(nBins = 4, min = -0.8, max = 0.8, initialValues = *data)
        LOG.debug("Histogram: {}", histogram)

        Assert.assertEquals(1, histogram.underflow)
        Assert.assertEquals(2, histogram.overflow)
        Assert.assertEquals(data.size.toLong(), histogram.count)
        Assert.assertEquals(2, histogram[0])
        Assert.assertEquals(0, histogram[1])
        Assert.assertEquals(1, histogram[2])
        Assert.assertEquals(3, histogram[3])

        Assert.assertEquals(4 + 5, histogram.packedSizeInDoubles())
        val target = ArrayImgs.doubles(histogram.packedSizeInDoubles().toLong(), 1L)
        histogram.pack().serializeInto(Views.flatIterable(Views.hyperSlice(target, 1, 0L)).cursor())
        // bins, min, max, count, overflow, underflow
        Assert.assertArrayEquals(doubleArrayOf(2.0 / 6.0, 0.0, 1.0 / 6.0, 3.0 / 6.0, -0.8, 0.8, 6.0, 1.0 / 6.0, 2.0 / 6.0), target.update(null).currentStorageArray, 0.0)

        val deserialized = Histogram(nBins = 4, min = -0.8, max = 0.8)
        Assert.assertNotEquals(histogram, deserialized)
        deserialized.deserializeFrom(histogram.serializeToByteBuffer())
        Assert.assertEquals(histogram, deserialized)

        val copy = histogram.copy()
        Assert.assertEquals(histogram, copy)

        val sum = histogram + histogram
        // check that histogram remains unchanged
        Assert.assertEquals(copy, histogram)
        Assert.assertNotEquals(copy, sum)
        Assert.assertNotEquals(histogram, sum)
        Assert.assertEquals(2, sum.underflow)
        Assert.assertEquals(4, sum.overflow)
        Assert.assertEquals(2 * data.size.toLong(), sum.count)
        Assert.assertEquals(4, sum[0])
        Assert.assertEquals(0, sum[1])
        Assert.assertEquals(2, sum[2])
        Assert.assertEquals(6, sum[3])


        copy += histogram
        Assert.assertNotEquals(histogram, copy)
        Assert.assertEquals(sum, copy)

        try {
            histogram[-1]
            Assert.fail("Out of range bin access did not fail.")
        } catch (e: ArrayIndexOutOfBoundsException) {
        }

        try {
            histogram[4]
            Assert.fail("Out of range bin access did not fail.")
        } catch (e: ArrayIndexOutOfBoundsException) {
        }

        val normalized = histogram.normalized()

        Assert.assertEquals(1.0 / data.size, normalized.underflow, 0.0)
        Assert.assertEquals(2.0 / data.size, normalized.overflow, 0.0)
        Assert.assertEquals(data.size.toLong(), normalized.count)
        Assert.assertEquals(2.0 / data.size, normalized[0], 0.0)
        Assert.assertEquals(0.0, normalized[1], 0.0)
        Assert.assertEquals(1.0 / data.size, normalized[2], 0.0)
        Assert.assertEquals(3.0 / data.size, normalized[3], 0.0)

        try {
            normalized[-1]
            Assert.fail("Out of range bin access did not fail.")
        } catch (e: ArrayIndexOutOfBoundsException) {
        }

        try {
            normalized[4]
            Assert.fail("Out of range bin access did not fail.")
        } catch (e: ArrayIndexOutOfBoundsException) {
        }

    }

    @Test
    fun testFromImage() {
        val dims = longArrayOf(10, 20)
        val labels = ArrayImgs.unsignedLongs(*dims)
        val weights = ArrayImgs.floats(*dims)
        Views.interval(labels, FinalInterval(10, 10)).forEach(UnsignedLongType::setOne)
        Views.interval(labels, Intervals.translate(FinalInterval(10, 10), 0, 10L)).forEach {it.setInteger(3L)}
        Views.interval(labels, Intervals.createMinMax(3, 4, 6, 12)).forEach {it.setInteger(2L)}
        weights.forEach(FloatType::setOne)

        val features = DoubleStatisticsFeature.addAll(
                edges = Views.extendValue(weights, FloatType(Float.NaN)),
                fragments = Views.extendZero(labels),
                features = *arrayOf({Histogram(nBins = 10)}),
                block = labels)

        LOG.debug("Got features {}", features)
        Assert.assertEquals(2, features.size())
        Assert.assertEquals(TLongHashSet(longArrayOf(1, 2)), features.keySet())
        Assert.assertEquals(TLongHashSet(longArrayOf(2, 3)), features[1].keySet())
        Assert.assertEquals(TLongHashSet(longArrayOf(3)), features[2].keySet())

        Assert.assertEquals(1, features[1][2].size)
        Assert.assertEquals(1, features[1][3].size)
        Assert.assertEquals(1, features[2][3].size)

        Assert.assertEquals(Histogram::class, features[1][2][0]::class)
        Assert.assertEquals(Histogram::class, features[1][3][0]::class)
        Assert.assertEquals(Histogram::class, features[2][3][0]::class)

        // check only counts to make sure that all relevant values are added
        // corner pixels will be added twice like this. Maybe use neighborhoods instead, but: less efficient!
        Assert.assertEquals(32L, (features[1][2][0] as Histogram).count)
        Assert.assertEquals(12L, (features[1][3][0] as Histogram).count)
        Assert.assertEquals(20L, (features[2][3][0] as Histogram).count)


    }

    companion object {

        private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())

    }

}