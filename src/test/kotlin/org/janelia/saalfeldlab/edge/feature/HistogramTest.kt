package org.janelia.saalfeldlab.edge.feature

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

    companion object {

        private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())

    }

}