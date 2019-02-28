package org.janelia.saalfeldlab.util

import org.apache.commons.io.FileUtils
import org.janelia.saalfeldlab.n5.DataType
import org.janelia.saalfeldlab.n5.DatasetAttributes
import org.janelia.saalfeldlab.n5.N5FSWriter
import org.janelia.saalfeldlab.n5.RawCompression
import org.slf4j.LoggerFactory
import pl.touk.throwing.ThrowingRunnable
import java.io.IOException
import java.lang.invoke.MethodHandles
import java.nio.file.Files
import java.nio.file.Path

class N5TestUtil {

    companion object {
        private val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())


        @Throws(IOException::class)
        @JvmOverloads
        @JvmStatic
        fun tmpDir(deleteOnExit: Boolean = true): Path {
            val tmp = Files.createTempDirectory(null)

            LOG.debug("Creating tmp dir at {} (delete on exit? {})", tmp, deleteOnExit)

            val dir = tmp.toFile()
            if (deleteOnExit) {
                dir.deleteOnExit()
                Runtime.getRuntime().addShutdownHook(Thread(ThrowingRunnable.unchecked<IOException> { FileUtils.deleteDirectory(dir) }))
            }
            return tmp
        }

        @Throws(IOException::class)
        @JvmOverloads
        @JvmStatic
        fun fileSystemWriterAtTmpDir(deleteOnExit: Boolean = true): N5FSWriter {
            val tmp = tmpDir(deleteOnExit = deleteOnExit)

            LOG.debug("Creating temporary N5Writer at {} (delete on exit? {})", tmp, deleteOnExit)
            return N5FSWriter(tmp.toAbsolutePath().toString())
        }

        @JvmOverloads
        internal fun defaultAttributes(t: DataType = DataType.UINT8): DatasetAttributes {
            return DatasetAttributes(longArrayOf(1), intArrayOf(1), t, RawCompression())
        }
    }
}
