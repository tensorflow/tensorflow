package org.tensorflow.lite.examples.fastprocessor

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * FastImageProcessor
 *
 * Minimal, allocation-free (per-frame) preprocess/postprocess for full-HD FP32 models.
 * - Allocates a direct ByteBuffer/FloatBuffer once at construction and reuses it every frame.
 * - Uses `Bitmap.getPixels()` and one `Bitmap.setPixels()` call to avoid per-pixel Java calls.
 *
 * Usage:
 *  val proc = FastImageProcessor(width, height)
 *  proc.preprocess(inputBitmap)
 *  val input = proc.getInputBuffer() // pass to TFLite Interpreter
 *  // run inference, get output into FloatBuffer or FloatArray
 *  proc.postprocess(outputFloatBuffer, outputBitmap)
 */
class FastImageProcessor(
    private val width: Int,
    private val height: Int,
    private val channels: Int = 3
) {
    private val pixelCount = width * height
    private val byteSize = pixelCount * channels * 4

    // Direct buffer used for feeding interpreter. FloatBuffer view for easy puts/gets.
    private val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(byteSize).order(ByteOrder.nativeOrder())
    private val floatBuffer: FloatBuffer = byteBuffer.asFloatBuffer()

    // Reusable integer arrays for native pixel copy and output staging.
    private val intPixels = IntArray(pixelCount)
    private val outPixels = IntArray(pixelCount)

    /** Preprocess: copy from ARGB_8888 bitmap into our FloatBuffer in RGB order normalized to [0,1]. */
    fun preprocess(bitmap: Bitmap) {
        // Fast native copy from Bitmap into int array (ARGB)
        bitmap.getPixels(intPixels, 0, width, 0, 0, width, height)

        // Write normalized floats into the FloatBuffer (R, G, B order)
        floatBuffer.rewind()
        val inv255 = 1f / 255f
        var i = 0
        while (i < pixelCount) {
            val p = intPixels[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            floatBuffer.put(r * inv255)
            floatBuffer.put(g * inv255)
            floatBuffer.put(b * inv255)
            i++
        }
        floatBuffer.rewind()
    }

    /** Returns the ByteBuffer suitable for feeding into Interpreter (rewound). */
    fun getInputBuffer(): ByteBuffer {
        byteBuffer.rewind()
        return byteBuffer
    }

    /**
     * Postprocess: convert model float output (FloatBuffer with R,G,B floats in [0,1])
     * into an ARGB_8888 Bitmap via a single `setPixels` call.
     */
    fun postprocess(outputFloatBuffer: FloatBuffer, outBitmap: Bitmap) {
        outputFloatBuffer.rewind()
        var i = 0
        while (i < pixelCount) {
            val rf = (outputFloatBuffer.get() * 255f).toInt().coerceIn(0, 255)
            val gf = (outputFloatBuffer.get() * 255f).toInt().coerceIn(0, 255)
            val bf = (outputFloatBuffer.get() * 255f).toInt().coerceIn(0, 255)
            outPixels[i] = (0xFF shl 24) or (rf shl 16) or (gf shl 8) or bf
            i++
        }
        outBitmap.setPixels(outPixels, 0, width, 0, 0, width, height)
        outputFloatBuffer.rewind()
    }

    /** Helper overload: postprocess from a FloatArray (flat RGB floats). */
    fun postprocess(outputArray: FloatArray, outBitmap: Bitmap) {
        var idx = 0
        for (i in 0 until pixelCount) {
            val rf = (outputArray[idx++] * 255f).toInt().coerceIn(0, 255)
            val gf = (outputArray[idx++] * 255f).toInt().coerceIn(0, 255)
            val bf = (outputArray[idx++] * 255f).toInt().coerceIn(0, 255)
            outPixels[i] = (0xFF shl 24) or (rf shl 16) or (gf shl 8) or bf
        }
        outBitmap.setPixels(outPixels, 0, width, 0, 0, width, height)
    }

    /** Optional: direct access to internal FloatBuffer if needed. */
    fun getFloatBuffer(): FloatBuffer {
        floatBuffer.rewind()
        return floatBuffer
    }
}
