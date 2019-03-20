/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.tflitecamerademo

import android.content.Context
import java.io.IOException
import kotlin.experimental.and

/**
 * This classifier works with the quantized MobileNet model.
 */
class ImageClassifierQuantizedMobileNet
/**
 * Initializes an `ImageClassifier`.
 *
 * @param context
 */
@Throws(IOException::class)
internal constructor(context: Context) : ImageClassifier(context) {

    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
     * This isn't part of the super class, because we need a primitive array here.
     */
    private val labelProbArray = Array(1) { ByteArray(numLabels) }

    // you can download this file from
    // see build.gradle for where to obtain this file. It should be auto
    // downloaded into assets.
    override val modelPath: String
        get() = "mobilenet_v1_1.0_224_quant.tflite"

    override val labelPath: String
        get() = "labels_mobilenet_quant_v1_224.txt"

    override val imageSizeX: Int
        get() = 224

    override val imageSizeY: Int
        get() = 224

    override val numBytesPerChannel: Int
        // the quantized model uses a single byte only
        get() = 1

    override fun addPixelValue(pixelValue: Int) {
        imgData.put((pixelValue shr 16 and 0xFF).toByte())
        imgData.put((pixelValue shr 8 and 0xFF).toByte())
        imgData.put((pixelValue and 0xFF).toByte())
    }

    override fun getProbability(labelIndex: Int): Float {
        return labelProbArray[0][labelIndex].toFloat()
    }

    override fun setProbability(labelIndex: Int, value: Number) {
        labelProbArray[0][labelIndex] = value.toByte()
    }

    override fun getNormalizedProbability(labelIndex: Int): Float {
        return (labelProbArray[0][labelIndex] and 0xff.toByte()) / 255.0f
    }

    override fun runInference() {
        tflite?.run(imgData, labelProbArray)
    }
}
