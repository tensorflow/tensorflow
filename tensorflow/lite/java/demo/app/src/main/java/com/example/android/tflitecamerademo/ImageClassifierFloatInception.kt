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

/**
 * This classifier works with the Inception-v3 slim model.
 * It applies floating point inference rather than using a quantized model.
 */
class ImageClassifierFloatInception
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
    private val labelProbArray = Array(1) { FloatArray(numLabels) }

    // you can download this file from
    // https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip
    override val modelPath: String
        get() = "inceptionv3_slim_2016.tflite"

    override val labelPath: String
        get() = "labels_imagenet_slim.txt"

    override val imageSizeX: Int
        get() = 299

    override val imageSizeY: Int
        get() = 299

    // a 32bit float value requires 4 bytes
    override val numBytesPerChannel: Int
        get() = 4

    override fun addPixelValue(pixelValue: Int) {
        imgData.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
        imgData.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
        imgData.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
    }

    override fun getProbability(labelIndex: Int): Float {
        return labelProbArray[0][labelIndex]
    }

    override fun setProbability(labelIndex: Int, value: Number) {
        labelProbArray[0][labelIndex] = value.toFloat()
    }

    override fun getNormalizedProbability(labelIndex: Int): Float {
        // TODO the following value isn't in [0,1] yet, but may be greater. Why?
        return getProbability(labelIndex)
    }

    override fun runInference() {
        tflite?.run(imgData, labelProbArray)
    }

    companion object {

        /**
         * The inception net requires additional normalization of the used input.
         */
        private const val IMAGE_MEAN = 128
        private const val IMAGE_STD = 128.0f
    }
}
