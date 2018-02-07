/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package com.example.android.tflitecamerademo;

import android.app.Activity;

import java.io.IOException;
import java.nio.ByteBuffer;


public class ImageClassifierQuantizedMobileNet extends ImageClassifier {

    /**
     * Initializes an {@code ImageClassifier}.
     *
     * @param activity
     */
    ImageClassifierQuantizedMobileNet(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected String getModelPath() {
        // you can download this file from
        // https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip
        return "mobilenet_quant_v1_224.tflite";
    }

    @Override
    protected String getLabelPath() {
        return "labels_mobilenet_quant_v1_224.txt";
    }

    @Override
    protected int getImageSizeX() {
        return 224;
    }

    @Override
    protected int getImageSizeY() {
        return 224;
    }

    @Override
    protected Byte[][] createLabelProbArray(int numLabels) {
        return new Byte[0][numLabels];
    }

    @Override
    protected int getNumBytesPerChannel() {
        // the quantized model uses a single byte only
        return 1;
    }

    @Override
    protected void addPixelValue(int pixelValue, ByteBuffer imgData) {
        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
        imgData.put((byte) (pixelValue & 0xFF));
    }

    @Override
    protected float getProbability(int labelIndex) {
        return (((byte)labelProbArray[0][labelIndex] & 0xff) / 255.0f);
    }
}
