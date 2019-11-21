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

package org.tensorflow.lite.support.image.ops;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.support.image.ImageOperator;
import org.tensorflow.lite.support.image.TensorImage;

/** Rotates image counter-clockwise. */
public class Rot90Op implements ImageOperator {

  private final int numRotation;

  /** Creates a Rot90 Op which will rotate image by 90 degree counter-clockwise. */
  public Rot90Op() {
    this(1);
  }

  /**
   * Creates a Rot90 Op which will rotate image by 90 degree for {@code k} times counter-clockwise.
   *
   * @param k: The number of times the image is rotated by 90 degrees.
   */
  public Rot90Op(int k) {
    numRotation = k % 4;
  }

  /**
   * Applies the defined rotation on given image and returns the result.
   *
   * <p>Note: the content of input {@code image} will change, and {@code image} is the same instance
   * with the output.
   *
   * @param image input image.
   * @return output image.
   */
  @NonNull
  @Override
  public TensorImage apply(@NonNull TensorImage image) {
    Bitmap input = image.getBitmap();
    if (numRotation == 0) {
      return image;
    }
    int w = input.getWidth();
    int h = input.getHeight();
    Matrix matrix = new Matrix();
    matrix.postTranslate(w * 0.5f, h * 0.5f);
    matrix.postRotate(-90 * numRotation);
    int newW = (numRotation % 2 == 0) ? w : h;
    int newH = (numRotation % 2 == 0) ? h : w;
    matrix.postTranslate(newW * 0.5f, newH * 0.5f);
    Bitmap output = Bitmap.createBitmap(input, 0, 0, w, h, matrix, false);
    image.load(output);
    return image;
  }
}
