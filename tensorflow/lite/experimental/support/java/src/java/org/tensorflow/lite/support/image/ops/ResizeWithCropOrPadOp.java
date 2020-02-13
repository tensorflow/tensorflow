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
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Rect;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.support.image.ImageOperator;
import org.tensorflow.lite.support.image.TensorImage;

/**
 * As a computation unit for processing images, it could resize image to predefined size.
 *
 * <p>It will not stretch or compress the content of image. However, to fit the new size, it crops
 * or pads pixels. When it crops image, it performs a center-crop; when it pads pixels, it performs
 * a zero-padding.
 *
 * @see ResizeOp for reszing images while stretching / compressing the content.
 */
public class ResizeWithCropOrPadOp implements ImageOperator {
  private final int targetHeight;
  private final int targetWidth;
  private final Bitmap output;

  /**
   * Creates a ResizeWithCropOrPadOp which could crop/pad images to specified size. It adopts
   * center-crop and zero-padding.
   *
   * @param targetHeight: The expected height of cropped/padded image.
   * @param targetWidth: The expected width of cropped/padded image.
   */
  public ResizeWithCropOrPadOp(int targetHeight, int targetWidth) {
    this.targetHeight = targetHeight;
    this.targetWidth = targetWidth;
    output = Bitmap.createBitmap(this.targetWidth, this.targetHeight, Config.ARGB_8888);
  }

  /**
   * Applies the defined resizing with cropping or/and padding on given image and returns the
   * result.
   *
   * <p>Note: the content of input {@code image} will change, and {@code image} is the same instance
   * with the output.
   *
   * @param image input image.
   * @return output image.
   */
  @Override
  @NonNull
  public TensorImage apply(@NonNull TensorImage image) {
    Bitmap input = image.getBitmap();
    int srcL;
    int srcR;
    int srcT;
    int srcB;
    int dstL;
    int dstR;
    int dstT;
    int dstB;
    int w = input.getWidth();
    int h = input.getHeight();
    if (targetWidth > w) { // padding
      srcL = 0;
      srcR = w;
      dstL = (targetWidth - w) / 2;
      dstR = dstL + w;
    } else { // cropping
      dstL = 0;
      dstR = targetWidth;
      srcL = (w - targetWidth) / 2;
      srcR = srcL + targetWidth;
    }
    if (targetHeight > h) { // padding
      srcT = 0;
      srcB = h;
      dstT = (targetHeight - h) / 2;
      dstB = dstT + h;
    } else { // cropping
      dstT = 0;
      dstB = targetHeight;
      srcT = (h - targetHeight) / 2;
      srcB = srcT + targetHeight;
    }
    Rect src = new Rect(srcL, srcT, srcR, srcB);
    Rect dst = new Rect(dstL, dstT, dstR, dstB);
    new Canvas(output).drawBitmap(input, src, dst, null);
    image.load(output);
    return image;
  }
}
