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
import android.graphics.PointF;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.support.image.ImageOperator;
import org.tensorflow.lite.support.image.TensorImage;

/**
 * As a computation unit for processing images, it can resize an image to user-specified size.
 *
 * <p>It interpolates pixels when image is stretched, and discards pixels when image is compressed.
 *
 * @see ResizeWithCropOrPadOp for resizing without content distortion.
 */
public class ResizeOp implements ImageOperator {

  /** Algorithms for resizing. */
  public enum ResizeMethod {
    BILINEAR,
    NEAREST_NEIGHBOR
  }

  private final int targetHeight;
  private final int targetWidth;
  private final boolean useBilinear;

  /**
   * Creates a ResizeOp which can resize images to specified size in specified method.
   *
   * @param targetHeight: The expected height of resized image.
   * @param targetWidth: The expected width of resized image.
   * @param resizeMethod: The algorithm to use for resizing. Options: {@link ResizeMethod}
   */
  public ResizeOp(int targetHeight, int targetWidth, ResizeMethod resizeMethod) {
    this.targetHeight = targetHeight;
    this.targetWidth = targetWidth;
    useBilinear = (resizeMethod == ResizeMethod.BILINEAR);
  }

  /**
   * Applies the defined resizing on given image and returns the result.
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
    Bitmap scaled =
        Bitmap.createScaledBitmap(image.getBitmap(), targetWidth, targetHeight, useBilinear);
    image.load(scaled);
    return image;
  }

  @Override
  public int getOutputImageHeight(int inputImageHeight, int inputImageWidth) {
    return targetHeight;
  }

  @Override
  public int getOutputImageWidth(int inputImageHeight, int inputImageWidth) {
    return targetWidth;
  }

  @Override
  public PointF inverseTransform(PointF point, int inputImageHeight, int inputImageWidth) {
    return new PointF(
        point.x * inputImageWidth / targetWidth, point.y * inputImageHeight / targetHeight);
  }
}
