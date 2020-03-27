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

import android.graphics.PointF;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.support.common.SupportPreconditions;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.image.ImageOperator;
import org.tensorflow.lite.support.image.TensorImage;

/**
 * The adapter that makes a TensorOperator able to run with TensorImage.
 *
 * @see org.tensorflow.lite.support.common.TensorOperator
 * @see org.tensorflow.lite.support.image.TensorImage
 */
public class TensorOperatorWrapper implements ImageOperator {

  private final TensorOperator tensorOp;

  /**
   * Wraps a {@link TensorOperator} object as an {@link ImageOperator}, so that the {@link
   * TensorOperator} could handle {@link TensorImage} objects by handling its underlying {@link
   * org.tensorflow.lite.support.tensorbuffer.TensorBuffer}.
   *
   * <p>Requirement: The {@code op} should not change coordinate system when applied on an image.
   *
   * @param op The created operator.
   */
  public TensorOperatorWrapper(TensorOperator op) {
    tensorOp = op;
  }

  @Override
  @NonNull
  public TensorImage apply(@NonNull TensorImage image) {
    SupportPreconditions.checkNotNull(image, "Op cannot apply on null image.");
    image.load(tensorOp.apply(image.getTensorBuffer()));
    return image;
  }

  @Override
  public int getOutputImageHeight(int inputImageHeight, int inputImageWidth) {
    return inputImageHeight;
  }

  @Override
  public int getOutputImageWidth(int inputImageHeight, int inputImageWidth) {
    return inputImageWidth;
  }

  @Override
  public PointF inverseTransform(PointF point, int inputImageHeight, int inputImageWidth) {
    return point;
  }
}
