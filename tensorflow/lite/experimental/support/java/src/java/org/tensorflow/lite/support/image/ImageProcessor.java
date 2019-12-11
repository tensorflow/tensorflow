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

package org.tensorflow.lite.support.image;

import org.tensorflow.lite.support.common.SequentialProcessor;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.image.ops.TensorOperatorWrapper;

/**
 * ImageProcessor is a helper class for preprocessing and postprocessing {@link TensorImage}. It
 * could transform a {@link TensorImage} to another by executing a chain of {@link ImageOperator}.
 *
 * <p>Example Usage:
 *
 * <pre>
 *   ImageProcessor processor = new ImageProcessor.Builder()
 *       .add(new ResizeOp(224, 224, ResizeMethod.NEAREST_NEIGHBOR)
 *       .add(new Rot90Op())
 *       .add(new NormalizeOp(127.5f, 127.5f))
 *       .build();
 *   TensorImage anotherTensorImage = processor.process(tensorImage);
 * </pre>
 *
 * @see ImageProcessor.Builder to build a {@link ImageProcessor} instance.
 * @see ImageProcessor#process(TensorImage) to apply the processor on a {@link TensorImage}.
 */
public class ImageProcessor extends SequentialProcessor<TensorImage> {
  private ImageProcessor(Builder builder) {
    super(builder);
  }

  /**
   * The Builder to create an ImageProcessor, which could be executed later.
   *
   * @see #add(TensorOperator) to add a general TensorOperator.
   * @see #add(ImageOperator) to add an ImageOperator.
   * @see #build() complete the building process and get a built Processor.
   */
  public static class Builder extends SequentialProcessor.Builder<TensorImage> {
    public Builder() {
      super();
    }

    /**
     * Adds an {@link ImageOperator} into the Operator chain.
     *
     * @param op the Operator instance to be executed then.
     */
    public Builder add(ImageOperator op) {
      super.add(op);
      return this;
    }

    /**
     * Adds a {@link TensorOperator} into the Operator chain. In execution, the processor calls
     * {@link TensorImage#getTensorBuffer()} to transform the {@link TensorImage} by transforming
     * the underlying {@link org.tensorflow.lite.support.tensorbuffer.TensorBuffer}.
     *
     * @param op the Operator instance to be executed then.
     */
    public Builder add(TensorOperator op) {
      return add(new TensorOperatorWrapper(op));
    }

    /** Completes the building process and gets the {@link ImageProcessor} instance. */
    @Override
    public ImageProcessor build() {
      return new ImageProcessor(this);
    }

  }
}
