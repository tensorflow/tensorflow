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

package org.tensorflow.lite.support.common;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * TensorProcessor is a helper class for preprocessing and postprocessing tensors. It could
 * transform a {@link TensorBuffer} to another by executing a chain of {@link TensorOperator}.
 *
 * <p>Example Usage:
 *
 * <pre>
 *   TensorProcessor processor = new TensorProcessor.Builder().add(new NormalizeOp(1, 2)).build();
 *   TensorBuffer anotherTensorBuffer = processor.process(tensorBuffer);
 * </pre>
 *
 * @see TensorProcessor.Builder to build a {@link TensorProcessor} instance.
 * @see TensorProcessor#process(TensorBuffer) to apply the processor on a {@link TensorBuffer}.
 */
public class TensorProcessor extends SequentialProcessor<TensorBuffer> {
  private TensorProcessor(Builder builder) {
    super(builder);
  }

  /** The Builder to create an {@link TensorProcessor}, which could be executed later. */
  public static class Builder extends SequentialProcessor.Builder<TensorBuffer> {

    /**
     * Creates a Builder to build {@link TensorProcessor}.
     *
     * @see #add(TensorOperator) to add an Op.
     * @see #build() to complete the building process and get a built Processor.
     */
    public Builder() {
      super();
    }

    /**
     * Adds an {@link TensorOperator} into the Operator chain.
     *
     * @param op the Operator instance to be executed then.
     */
    public TensorProcessor.Builder add(TensorOperator op) {
      super.add(op);
      return this;
    }

    /** Completes the building process and gets the {@link TensorProcessor} instance. */
    @Override
    public TensorProcessor build() {
      return new TensorProcessor(this);
    }
  }
}
