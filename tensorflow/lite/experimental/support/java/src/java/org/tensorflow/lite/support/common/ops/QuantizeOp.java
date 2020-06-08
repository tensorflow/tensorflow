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

package org.tensorflow.lite.support.common.ops;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * Quantizes a {@link TensorBuffer} with given {@code zeroPoint} and {@code scale}.
 *
 * <p>Note: {@link QuantizeOp} does not cast output to UINT8, but only performs the quantization
 * math on top of input. The data type of output tensor is always {@code FLOAT32} except that the Op
 * is effectively an identity Op (in this case, the output tensor is the same instance as the
 * input). To connect with quantized model, a {@link CastOp} is probably needed.
 *
 * <p>If both {@code zeroPoint} and {@code scale} are 0, the {@link QuantizeOp} will be bypassed,
 * which is equivalent to setting {@code zeroPoint} to 0 and {@code scale} to 1. This can be useful
 * when passing in the quantization parameters that are extracted directly from the TFLite model
 * flatbuffer. If the tensor is not quantized, both {@code zeroPoint} and {@code scale} will be read
 * as 0.
 */
public class QuantizeOp extends NormalizeOp implements TensorOperator {

  public QuantizeOp(float zeroPoint, float scale) {
    // Quantization: f = (q - z) * s, i.e. q = f / s + z = (f - (-z * s)) / s
    super(-zeroPoint * scale, scale);
  }
}
