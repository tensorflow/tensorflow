/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_ADD_QUANT_ADJUSTMENTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_ADD_QUANT_ADJUSTMENTS_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"

namespace tflite {
namespace gpu {

// This pass is used to support inference on quantized models with the GPU
// delegate.
//
// When delegating quantized models, we still run float-point inference on GPU
// under-the-hood. This is done by dequantizing inputs (at runtime) & constants
// (during delegation).
// However, intermediate tensors can still deviate from the original quantized
// inference, since activations may not follow the attributes set by the
// original quantizion parameters.
// To prevent this, we add "QuantizeAndDequantize" nodes for each node-output
// that was originally fixed-point:
// op1 -> op2
// becomes
// op1 -> QuantizeAndDequantize -> op2
std::unique_ptr<NodeTransformation> NewAddQuantAdjustments();

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_ADD_QUANT_ADJUSTMENTS_H_
