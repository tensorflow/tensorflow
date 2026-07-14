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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

// Performs the operation: {Quantize, Dequantize} on floating-point data.
// We need this operation to emulate the error introduced by quantization
// on the GPU, which cannot represent int8 tensors.
//
// Implemented as:
// qvalue = round((min(qmax, max(qmin, src_val)) - qmin) * (1/qscale))
// dq_value = qvalue * qscale + qmin
// Here, qmin, qmax & qscale refer to the quantization values as implemented in
// TensorFlow Lite's 'FakeQuant' kernel.
//
// NOTE: We do not need to nudge min/max values in this op, since they would
// already be adjusted while generating the quantized model.
std::unique_ptr<NodeShader> NewQuantizeAndDequantizeNodeShader();

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_QUANTIZE_AND_DEQUANTIZE_H_
