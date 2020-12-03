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

#include "tensorflow/lite/delegates/gpu/metal/kernels/quantize_and_dequantize.h"

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
ComputeTaskDescriptor QuantizeAndDequantize(
    const OperationDef& definition,
    const QuantizeAndDequantizeAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;
  desc.shader_source = R"(
    FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid, float3 params) {
      value = clamp(value, FLT4(params.x), FLT4(params.y));
      value = (value - FLT4(params.x)) / FLT4(params.z);
      return round(value) * FLT4(params.z) + FLT4(params.x);
    }
  )";

  desc.AddSrcTensor("", definition.src_tensors[0]);
  desc.AddDstTensor("", definition.dst_tensors[0]);
  desc.uniform_buffers = {
      {"constant float3&",
       [attr](const std::vector<BHWC>& src_shapes,
              const std::vector<BHWC>& dst_shapes) {
         return GetByteBuffer(
             std::vector<float>{attr.min, attr.max, attr.scale});
       }},
  };
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
