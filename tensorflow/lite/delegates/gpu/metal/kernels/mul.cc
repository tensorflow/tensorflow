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

#include "tensorflow/lite/delegates/gpu/metal/kernels/mul.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> Multiply(
    int id, ValueId input_id, ValueId output_id,
    const MultiplyScalarAttributes& attr, const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  auto multiplier = absl::get_if<float>(&attr.param);
  auto mul_buffer =
      absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
  const bool scalar = multiplier != nullptr;
  const std::string param_desc =
      scalar ? "float multiplier" : "device FLT4* const mul_buf";
  std::string code =
      "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid, ";
  code += param_desc + ") {\n";
  if (scalar) {
    code += "return value * multiplier;\n";
  } else {
    code += "return value * mul_buf[gid.z];\n";
  }
  code += "}\n";
  desc->shader_source = code;
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  if (scalar) {
    std::vector<uint8_t> multiplier_bits =
        GetByteBuffer(std::vector<float>{*multiplier});
    desc->uniform_buffers = {
        {"constant float&",
         [multiplier_bits](const std::map<ValueId, BHWC>& buffers) {
           return multiplier_bits;
         }},
    };
  } else {
    desc->immutable_buffers = {
        {"device FLT4* const",
         GetByteBufferConverted(mul_buffer->data, options.storage_precision)},
    };
  }
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
