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

#include "tensorflow/lite/delegates/gpu/metal/kernels/add.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetAddTableCodeFused(int src_count) {
  std::string code = "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid";
  for (int i = 0; i < src_count; ++i) {
    code += ", device FLT4* const src_buf" + std::to_string(i);
  }
  code += ") {\n";
  for (int i = 0; i < src_count; ++i) {
    code += "  value += src_buf" + std::to_string(i) + "[linear_index];\n";
    code += "  return value;\n";
  }
  code += "}\n";
  return code;
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> Add(int id,
                                          const std::vector<ValueId> input_ids,
                                          ValueId output_id,
                                          const AddAttributes& attr,
                                          const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;

  // Add scalar
  const float* add_value = absl::get_if<float>(&attr.param);
  if (add_value) {
    desc->is_linkable = true;
    desc->shader_source =
        R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid) {
      return value + )" +
        std::to_string(*add_value) + ";}";
    desc->input_buffers = {{input_ids[0]}};
    desc->output_buffer = {output_id};
    return {desc};
  }
  // Add vector
  auto broadcast = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
  if (broadcast) {
    desc->is_linkable = true;
    desc->shader_source =
        R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid,
      device FLT4* const broadcast) { return value + broadcast[gid.z]; })";
    desc->input_buffers = {{input_ids[0]}};
    desc->output_buffer = {output_id};
    desc->immutable_buffers = {
        {"device FLT4* const",
         GetByteBufferConverted(broadcast->data, options.storage_precision)},
    };
    return {desc};
  }

  desc->is_linkable = true;
  desc->is_associative_op = true;
  desc->shader_source = GetAddTableCodeFused(input_ids.size() - 1);

  for (int i = 0; i < input_ids.size(); ++i) {
    desc->input_buffers.push_back({input_ids[i], "device FLT4* const"});
  }
  desc->output_buffer = {output_id};

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
