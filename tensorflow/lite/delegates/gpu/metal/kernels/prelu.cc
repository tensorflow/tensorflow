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

#include "tensorflow/lite/delegates/gpu/metal/kernels/prelu.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
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

std::vector<ComputeTaskDescriptorPtr> PReLU(int id, ValueId input_id,
                                            ValueId output_id,
                                            const PReLUAttributes& attr,
                                            const RuntimeOptions& options) {
  auto alpha_buffer =
      absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha_buffer) {
    return {};
  }
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  if (attr.clip != 0) {
    desc->shader_source =
        R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid,
      device FLT4* const alphas, float clip) {
        return FLT4(clamp(value, FLT4(0.0f), FLT4(clip)) + alphas[gid.z] * min(FLT4(0.0f), value));
    })";
  } else {
    desc->shader_source =
        R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid,
      device FLT4* const alphas) {
        return FLT4(max(FLT4(0.0f), value) + alphas[gid.z] * min(FLT4(0.0f), value));
    })";
  }
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  desc->immutable_buffers = {
      {"device FLT4* const",
       GetByteBufferConverted(alpha_buffer->data, options.storage_precision)},
  };
  if (attr.clip != 0) {
    desc->uniform_buffers = {
        {"constant float&",
         [attr](const std::map<ValueId, BHWC>& buffers) {
           std::vector<uint8_t> attr_clip =
               GetByteBuffer(std::vector<float>{attr.clip});
           return attr_clip;
         }},
    };
  }
  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> PReLUFull(int id, ValueId input_id,
                                                ValueId output_id,
                                                const PReLUAttributes& attr,
                                                const RuntimeOptions& options) {
  auto alpha = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha) {
    return {};
  }
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  if (attr.clip != 0) {
    desc->shader_source =
        R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid,
      device FLT4* const alphas, float clip) {
        return FLT4(clamp(value, FLT4(0.0f), FLT4(clip)) + alphas[linear_index] * min(FLT4(0.0f), value));
    })";
  } else {
    desc->shader_source =
        R"(FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid,
      device FLT4* const alphas) {
        return FLT4(max(FLT4(0.0f), value) + alphas[linear_index] * min(FLT4(0.0f), value));
    })";
  }
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  desc->immutable_buffers = {
      {"device FLT4* const", GetByteBufferConverted(ConvertToPHWC4(*alpha),
                                                    options.storage_precision)},
  };
  if (attr.clip != 0) {
    desc->uniform_buffers = {
        {"constant float&",
         [attr](const std::map<ValueId, BHWC>& buffers) {
           std::vector<uint8_t> attr_clip =
               GetByteBuffer(std::vector<float>{attr.clip});
           return attr_clip;
         }},
    };
  }
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
