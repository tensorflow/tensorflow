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

#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

#include <cstdint>
#include <string>
#include <vector>

#include <fp16.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
std::string GetElementWiseCode(const OperationDef& op_def) {
  return R"(
kernel void ComputeFunction($0
                            uint3 gid[[thread_position_in_grid]]) {
  int X = static_cast<int>(gid.x);
  int Y = static_cast<int>(gid.y);
  int Z = static_cast<int>(gid.z);
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) {
    return;
  }
  FLT4 value = args.src_tensor.Read(X, Y, Z);
  args.dst_tensor.Write(value, X, Y, Z);
}
)";
}

}  // namespace

/// Converts float to destination type (if needed) and stores as bytes array.
std::vector<uint8_t> GetByteBufferConverted(
    const std::vector<float>& input_vector, DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return GetByteBuffer(input_vector);
  } else {
    std::vector<uint8_t> result;
    result.reserve(input_vector.size() * sizeof(HalfBits));
    for (const float value : input_vector) {
      const HalfBits converted = fp16_ieee_from_fp32_value(value);
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&converted);
      result.insert(result.end(), bytes, bytes + sizeof(HalfBits));
    }
    return result;
  }
}

/// Resizes, Converts float to destination type (if needed) and stores as bytes
/// array.
std::vector<uint8_t> GetByteBufferConvertedResized(
    const std::vector<float>& input_vector, DataType data_type,
    size_t elements_count) {
  auto result = GetByteBufferConverted(input_vector, data_type);
  const size_t type_size =
      data_type == DataType::FLOAT32 ? sizeof(float) : sizeof(HalfBits);
  result.resize(type_size * elements_count);
  return result;
}

ComputeTaskDescriptor::ComputeTaskDescriptor(const OperationDef& def)
    : definition(def) {}

void ComputeTaskDescriptor::AddSrcTensor(const std::string& tensor_name,
                                         const TensorDescriptor& desc) {
  src_tensors_names.push_back(tensor_name);
  auto desc_new = absl::make_unique<TensorDescriptor>(desc);
  args.AddObjectRef(tensor_name, AccessType::READ, std::move(desc_new));
}

void ComputeTaskDescriptor::AddDstTensor(const std::string& tensor_name,
                                         const TensorDescriptor& desc) {
  dst_tensors_names.push_back(tensor_name);
  auto desc_new = absl::make_unique<TensorDescriptor>(desc);
  args.AddObjectRef(tensor_name, AccessType::WRITE, std::move(desc_new));
}

absl::Status ComputeTaskDescriptor::AddTask(ComputeTaskDescriptor* task_desc) {
  linkable_count += 1;
  std::string code = task_desc->shader_source;
  std::string unique_postfix = absl::StrCat("_link", linkable_count);
  task_desc->args.RenameArgs(unique_postfix, &code);
  elementwise_code += "{\n" + code + "\n}\n";
  RETURN_IF_ERROR(args.Merge(std::move(task_desc->args), unique_postfix));
  for (int i = 0; i < task_desc->src_tensors_names.size(); ++i) {
    definition.src_tensors.push_back(task_desc->definition.src_tensors[i + 1]);
    src_tensors_names.push_back(task_desc->src_tensors_names[i] +
                                unique_postfix);
  }
  for (int i = 0; i < task_desc->dst_tensors_names.size(); ++i) {
    dst_tensors_names.push_back(task_desc->dst_tensors_names[i] +
                                unique_postfix);
  }
  return absl::OkStatus();
}

void ComputeTaskDescriptor::AssembleCode() {
  if (is_linkable) {
    auto src_desc =
        absl::make_unique<TensorDescriptor>(definition.src_tensors[0]);
    if (definition.IsBatchSupported()) {
      src_desc->SetStateVar("BatchedWidth", "true");
    }
    src_tensors_names.insert(src_tensors_names.begin(), "src_tensor");
    args.AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));

    auto dst_desc =
        absl::make_unique<TensorDescriptor>(definition.dst_tensors[0]);
    if (definition.IsBatchSupported()) {
      dst_desc->SetStateVar("BatchedWidth", "true");
    }
    dst_tensors_names.insert(dst_tensors_names.begin(), "dst_tensor");
    args.AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));

    elementwise_code = "{\n" + shader_source + "\n}\n" + elementwise_code;
    shader_source = GetElementWiseCode(definition);

    resize_function = [](const std::vector<BHWC>& src_shapes,
                         const std::vector<BHWC>& dst_shapes) {
      uint3 groups_size{8, 8, 1};
      uint3 groups_count{DivideRoundUp(dst_shapes[0].w, groups_size.x),
                         DivideRoundUp(dst_shapes[0].h, groups_size.y),
                         DivideRoundUp(dst_shapes[0].c, 4)};
      return std::make_pair(groups_size, groups_count);
    };
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
