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
#include <vector>

#include <fp16.h>
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace metal {

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
  if (tensors_as_args) {
    src_tensors_names.push_back(tensor_name + "_buffer");
    auto desc_new = absl::make_unique<TensorDescriptor>(desc);
    args.AddObjectRef(tensor_name, AccessType::READ, std::move(desc_new));
  } else {
    src_tensors_names.push_back(tensor_name);
  }
}

void ComputeTaskDescriptor::AddDstTensor(const std::string& tensor_name,
                                         const TensorDescriptor& desc) {
  if (tensors_as_args) {
    dst_tensors_names.push_back(tensor_name + "_buffer");
    auto desc_new = absl::make_unique<TensorDescriptor>(desc);
    args.AddObjectRef(tensor_name, AccessType::WRITE, std::move(desc_new));
  } else {
    dst_tensors_names.push_back(tensor_name);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
