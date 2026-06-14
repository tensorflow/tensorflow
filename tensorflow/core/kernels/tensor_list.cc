/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/tensor_list.h"

#include <limits>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/coding.h"

namespace tensorflow {

TensorList::~TensorList() {
  if (tensors_) tensors_->Unref();
}

void TensorList::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  std::vector<size_t> invalid_indices;
  for (size_t i = 0; i < tensors().size(); i++) {
    if (tensors().at(i).dtype() != DT_INVALID) {
      *data->add_tensors() = tensors().at(i);
    } else {
      invalid_indices.push_back(i);
    }
  }
  std::string metadata;
  // TODO(b/118838800): Add a proto for storing the metadata.
  // Metadata format:
  // <num_invalid_tensors><invalid_indices><element_dtype><element_shape_proto>
  core::PutVarint64(&metadata, static_cast<uint64_t>(invalid_indices.size()));
  for (size_t i : invalid_indices) {
    core::PutVarint64(&metadata, static_cast<uint64_t>(i));
  }
  core::PutVarint64(&metadata, static_cast<uint64_t>(element_dtype));
  core::PutVarint64(&metadata, static_cast<uint64_t>(max_num_elements));
  TensorShapeProto element_shape_proto;
  element_shape.AsProto(&element_shape_proto);
  element_shape_proto.AppendToString(&metadata);
  data->set_metadata(metadata);
}

bool TensorList::Decode(const VariantTensorData& data) {
  // TODO(srbs): Change the signature to Decode(VariantTensorData data) so
  // that we do not have to copy each tensor individually below. This would
  // require changing VariantTensorData::tensors() as well.
  std::string metadata;
  data.get_metadata(&metadata);
  uint64_t scratch;
  absl::string_view iter(metadata);
  if (!core::GetVarint64(&iter, &scratch)) return false;
  if (scratch > std::numeric_limits<size_t>::max()) return false;
  const size_t num_invalid_tensors = static_cast<size_t>(scratch);

  if (num_invalid_tensors >
      std::numeric_limits<size_t>::max() - data.tensors().size()) {
    return false;
  }
  const size_t total_num_tensors = data.tensors().size() + num_invalid_tensors;

  std::vector<Tensor> decoded_tensors;
  if (total_num_tensors > decoded_tensors.max_size()) return false;
  decoded_tensors.reserve(total_num_tensors);

  size_t output_index = 0;
  size_t tensor_index = 0;
  bool have_previous_invalid_index = false;
  size_t previous_invalid_index = 0;
  for (size_t i = 0; i < num_invalid_tensors; ++i) {
    if (!core::GetVarint64(&iter, &scratch)) return false;
    if (scratch > std::numeric_limits<size_t>::max()) return false;
    const size_t invalid_index = static_cast<size_t>(scratch);
    if (invalid_index >= total_num_tensors) return false;
    if (have_previous_invalid_index && invalid_index <= previous_invalid_index) {
      return false;
    }

    while (output_index < invalid_index) {
      if (tensor_index >= data.tensors().size()) return false;
      decoded_tensors.emplace_back(data.tensors()[tensor_index]);
      ++tensor_index;
      ++output_index;
    }
    decoded_tensors.emplace_back(Tensor(DT_INVALID));
    ++output_index;
    previous_invalid_index = invalid_index;
    have_previous_invalid_index = true;
  }

  while (output_index < total_num_tensors) {
    if (tensor_index >= data.tensors().size()) return false;
    decoded_tensors.emplace_back(data.tensors()[tensor_index]);
    ++tensor_index;
    ++output_index;
  }
  if (tensor_index != data.tensors().size()) return false;

  if (!core::GetVarint64(&iter, &scratch)) return false;
  if (scratch > std::numeric_limits<int>::max()) return false;
  const DataType decoded_element_dtype = static_cast<DataType>(scratch);

  if (!core::GetVarint64(&iter, &scratch)) return false;
  int decoded_max_num_elements;
  if (scratch == std::numeric_limits<uint64_t>::max()) {
    decoded_max_num_elements = -1;
  } else {
    if (scratch > std::numeric_limits<int>::max()) return false;
    decoded_max_num_elements = static_cast<int>(scratch);
  }

  TensorShapeProto element_shape_proto;
  if (!element_shape_proto.ParseFromString(iter)) return false;

  const PartialTensorShape decoded_element_shape(element_shape_proto);

  element_dtype = decoded_element_dtype;
  max_num_elements = decoded_max_num_elements;
  tensors() = std::move(decoded_tensors);
  element_shape = decoded_element_shape;
  return true;
}

const char TensorList::kTypeName[] = "tensorflow::TensorList";

}  // namespace tensorflow
