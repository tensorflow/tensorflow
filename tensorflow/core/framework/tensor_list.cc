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
#include "tensorflow/core/framework/tensor_list.h"

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
  string metadata;
  // TODO(b/118838800): Add a proto for storing the metadata.
  // Metadata format:
  // <num_invalid_tensors><invalid_indices><element_dtype><element_shape_proto>
  core::PutVarint64(&metadata, static_cast<uint64>(invalid_indices.size()));
  for (size_t i : invalid_indices) {
    core::PutVarint64(&metadata, static_cast<uint64>(i));
  }
  core::PutVarint64(&metadata, static_cast<uint64>(element_dtype));
  core::PutVarint64(&metadata, static_cast<uint64>(max_num_elements));
  TensorShapeProto element_shape_proto;
  element_shape.AsProto(&element_shape_proto);
  element_shape_proto.AppendToString(&metadata);
  data->set_metadata(metadata);
}

bool TensorList::Decode(const VariantTensorData& data) {
  // TODO(srbs): Change the signature to Decode(VariantTensorData data) so
  // that we do not have to copy each tensor individually below. This would
  // require changing VariantTensorData::tensors() as well.
  string metadata;
  data.get_metadata(&metadata);
  uint64 scratch;
  StringPiece iter(metadata);
  std::vector<size_t> invalid_indices;
  core::GetVarint64(&iter, &scratch);
  size_t num_invalid_tensors = static_cast<size_t>(scratch);
  invalid_indices.resize(num_invalid_tensors);
  for (size_t i = 0; i < num_invalid_tensors; i++) {
    core::GetVarint64(&iter, &scratch);
    invalid_indices[i] = static_cast<size_t>(scratch);
  }

  size_t total_num_tensors = data.tensors().size() + num_invalid_tensors;
  tensors().reserve(total_num_tensors);
  std::vector<size_t>::iterator invalid_indices_it = invalid_indices.begin();
  std::vector<Tensor>::const_iterator tensors_it = data.tensors().begin();
  for (size_t i = 0; i < total_num_tensors; i++) {
    if (invalid_indices_it != invalid_indices.end() &&
        *invalid_indices_it == i) {
      tensors().emplace_back(Tensor(DT_INVALID));
      invalid_indices_it++;
    } else if (tensors_it != data.tensors().end()) {
      tensors().emplace_back(*tensors_it);
      tensors_it++;
    } else {
      // VariantTensorData is corrupted.
      return false;
    }
  }

  core::GetVarint64(&iter, &scratch);
  element_dtype = static_cast<DataType>(scratch);
  core::GetVarint64(&iter, &scratch);
  max_num_elements = static_cast<int>(scratch);
  TensorShapeProto element_shape_proto;
  element_shape_proto.ParseFromString(string(iter.data(), iter.size()));
  element_shape = PartialTensorShape(element_shape_proto);
  return true;
}

const char TensorList::kTypeName[] = "tensorflow::TensorList";

}  // namespace tensorflow
