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

#include "tensorflow/core/kernels/composite_tensor_variant.h"

#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/composite_tensor_variant.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

constexpr const char CompositeTensorVariant::kTypeName[];

CompositeTensorVariant::CompositeTensorVariant(
    const CompositeTensorVariantMetadata& metadata,
    absl::Span<Tensor> flat_components)
    : flat_components_(flat_components.begin(), flat_components.end()),
      metadata_(new CompositeTensorVariantMetadata()) {
  *metadata_ = metadata;
}

CompositeTensorVariant::CompositeTensorVariant()
    : metadata_(new CompositeTensorVariantMetadata()) {}

CompositeTensorVariant::CompositeTensorVariant(
    const CompositeTensorVariant& other)
    : flat_components_(other.flat_components_),
      metadata_(new CompositeTensorVariantMetadata()) {
  *metadata_ = *other.metadata_;
}

void CompositeTensorVariant::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  metadata_->SerializeToString(&data->metadata_string());
  for (const Tensor& tensor : flat_components_) {
    data->add_tensor(tensor);
  }
}

bool CompositeTensorVariant::Decode(const VariantTensorData& data) {
  if (!metadata_->ParseFromString(data.metadata_string())) {
    return false;
  }
  flat_components_ = data.tensors();
  return true;
}

string CompositeTensorVariant::DebugString() const {
  string result("<CompositeTensorVariant type=");
  result.append(TypeSpecProto::TypeSpecClass_Name(
      metadata_->type_spec_proto().type_spec_class()));
  result.append(", components=[");
  for (const auto& tensor : flat_components_) {
    if (&tensor != &flat_components_[0]) {
      result.append(", ");
    }
    result.append(tensor.DebugString());
  }
  result.append("]>");
  return result;
}

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CompositeTensorVariant,
                                       CompositeTensorVariant::kTypeName);

}  // namespace tensorflow
