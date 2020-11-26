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

#include "tensorflow/core/framework/extension_type_variant.h"

#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

constexpr const char ExtensionTypeVariant::kTypeName[];

void ExtensionTypeVariant::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  metadata_.type_spec_proto().SerializeToString(&data->metadata_string());
  for (const Tensor& tensor : flat_components_) {
    data->add_tensor(tensor);
  }
}

bool ExtensionTypeVariant::Decode(const VariantTensorData& data) {
  if (!metadata_.mutable_type_spec_proto()->ParseFromString(
          data.metadata_string())) {
    return false;
  }
  flat_components_ = data.tensors();
  return true;
}

string ExtensionTypeVariant::DebugString() const {
  string type_spec;
  ::tensorflow::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  printer.PrintToString(metadata_.type_spec_proto(), &type_spec);
  string result("<ExtensionTypeVariant type_spec={");
  result.append(type_spec.empty() ? "none" : type_spec);
  result.append("}, components=[");
  for (const auto& tensor : flat_components_) {
    if (&tensor != &flat_components_[0]) {
      result.append(", ");
    }
    result.append(tensor.DebugString());
  }
  result.append("]>");
  return result;
}

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(ExtensionTypeVariant,
                                       ExtensionTypeVariant::kTypeName);

}  // namespace tensorflow
