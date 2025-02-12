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

#include "tensorflow/core/framework/node_properties.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// static
absl::Status NodeProperties::CreateFromNodeDef(
    NodeDef node_def, const OpRegistryInterface* op_registry,
    std::shared_ptr<const NodeProperties>* props) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(op_registry->LookUpOpDef(node_def.op(), &op_def));
  DataTypeVector input_types;
  DataTypeVector output_types;
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(node_def, *op_def, &input_types, &output_types));
  props->reset(new NodeProperties(op_def, std::move(node_def),
                                  std::move(input_types),
                                  std::move(output_types)));
  return absl::OkStatus();
}

}  // namespace tensorflow
