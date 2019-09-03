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

#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringSet.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace {
using stream_executor::port::StatusOr;

// Sets type list attribute with the given `name` to the given `types`. If the
// attribute already exists with a different value, returns an error.
template <typename ContainerT,
          typename = typename std::enable_if<
              std::is_same<mlir::Type, decltype(*std::declval<ContainerT>()
                                                     .begin())>::value>::type>
Status SetAttribute(absl::string_view name, ContainerT types,
                    AttrValueMap* values) {
  AttrValue value;
  auto& type_list = *value.mutable_list();
  for (auto type : types) {
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, &dtype));
    type_list.add_type(dtype);
  }

  auto result = values->insert({string(name), value});
  if (!result.second) {
    const auto& prev_dtypes = result.first->second.list();
    int count = prev_dtypes.type_size();
    if (count != type_list.type_size()) {
      return errors::InvalidArgument("Type list count mismatch");
    }

    for (int i = 0; i < count; ++i) {
      if (prev_dtypes.type(i) != type_list.type(i))
        return errors::InvalidArgument("Type list mismatch");
    }
  }

  return Status::OK();
}

// Include the auto generated derived attribute populator function taking
// TensorFlow dialect operation as an argument. This file contains the function
// definitions and isn't a header file.
#include "tensorflow/compiler/mlir/tensorflow/translate/derived_attr_populator.inc"

// Collect all the unregistered attributes for an TF dialect operation.
// Attributes "name" and "device" are not included because they are not part
// of an TF op attributes.
Status GetUnregisteredAttrs(
    mlir::Operation* inst,
    absl::flat_hash_set<absl::string_view>* attrs_to_ignore) {
  TF_ASSIGN_OR_RETURN(auto op_name,
                      GetTensorFlowOpName(inst->getName().getStringRef()));

  const tensorflow::OpRegistrationData* op_reg_data;
  auto status = tensorflow::OpRegistry::Global()->LookUp(op_name, &op_reg_data);
  if (!status.ok()) {
    // This is likely a function call node, so we should continue.
    VLOG(1) << status.ToString();
    return Status::OK();
  }

  // Collect all the registered attributes.
  llvm::DenseSet<llvm::StringRef> registered_attrs;
  registered_attrs.insert("name");
  registered_attrs.insert("device");
  for (const auto& attr_def : op_reg_data->op_def.attr()) {
    registered_attrs.insert(attr_def.name());
  }
  // Attributes are not in the registered attributes set will be ignored.
  for (auto& attr : inst->getAttrs()) {
    auto attr_name = attr.first.c_str();
    if (registered_attrs.find(attr_name) == registered_attrs.end()) {
      attrs_to_ignore->insert(attr_name);
    }
  }
  return Status::OK();
}

}  // namespace

StatusOr<std::unique_ptr<NodeDef>> ConvertTFDialectOpToNodeDef(
    mlir::Operation* inst, llvm::StringRef name,
    bool ignore_unregistered_attrs) {
  // The elements are owned by the MLIRContext.
  absl::flat_hash_set<absl::string_view> attrs_to_ignore;
  if (ignore_unregistered_attrs) {
    TF_RETURN_IF_ERROR(GetUnregisteredAttrs(inst, &attrs_to_ignore));
  }

  TF_ASSIGN_OR_RETURN(auto node_def,
                      GetOperationNodeDef(attrs_to_ignore, inst, name));

  // Use auto generated function to populate derived attribute.
  //
  // Note: This only populates derived attributes for TensorFlow ops that are
  // generated using the TableGen. Manually defined ops and TF ops with control
  // edges (i.e TF op names with leading '_' in names) should have all the
  // attributes present as native MLIR op attributes.
  //
  // TODO(hinsu): Handle TF ops with control edges that has auto-generated
  // TensorFlow ops available. Such ops can drop all the derived attributes
  // during import to MLIR and rely on exporter to reintroduce them.
  auto status = PopulateDerivedAttrs(inst, node_def->mutable_attr());
  if (!status.ok()) {
    return errors::Internal("Falied to populate derived attrs: " +
                            status.ToString());
  }
  return node_def;
}

}  // namespace tensorflow
