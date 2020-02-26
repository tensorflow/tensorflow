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

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
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
Status SetTypeAttribute(absl::string_view name, ContainerT types,
                        AttrValueMap* values) {
  AttrValue value;
  auto& type_list = *value.mutable_list();
  for (auto type : types) {
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, &dtype));
    type_list.add_type(dtype);
  }

  auto result = values->insert({string(name), value});
  assert(result.second && "cannot have multiple attributes with the same name");
  (void)result;

  return Status::OK();
}

// Sets shape list attribute with the given `name` to the given `shapes`. If the
// attribute already exists with a different value, returns an error.
template <typename ContainerT,
          typename = typename std::enable_if<std::is_same<
              llvm::Optional<llvm::ArrayRef<int64_t>>,
              decltype(*std::declval<ContainerT>().begin())>::value>::type>
Status SetShapeAttribute(absl::string_view name, ContainerT shapes,
                         AttrValueMap* values) {
  AttrValue value;
  auto& shape_list = *value.mutable_list();
  for (const llvm::Optional<llvm::ArrayRef<int64_t>>& shape : shapes) {
    TensorShapeProto& tshape = *shape_list.add_shape();
    if (shape.hasValue()) {
      for (int64_t dim : *shape) tshape.add_dim()->set_size(dim);
    } else {
      tshape.set_unknown_rank(true);
    }
  }

  auto result = values->insert({string(name), value});
  assert(result.second && "cannot have multiple attributes with the same name");
  (void)result;

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

  const tensorflow::OpRegistrationData* op_reg_data =
      tensorflow::OpRegistry::Global()->LookUp(std::string(op_name));
  if (!op_reg_data) {
    // This is likely a function call node, so we should continue.
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
  // Use auto generated function to populate derived attribute.
  //
  // Note: This only populates derived attributes for TensorFlow ops that are
  // generated using the TableGen. Manually defined ops and TF ops with control
  // edges (i.e TF op names with leading '_' in names) should have all the
  // attributes present as native MLIR op attributes.

  // If the operation is in the TensorFlow control dialect, we create a
  // temporary copy in the TensorFlow dialect. This is needed because we
  // auto-generated the registration for TensorFlow dialect only.
  // TODO(aminim): this is only done while we're using the TF control dialect
  // as a temporary stage when exporting to GraphDef. Remove when we update the
  // export.
  auto erase_clone = [](mlir::Operation* op) { op->erase(); };
  std::unique_ptr<mlir::Operation, decltype(erase_clone)> cloned_inst(
      nullptr, erase_clone);
  if (inst->getDialect() && inst->getDialect()->getNamespace() == "_tf") {
    mlir::OperationState result(inst->getLoc(),
                                inst->getName().getStringRef().drop_front());
    for (mlir::Value operand : inst->getOperands())
      if (!operand.getType().isa<mlir::TFControlFlow::TFControlType>())
        result.operands.push_back(operand);

    // Add a result type for each non-control result we find
    for (mlir::Type result_type : inst->getResultTypes()) {
      if (result_type.isa<mlir::TFControlFlow::TFControlType>()) break;
      result.types.push_back(result_type);
    }
    cloned_inst.reset(mlir::Operation::create(result));
    cloned_inst->setAttrs(inst->getAttrs());
    inst = cloned_inst.get();
  }

  // The elements are owned by the MLIRContext.
  absl::flat_hash_set<absl::string_view> attrs_to_ignore;
  if (inst->isRegistered()) {
    // We ignore attributes attached to the operation when there is already a
    // derived attribute defined in ODS.
    // TODO(aminim) replace absl::flat_hash_set with a SmallDenseSet.
    llvm::SmallDenseSet<llvm::StringRef> derived_attrs;
    CollectDerivedAttrsName(inst, &derived_attrs);
    for (auto name : derived_attrs) attrs_to_ignore.insert(name.data());
  }

  if (ignore_unregistered_attrs) {
    TF_RETURN_IF_ERROR(GetUnregisteredAttrs(inst, &attrs_to_ignore));
  }

  if (inst->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
    // TODO(b/146937733): Don't use <void> here.
    llvm::StringRef attr_name = mlir::OpTrait::AttrSizedResultSegments<
        void>::getResultSegmentSizeAttr();
    attrs_to_ignore.insert(attr_name.data());
  }

  TF_ASSIGN_OR_RETURN(auto node_def,
                      GetOperationNodeDef(attrs_to_ignore, inst, name));

  // If the operation is not registered, we won't be able to infer any attribute
  if (inst->isRegistered()) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        PopulateDerivedAttrs(inst, node_def->mutable_attr()),
        "When populating derived attrs for ",
        inst->getName().getStringRef().str());
  }
  return node_def;
}

}  // namespace tensorflow
