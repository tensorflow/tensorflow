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
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace {

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

  return OkStatus();
}

// Sets shape list attribute with the given `name` to the given `shapes`. If the
// attribute already exists then this will just retain the set value.
template <typename ContainerT,
          typename = typename std::enable_if<std::is_same<
              llvm::Optional<llvm::ArrayRef<int64_t>>,
              decltype(*std::declval<ContainerT>().begin())>::value>::type>
void SetShapeAttribute(absl::string_view name, ContainerT shapes,
                       AttrValueMap* values) {
  AttrValue value;
  auto& shape_list = *value.mutable_list();
  for (const llvm::Optional<llvm::ArrayRef<int64_t>>& shape : shapes) {
    TensorShapeProto& tshape = *shape_list.add_shape();
    if (shape.has_value()) {
      for (int64_t dim : *shape) {
        tshape.add_dim()->set_size(mlir::ShapedType::isDynamic(dim) ? -1 : dim);
      }
    } else {
      tshape.set_unknown_rank(true);
    }
  }

  // If shape is already set, override it. This can happen if we import
  // without shape inference enabled and so couldn't be removed on import and
  // are not explicitly dropped later.
  (*values)[string(name)] = value;
}

// Collects all the unregistered attributes for an TF dialect operation.
// Attributes "name" and "device" are not included because they are not part
// of an TF op attributes.
Status GetUnregisteredAttrs(
    mlir::Operation* inst, const tensorflow::OpRegistrationData* op_reg_data,
    absl::flat_hash_set<absl::string_view>* attrs_to_ignore) {
  if (!op_reg_data) {
    // This is likely a function call node, so we should continue.
    return OkStatus();
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
    if (registered_attrs.find(attr.getName()) == registered_attrs.end()) {
      attrs_to_ignore->insert(
          absl::string_view(attr.getName().data(), attr.getName().size()));
    }
  }
  return OkStatus();
}

// Collects all attribute names to ignore in an MLIR operation when exporting to
// a TensorFlow NodeDef.
StatusOr<absl::flat_hash_set<absl::string_view>> GetAttributesToIgnore(
    mlir::Operation* inst, mlir::DictionaryAttr derived_attrs,
    const tensorflow::OpRegistrationData* op_reg_data,
    bool ignore_unregistered_attrs) {
  // The elements are owned by the MLIRContext.
  absl::flat_hash_set<absl::string_view> attrs_to_ignore;

  // We ignore attributes attached to the operation when there is already a
  // derived attribute defined in ODS.
  if (derived_attrs) {
    for (auto derived_attr : derived_attrs) {
      attrs_to_ignore.insert(
          mlir::StringRefToView(derived_attr.getName().strref()));
    }
  }

  if (ignore_unregistered_attrs) {
    TF_RETURN_IF_ERROR(
        GetUnregisteredAttrs(inst, op_reg_data, &attrs_to_ignore));
  }

  if (inst->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
    // TODO(b/146937733): Don't use <void> here.
    llvm::StringRef attr_name = mlir::OpTrait::AttrSizedOperandSegments<
        void>::getOperandSegmentSizeAttr();
    attrs_to_ignore.insert(attr_name.data());
  }

  if (inst->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
    // TODO(b/146937733): Don't use <void> here.
    llvm::StringRef attr_name = mlir::OpTrait::AttrSizedResultSegments<
        void>::getResultSegmentSizeAttr();
    attrs_to_ignore.insert(attr_name.data());
  }

  if (llvm::isa<mlir::TF::CaseOp, mlir::TF::IfOp, mlir::TF::WhileOp>(inst))
    attrs_to_ignore.insert("is_stateless");

  if (llvm::isa<mlir::TF::WhileOp>(inst))
    attrs_to_ignore.insert("shape_invariant");

  return attrs_to_ignore;
}

// Populates all derived attributes of a MLIR operation in a proto
// map<string, AttrValue>.
Status PopulateDerivedAttributes(mlir::Operation* inst, llvm::StringRef name,
                                 mlir::DictionaryAttr derived_attrs,
                                 bool ignore_unregistered_attrs,
                                 AttrValueMap* attributes) {
  if (derived_attrs) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        ConvertAttributes(derived_attrs.getValue(), /*attrs_to_ignore=*/{},
                          /*remove_ref_type=*/true, attributes),
        "while converting derived attributes for node: ",
        mlir::StringRefToView(name));
  }

  // Here we only add the shapes for the leading values with ShapedType,
  // assuming values with non-ShapedType are put at the end of the result.
  if (!ignore_unregistered_attrs && inst->getNumResults() > 0) {
    auto values = inst->getResults();
    auto begin = values.begin();
    auto end = values.begin();
    while (end != values.end() && (*end).getType().isa<mlir::ShapedType>())
      end++;
    if (begin != end) {
      mlir::TF::ResultShapeRange output_shapes = {
          mlir::TF::ResultShapeIterator(begin),
          mlir::TF::ResultShapeIterator(end)};
      SetShapeAttribute("_output_shapes", output_shapes, attributes);
    }
  }

  return OkStatus();
}

// A `Cast` with DstT == SrcT can be introduced in MLIR as a shape cast. But
// `Cast` only has shapes in the TF dialect's types, not TF graph, so it is
// valid to convert a `Cast` to an `Identity`. The `_output_shapes` attribute of
// the `Cast` will be preserved. This transform is needed for the graph to be
// executed on TPU or GPU devices, which do not have `Cast` registered as a
// runtime OpKernel.
void RemoveIdentityCast(NodeDef* node_def) {
  auto attr = node_def->mutable_attr();
  if (node_def->op() == "Cast" && attr->contains("SrcT") &&
      attr->contains("DstT") &&
      attr->at("SrcT").type() == attr->at("DstT").type() &&
      attr->contains("Truncate") && !attr->at("Truncate").b()) {
    node_def->set_op("Identity");
    attr->insert({{"T", attr->at("SrcT")}});
    attr->erase("SrcT");
    attr->erase("DstT");
    attr->erase("Truncate");
  }
}

}  // namespace

Status GetAttrValuesFromOperation(
    mlir::Operation* inst, llvm::StringRef name,
    const tensorflow::OpRegistrationData* op_reg_data,
    bool ignore_unregistered_attrs, AttrValueMap* attributes) {
  mlir::DictionaryAttr derived_attrs = nullptr;
  if (auto interface = llvm::dyn_cast<mlir::DerivedAttributeOpInterface>(inst))
    derived_attrs = interface.materializeDerivedAttributes();
  TF_ASSIGN_OR_RETURN(auto attrs_to_ignore,
                      GetAttributesToIgnore(inst, derived_attrs, op_reg_data,
                                            ignore_unregistered_attrs));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertAttributes(inst->getAttrs(), attrs_to_ignore,
                        /*remove_ref_type=*/false, attributes),
      "while converting attributes for node: ", mlir::StringRefToView(name));
  TF_RETURN_IF_ERROR(PopulateDerivedAttributes(
      inst, name, derived_attrs, ignore_unregistered_attrs, attributes));

  //  Explicitly handle XlaHostCompute op which has required function attribute
  //  in TensorFlow op def but it could have an empty value to represent missing
  //  functions. This value can't be represented using MLIR SymbolRefAttr and
  //  instead uses optional symbol ref attribute.
  //
  // TODO(b/182315488): Remove custom handling by finding a better
  // representation in MLIR for empty function names. One option could be to use
  // TensorFlow op defs to figure out function attributes that are missing in
  // MLIR. This will also require some trait to identify optional attributes in
  // MLIR.
  constexpr char kShapeInferenceGraph[] = "shape_inference_graph";
  if (mlir::isa<mlir::TF::XlaHostComputeOp>(inst) &&
      !inst->hasAttr(kShapeInferenceGraph) &&
      !attrs_to_ignore.contains(kShapeInferenceGraph)) {
    AttrValue value;
    value.mutable_func()->set_name("");
    (*attributes)[kShapeInferenceGraph] = value;
  }
  return OkStatus();
}

StatusOr<std::unique_ptr<NodeDef>> ConvertTFDialectOpToNodeDef(
    mlir::Operation* inst, llvm::StringRef name,
    bool ignore_unregistered_attrs) {
  TF_ASSIGN_OR_RETURN(auto node_def, GetOperationNodeDef(inst, name));
  TF_ASSIGN_OR_RETURN(auto op_name,
                      GetTensorFlowOpName(inst->getName().getStringRef()));
  const tensorflow::OpRegistrationData* op_reg_data =
      tensorflow::OpRegistry::Global()->LookUp(op_name.str());
  TF_RETURN_IF_ERROR(GetAttrValuesFromOperation(inst, name, op_reg_data,
                                                ignore_unregistered_attrs,
                                                node_def->mutable_attr()));
  RemoveIdentityCast(node_def.get());
  return node_def;
}

}  // namespace tensorflow
