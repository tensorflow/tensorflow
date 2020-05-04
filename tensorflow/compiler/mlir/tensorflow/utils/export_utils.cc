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

#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace {
// Converts a location to the debug information for the node def.
Status ConvertLocation(mlir::Location inst_loc,
                       NodeDef::ExperimentalDebugInfo* debug_info) {
  if (auto call_site = inst_loc.dyn_cast<mlir::CallSiteLoc>()) {
    if (auto name_loc = call_site.getCallee().dyn_cast<mlir::NameLoc>()) {
      debug_info->add_original_node_names(name_loc.getName().c_str());
    }
  } else if (auto fused = inst_loc.dyn_cast<mlir::FusedLoc>()) {
    auto locations = fused.getLocations();
    if (locations.size() <= 1)
      return errors::InvalidArgument("expected experimental debuf info.");
    // skip the first one, which is the name of the node_def.
    for (int i = 0; i < locations.size() - 1; ++i) {
      TF_RETURN_IF_ERROR(ConvertLocation(locations[i], debug_info));
    }
  }
  return Status::OK();
}

Status ConvertAttribute(const mlir::BoolAttr& attr, AttrValue* value) {
  value->set_b(attr.getValue());
  return Status::OK();
}

Status ConvertAttribute(const mlir::IntegerAttr& attr, AttrValue* value) {
  value->set_i(attr.getInt());
  return Status::OK();
}

Status ConvertAttribute(const mlir::FloatAttr& attr, AttrValue* value) {
  value->set_f(attr.getValueAsDouble());
  return Status::OK();
}

Status ConvertAttribute(const mlir::ElementsAttr& attr, AttrValue* value) {
  return ConvertToTensorProto(attr, value->mutable_tensor());
}

Status ConvertAttribute(const mlir::TF::ShapeAttr& attr, AttrValue* value) {
  auto* shape = value->mutable_shape();
  if (attr.hasRank()) {
    for (auto dim_size : attr.getShape()) {
      auto* dim = shape->add_dim();
      dim->set_size(dim_size);
    }
  } else {
    shape->set_unknown_rank(true);
  }
  return Status::OK();
}

Status ConvertAttribute(const mlir::StringAttr& attr, AttrValue* value) {
  absl::string_view attr_value(attr.getValue().data(), attr.getValue().size());
  switch (mangling_util::GetMangledKind(attr_value)) {
    case mangling_util::MangledKind::kUnknown: {
      value->set_s(std::string(attr_value));
      return Status::OK();
    }
    case mangling_util::MangledKind::kDataType: {
      DataType dtype;
      TF_RETURN_IF_ERROR(mangling_util::DemangleDataType(attr_value, &dtype));
      value->set_type(dtype);
      return Status::OK();
    }
    case mangling_util::MangledKind::kTensorShape:
      TF_RETURN_IF_ERROR(
          mangling_util::DemangleShape(attr_value, value->mutable_shape()));
      return Status::OK();
    default:
      return errors::Unimplemented("Mangled string couldn't be handled!");
  }
  return Status::OK();
}

Status ConvertAttribute(mlir::Type type, AttrValue* value) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &dtype));
  value->set_type(dtype);
  return Status::OK();
}

Status ConvertAttribute(const mlir::TypeAttr& type, AttrValue* value) {
  return ConvertAttribute(type.getValue(), value);
}

Status ConvertAttribute(const mlir::UnitAttr& attr, AttrValue* value) {
  value->clear_value();
  return Status::OK();
}

Status ConvertAttribute(const mlir::FlatSymbolRefAttr& attr, AttrValue* value) {
  value->mutable_func()->set_name(std::string(attr.getValue()));
  return Status::OK();
}

Status ConvertAttribute(const mlir::ArrayAttr& attr, AttrValue* value) {
  auto* list = value->mutable_list();
  for (mlir::Attribute a : attr.getValue()) {
    if (auto attr = a.dyn_cast<mlir::BoolAttr>()) {
      list->add_b(attr.getValue());
    } else if (auto attr = a.dyn_cast<mlir::IntegerAttr>()) {
      list->add_i(attr.getInt());
    } else if (auto attr = a.dyn_cast<mlir::FloatAttr>()) {
      list->add_f(attr.getValueAsDouble());
    } else if (auto attr = a.dyn_cast<mlir::StringAttr>()) {
      AttrValue nested_value;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &nested_value));
      switch (nested_value.value_case()) {
        case AttrValue::kS:
          list->add_s(nested_value.s());
          break;
        case AttrValue::kType:
          list->add_type(nested_value.type());
          break;
        case AttrValue::kShape:
          *list->add_shape() = nested_value.shape();
          break;
        default:
          return errors::Unimplemented("Unhandled nested attribute!");
      }
    } else if (auto attr = a.dyn_cast<mlir::ElementsAttr>()) {
      TensorProto tensor;
      TF_RETURN_IF_ERROR(ConvertToTensorProto(attr, &tensor));
      *list->add_tensor() = tensor;
    } else if (auto attr = a.dyn_cast<mlir::FlatSymbolRefAttr>()) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attr_val));
      *list->add_func() = attr_val.func();
    } else if (auto attr = a.dyn_cast<mlir::TypeAttr>()) {
      AttrValue attr_val;
      // For type attributes, we only propagate the element type.
      mlir::Type elt_type = attr.getValue();
      if (auto shaped_type = elt_type.dyn_cast<mlir::ShapedType>()) {
        elt_type = shaped_type.getElementType();
      }
      TF_RETURN_IF_ERROR(ConvertAttribute(elt_type, &attr_val));
      list->add_type(attr_val.type());
    } else if (auto attr = a.dyn_cast<mlir::TF::ShapeAttr>()) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attr_val));
      *list->add_shape() = attr_val.shape();
    } else {
      return errors::Unimplemented("Unhandled attribute!");
    }
  }
  return Status::OK();
}

// Updates NodeDef constructed out of an MLIR If op to map it to either
// TensorFlow StatelessIf or If op depending on the additional attribute.
void UpdateCompositeIfOp(NodeDef* node_def) {
  auto it = node_def->mutable_attr()->find("is_stateless");
  if (it != node_def->attr().end()) {
    if (it->second.b()) {
      *node_def->mutable_op() = "StatelessIf";
    }
    node_def->mutable_attr()->erase(it);
  }
}

// Updates NodeDef constructed out of an MLIR While op to map it to either
// TensorFlow StatelessWhile or While op depending on the additional attribute.
void UpdateCompositeWhileOp(NodeDef* node_def) {
  auto it = node_def->mutable_attr()->find("is_stateless");
  if (it != node_def->attr().end()) {
    if (it->second.b()) {
      *node_def->mutable_op() = "StatelessWhile";
    }
    node_def->mutable_attr()->erase(it);
  }
}

// Returns true if the executor/control dialect op should map to Ref node in
// TensorFlow Graph. For control dialect NextIteration it uses the 1st operand
// type. For executor dialect NextIteration it uses the 2nd operand type. For
// all others (Enter/Exit/Merge/Switch), if the output type is ref, they
// correspond to the Ref equivalent op in TF Graph.
static bool IsRefTypeControlOp(mlir::Operation* op) {
  if (auto next_iter_sink =
          llvm::dyn_cast<mlir::tf_executor::NextIterationSinkOp>(op))
    return mlir::getElementTypeOrSelf(next_iter_sink.input().getType())
        .isa<mlir::TF::TensorFlowRefType>();

  auto op_name_or_status = GetTensorFlowOpName(op->getName().getStringRef());
  if (!op_name_or_status.ok()) return false;

  auto op_name = op_name_or_status.ConsumeValueOrDie();
  if (op_name.equals("NextIteration"))
    return mlir::getElementTypeOrSelf(op->getOperand(0).getType())
        .isa<mlir::TF::TensorFlowRefType>();

  if (op_name.equals("Enter") || op_name.equals("Exit") ||
      op_name.equals("Switch") || op_name.equals("Merge")) {
    return getElementTypeOrSelf(op->getResult(0).getType())
        .isa<mlir::TF::TensorFlowRefType>();
  }
  return false;
}

}  // anonymous namespace

StatusOr<llvm::StringRef> GetTensorFlowOpName(llvm::StringRef op_name) {
  // When being converted to MLIR, some prefixes and suffixes are added to the
  // operation types, and we have to remove them when converting the
  // operations back to a graph:
  // - "_tf.", "tf." or "tf_executor." : every operation type has this prefix.
  // - ".sink" or ".Sink": only the NextIteration operation has this suffix. We
  // don't need to consider ".source"/".Source" because the nodes with this
  // suffix are skipped by the caller and will not be added to the graph.
  if (!op_name.consume_front("_tf.") && !op_name.consume_front("tf.") &&
      !op_name.consume_front("tf_executor.")) {
    return errors::FailedPrecondition("op node '", op_name.str(),
                                      "' was not a TF op!");
  }
  // Control dialect NextIteration sink ends with ".sink" and Executor dialect
  // NextIteration sink ends with ".Sink".
  if (!op_name.consume_back(".sink")) op_name.consume_back(".Sink");
  return op_name;
}

StatusOr<std::unique_ptr<NodeDef>> GetOperationNodeDef(
    const absl::flat_hash_set<absl::string_view>& attrs_to_ignore,
    mlir::Operation* inst, llvm::StringRef name) {
  auto node_def = absl::make_unique<NodeDef>();
  // Note: we do not use NodeBuilder or NodeDefBuilder as that would require
  // mapping back from the inputs to the input arguments.

  llvm::SmallString<64> op_name;
  if (IsLegacyCallInstruction(inst)) {
    // The op_name is the name of the function.
    op_name.append(
        inst->getAttrOfType<mlir::SymbolRefAttr>("f").getLeafReference());
    // Remove the attribute from the instruction as it is already converted to
    // op_name.
    auto attr_id = mlir::Identifier::get("f", inst->getContext());
    inst->removeAttr(attr_id);
  } else {
    // Some control flow ops in TensorFlow Graph have their respective "Ref" ops
    // as well. For example there is Enter and RefEnter op. RefEnter forwards
    // the input ref buffer to output. However both Enter and RefEnter are
    // mapped to tf_executor::EnterOp during import and then to _tf.Enter op in
    // control dialect. Check if it is a Ref op to correctly map to the
    // TensorFlow Graph op.
    if (IsRefTypeControlOp(inst)) op_name = "Ref";
    TF_ASSIGN_OR_RETURN(auto tf_name,
                        GetTensorFlowOpName(inst->getName().getStringRef()));
    op_name.append(tf_name);
  }

  node_def->set_name(name.str());
  node_def->set_op(std::string(op_name.str()));

  // Add inputs to the NodeDef based on the number of operands. This is required
  // as later when edges are added to the Node using Graph::AddEdge the
  // associated NodeDef is not updated.
  for (int i = 0, e = inst->getNumOperands(); i < e; ++i) {
    node_def->add_input();
  }
  if (auto attr = inst->getAttrOfType<mlir::StringAttr>("device")) {
    node_def->set_device(std::string(attr.getValue()));
  }

  // Add the node attributes.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertAttributes(inst->getAttrs(), attrs_to_ignore,
                        node_def->mutable_attr()),
      "while converting attributes for node: ", name.str());

  // Add the node debug info.
  TF_RETURN_IF_ERROR(ConvertLocation(
      inst->getLoc(), node_def->mutable_experimental_debug_info()));

  if (node_def->op() == "If") UpdateCompositeIfOp(node_def.get());
  if (node_def->op() == "While") UpdateCompositeWhileOp(node_def.get());

  return node_def;
}

Status ConvertAttributes(
    const llvm::ArrayRef<mlir::NamedAttribute> attrs,
    const absl::flat_hash_set<absl::string_view>& attrs_to_ignore,
    AttrValueMap* values) {
  AttrValueMap func_call_attrs;
  for (const mlir::NamedAttribute& named_attr : attrs) {
    auto name_strref = named_attr.first.str();
    auto attr = named_attr.second;
    absl::string_view name(name_strref.data(), name_strref.size());
    if (name == "name" || name == "device" || attrs_to_ignore.contains(name)) {
      // The name, device spec of a TF op or function are not stored as
      // AttrValue inside NodeDef, but we model them using attribute inside
      // MLIR. So we need to ignore them when going back to AttrValue here.
      continue;
    }
    if (mangling_util::IsMangledAttributeName(name)) {
      // In MLIR, attributes for functions requires dialect prefix. We need to
      // remove TF dialect prefix before converting to AttrValue.
      name = mangling_util::DemangleAttributeName(name);
    }
    AttrValue value;
    switch (attr.getKind()) {
      case mlir::StandardAttributes::SymbolRef: {
        auto func_attr = attr.cast<mlir::FlatSymbolRefAttr>();
        value.mutable_func()->set_name(std::string(func_attr.getValue()));
        func_call_attrs[string(name)] = value;
        continue;
      }
      case mlir::StandardAttributes::Bool:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::BoolAttr>(), &value));
        break;
      case mlir::StandardAttributes::Integer:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::IntegerAttr>(), &value));
        break;
      case mlir::StandardAttributes::Float:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::FloatAttr>(), &value));
        break;
      case mlir::StandardAttributes::String:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::StringAttr>(), &value));
        break;
      case mlir::StandardAttributes::Array:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::ArrayAttr>(), &value));
        break;
      case mlir::StandardAttributes::DenseIntOrFPElements:
      case mlir::StandardAttributes::DenseStringElements:
      case mlir::StandardAttributes::OpaqueElements:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::ElementsAttr>(), &value));
        break;
      case mlir::StandardAttributes::Type:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::TypeAttr>(), &value));
        break;
      case mlir::StandardAttributes::Unit:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::UnitAttr>(), &value));
        break;
      case static_cast<unsigned>(mlir::TF::AttrKind::SHAPE):
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::TF::ShapeAttr>(), &value));
        break;
      // AffineMap kind is not implemented.
      case mlir::StandardAttributes::AffineMap:
        return errors::Unimplemented("AffineMap attribute (needed for '",
                                     name_strref, "') unimplemented");
      default:
        return errors::Unimplemented("Unhandled attribute kind for attribute '",
                                     name_strref, '\'');
    }
    // According to the NodeDef proto definition, an attribute name from the
    // input TensorFlow GraphDef shouldn't contain '.'. If it does appear in
    // the attribute from MLIR, it is treated as an attribute from function
    // calls.
    std::vector<string> name_tokens =
        absl::StrSplit(name, '.', absl::SkipEmpty());
    TF_RET_CHECK(name_tokens.size() <= 2);
    auto it = func_call_attrs.find(name_tokens[0]);
    if (it == func_call_attrs.end()) {
      (*values)[string(name)] = value;
    } else {
      (*it->second.mutable_func()->mutable_attr())[name_tokens[1]] = value;
    }
  }
  for (const auto& it : func_call_attrs) {
    (*values)[it.first] = it.second;
  }
  return Status::OK();
}

// Sets type attribute with the given name. If the attribute already exists with
// a different value, returns an error.
Status SetTypeAttribute(absl::string_view name, mlir::Type type,
                        AttrValueMap* values) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, &dtype));
  if (tensorflow::IsRefType(dtype)) dtype = tensorflow::RemoveRefType(dtype);
  AttrValue value;
  value.set_type(dtype);

  auto result = values->insert({string(name), value});
  if (!result.second) {
    DataType actual_dtype = result.first->second.type();
    if (actual_dtype != dtype) {
      return errors::InvalidArgument("Expected ", DataType_Name(dtype), " '",
                                     name, "' attribute but found ",
                                     DataType_Name(actual_dtype));
    }
  }
  return Status::OK();
}

Status SetShapeAttribute(absl::string_view name, mlir::ShapedType shaped_type,
                         AttrValueMap* values) {
  tensorflow::TensorShapeProto tshape;
  AttrValue value;
  if (shaped_type.hasRank()) {
    for (auto dim : shaped_type.getShape()) tshape.add_dim()->set_size(dim);
  } else {
    tshape.set_unknown_rank(true);
  }
  *value.mutable_shape() = tshape;

  auto result = values->insert({string(name), value});
  if (!result.second) {
    // This should be extremely rare as it means we are adding the same
    // attribute multiple times/have some redundancy in representing this
    // attribute.
    TensorShapeProto actual_shape = result.first->second.shape();
    // Just check via string output as we shouldn't get here and if we do they
    // should be trivially the same, else fail.
    if (actual_shape.ShortDebugString() != tshape.ShortDebugString()) {
      return errors::InvalidArgument("Expected ", tshape.ShortDebugString(),
                                     " '", name, "' attribute but found ",
                                     actual_shape.ShortDebugString());
    }
  }
  return Status::OK();
}

Status SetSizeAttribute(absl::string_view name, size_t size,
                        AttrValueMap* values) {
  AttrValue value;
  value.set_i(size);

  auto result = values->insert({string(name), value});
  if (!result.second) {
    // This should be extremely rare as it means we are adding the same
    // attribute multiple times/have some redundancy in representing this
    // attribute.
    int64 actual_size = result.first->second.i();
    // Just check via string output as we shouldn't get here and if we do they
    // should be trivially the same, else fail.
    if (actual_size != size)
      return errors::InvalidArgument("Expected '", name, "' attribute to be ",
                                     size, " but found ", actual_size);
  }
  return Status::OK();
}

bool IsLegacyCallInstruction(mlir::Operation* inst) {
  return llvm::dyn_cast<mlir::TF::LegacyCallOp>(inst) ||
         inst->getName().getStringRef().compare("_tf.LegacyCall") == 0;
}

}  // namespace tensorflow
