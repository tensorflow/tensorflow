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
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/Support/DebugStringHelper.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
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
#include "tensorflow/core/graph/graph_constructor.h"
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
    for (auto loc : fused.getLocations()) {
      TF_RETURN_IF_ERROR(ConvertLocation(loc, debug_info));
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

Status ConvertAttribute(const mlir::UnitAttr& attr, AttrValue* value) {
  value->clear_value();
  return Status::OK();
}

Status ConvertAttribute(const mlir::SymbolRefAttr& attr, AttrValue* value) {
  value->mutable_func()->set_name(attr.getValue());
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
    } else if (auto attr = a.dyn_cast<mlir::SymbolRefAttr>()) {
      AttrValue attrVal;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attrVal));
      *list->add_func() = attrVal.func();
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

}  // anonymous namespace

StatusOr<std::unique_ptr<NodeDef>> GetOperationNodeDef(
    mlir::Operation* inst, llvm::StringRef name,
    OpNameMappingFunc op_name_func) {
  auto node_def = absl::make_unique<NodeDef>();
  // Note: we do not use NodeBuilder or NodeDefBuilder as that would require
  // mapping back from the inputs to the input arguments.
  TF_ASSIGN_OR_RETURN(auto op_name,
                      op_name_func(inst->getName().getStringRef()));
  node_def->set_op(op_name);
  node_def->set_name(name);

  // Add inputs to the NodeDef based on the number of operands. This is required
  // as later when edges are added to the Node using Graph::AddEdge the
  // associated NodeDef is not updated.
  for (int i = 0, e = inst->getNumOperands(); i < e; ++i) {
    node_def->add_input();
  }
  if (auto attr = inst->getAttrOfType<mlir::StringAttr>("device")) {
    node_def->set_device(attr.getValue());
  }

  // Add the node attributes.
  absl::flat_hash_set<string> attrs_to_ignore;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertAttributes(inst->getAttrs(), attrs_to_ignore,
                        node_def->mutable_attr()),
      "TensorFlow node name: ", name.str());

  // Add the node debug info.
  TF_RETURN_IF_ERROR(ConvertLocation(
      inst->getLoc(), node_def->mutable_experimental_debug_info()));

  if (node_def->op() == "If") UpdateCompositeIfOp(node_def.get());

  return node_def;
}

Status ConvertAttributes(const llvm::ArrayRef<mlir::NamedAttribute> attrs,
                         const absl::flat_hash_set<string>& attrs_to_ignore,
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
        auto func_attr = attr.cast<mlir::SymbolRefAttr>();
        value.mutable_func()->set_name(func_attr.getValue());
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
      case mlir::StandardAttributes::DenseElements:
      case mlir::StandardAttributes::OpaqueElements:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::ElementsAttr>(), &value));
        break;
      case mlir::StandardAttributes::Unit:
        TF_RETURN_IF_ERROR(
            ConvertAttribute(attr.cast<mlir::UnitAttr>(), &value));
        break;
      // AffineMap and Type kinds are not implemented.
      default:
        return errors::Unimplemented("Unhandled attribute kind");
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
Status SetAttribute(absl::string_view name, mlir::Type type,
                    AttrValueMap* values) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, &dtype));

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

}  // namespace tensorflow
