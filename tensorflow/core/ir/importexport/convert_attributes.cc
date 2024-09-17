/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/importexport/convert_attributes.h"

#include <string>
#include <vector>

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/status_macros.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/mangling.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

using tensorflow::AttrValue;
using tensorflow::AttrValueMap;
using tensorflow::DataType;
using tensorflow::NodeDef;
using tensorflow::Status;
using tensorflow::StatusOr;
using tensorflow::TensorProto;
using tensorflow::TensorShapeProto;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::Unimplemented;

namespace mlir {
namespace tfg {

namespace {
// Converts a location to the debug information for the node def.
Status ConvertLocation(Location inst_loc,
                       NodeDef::ExperimentalDebugInfo* debug_info) {
  if (auto call_site = mlir::dyn_cast<CallSiteLoc>(inst_loc)) {
    if (auto name_loc = mlir::dyn_cast<NameLoc>(call_site.getCallee())) {
      debug_info->add_original_node_names(name_loc.getName().data());
    }
  } else if (auto fused = mlir::dyn_cast<FusedLoc>(inst_loc)) {
    auto locations = fused.getLocations();
    if (locations.size() <= 1)
      return InvalidArgument("Expected experimental debug info.");
    // skip the first one, which is the name of the node_def.
    for (int i = 0, end = locations.size() - 1; i < end; ++i) {
      TF_RETURN_IF_ERROR(ConvertLocation(locations[i], debug_info));
    }
  }
  return absl::OkStatus();
}

Status ConvertAttribute(BoolAttr attr, AttrValue* value) {
  value->set_b(attr.getValue());
  return absl::OkStatus();
}

Status ConvertAttribute(IntegerAttr attr, AttrValue* value) {
  value->set_i(attr.getInt());
  return absl::OkStatus();
}

Status ConvertAttribute(FloatAttr attr, AttrValue* value) {
  value->set_f(attr.getValueAsDouble());
  return absl::OkStatus();
}

Status ConvertAttribute(ElementsAttr attr, AttrValue* value) {
  return ConvertToTensorProto(attr, value->mutable_tensor());
}

Status ConvertAttribute(PlaceholderAttr attr, AttrValue* value) {
  value->set_placeholder(attr.getValue().str());
  return absl::OkStatus();
}

Status ConvertAttribute(ShapeAttr attr, AttrValue* value) {
  SetTensorShapeProto(attr, value->mutable_shape());
  return absl::OkStatus();
}

Status ConvertAttribute(FlatSymbolRefAttr attr, AttrValue* value) {
  value->mutable_func()->set_name(attr.getValue().str());
  return absl::OkStatus();
}

Status ConvertAttribute(FuncAttr attr, bool remove_ref_type, AttrValue* value) {
  TF_RETURN_IF_ERROR(
      ConvertAttribute(mlir::cast<FlatSymbolRefAttr>(attr.getName()), value));
  TF_RETURN_IF_ERROR(ConvertAttributes(attr.getAttrs().getValue(),
                                       /*attrs_to_ignore=*/{}, remove_ref_type,
                                       value->mutable_func()->mutable_attr()));
  return absl::OkStatus();
}

Status ConvertAttribute(StringAttr attr, AttrValue* value) {
  value->set_s(attr.str());
  return absl::OkStatus();
}

Status ConvertAttribute(Type type, bool remove_ref_type, AttrValue* value) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &dtype));
  if (tensorflow::IsRefType(dtype)) dtype = tensorflow::RemoveRefType(dtype);
  value->set_type(dtype);
  return absl::OkStatus();
}

Status ConvertAttribute(const TypeAttr& type, bool remove_ref_type,
                        AttrValue* value) {
  return ConvertAttribute(type.getValue(), remove_ref_type, value);
}

Status ConvertAttribute(const UnitAttr& attr, AttrValue* value) {
  value->clear_value();
  return absl::OkStatus();
}

Status ConvertAttribute(const ArrayAttr& attr, bool remove_ref_type,
                        AttrValue* value) {
  auto* list = value->mutable_list();
  for (Attribute a : attr.getValue()) {
    if (auto attr = mlir::dyn_cast<BoolAttr>(a)) {
      list->add_b(attr.getValue());
    } else if (auto attr = mlir::dyn_cast<IntegerAttr>(a)) {
      list->add_i(attr.getInt());
    } else if (auto attr = mlir::dyn_cast<FloatAttr>(a)) {
      list->add_f(attr.getValueAsDouble());
    } else if (auto attr = mlir::dyn_cast<StringAttr>(a)) {
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
          return Unimplemented("Unhandled nested attribute!");
      }
    } else if (auto attr = mlir::dyn_cast<ElementsAttr>(a)) {
      TensorProto tensor;
      TF_RETURN_IF_ERROR(ConvertToTensorProto(attr, &tensor));
      *list->add_tensor() = tensor;
    } else if (auto attr = mlir::dyn_cast<FlatSymbolRefAttr>(a)) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attr_val));
      *list->add_func() = attr_val.func();
    } else if (auto attr = mlir::dyn_cast<FuncAttr>(a)) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, remove_ref_type, &attr_val));
      *list->add_func() = attr_val.func();
    } else if (auto attr = mlir::dyn_cast<TypeAttr>(a)) {
      AttrValue attr_val;
      // For type attributes, we only propagate the element type.
      Type elt_type = attr.getValue();
      if (auto shaped_type = mlir::dyn_cast<ShapedType>(elt_type)) {
        elt_type = shaped_type.getElementType();
      }
      TF_RETURN_IF_ERROR(
          ConvertAttribute(elt_type, remove_ref_type, &attr_val));
      list->add_type(attr_val.type());
    } else if (auto attr = mlir::dyn_cast<ShapeAttr>(a)) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attr_val));
      *list->add_shape() = attr_val.shape();
    } else {
      return Unimplemented("Unhandled MLIR attribute in export to graph:",
                           debugString(a));
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<AttrValue> ConvertAttribute(Attribute attr) {
  AttrValue value;
  if (auto symbol_ref = mlir::dyn_cast<SymbolRefAttr>(attr)) {
    TF_RETURN_IF_ERROR(
        ConvertAttribute(mlir::cast<FlatSymbolRefAttr>(symbol_ref), &value));
    return value;
  }
  if (auto func_attr = mlir::dyn_cast<FuncAttr>(attr)) {
    TF_RETURN_IF_ERROR(
        ConvertAttribute(func_attr, /*remove_ref_type=*/false, &value));
    return value;
  }
  if (mlir::isa<AffineMapAttr>(attr))
    return Unimplemented("AffineMap attribute unimplemented");
  TF_RETURN_IF_ERROR(
      llvm::TypeSwitch<Attribute, Status>(attr)
          .Case<BoolAttr, IntegerAttr, FloatAttr, StringAttr, ElementsAttr,
                UnitAttr, ShapeAttr, PlaceholderAttr>([&](auto derived_attr) {
            return ConvertAttribute(derived_attr, &value);
          })
          .Case<ArrayAttr, TypeAttr>([&](auto derived_attr) {
            return ConvertAttribute(derived_attr,
                                    /*remove_ref_type=*/false, &value);
          })
          .Default([&](Attribute attr) {
            return Unimplemented("Unhandled attribute kind for attribute: ",
                                 debugString(attr));
          }));
  return value;
}

Status ConvertAttributes(ArrayRef<NamedAttribute> attrs,
                         ArrayRef<StringRef> attrs_to_ignore,
                         bool remove_ref_type, AttrValueMap* values) {
  StringSet<> ignored_attrs;
  ignored_attrs.insert(attrs_to_ignore.begin(), attrs_to_ignore.end());
  AttrValueMap func_call_attrs;
  for (const NamedAttribute& named_attr : attrs) {
    std::string name_str = named_attr.getName().str();
    auto attr = named_attr.getValue();
    absl::string_view name = name_str;
    if (ignored_attrs.contains(name_str)) {
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
    TF_ASSIGN_OR_RETURN(AttrValue value, ConvertAttribute(attr));
    if (mlir::isa<SymbolRefAttr>(attr)) {
      func_call_attrs[std::string(name)] = value;
      continue;
    }
    if (mlir::isa<FuncAttr>(attr)) {
      func_call_attrs[std::string(name)] = value;
      continue;
    }
    // According to the NodeDef proto definition, an attribute name from the
    // input TensorFlow GraphDef shouldn't contain '.'. If it does appear in
    // the attribute from MLIR, it is treated as an attribute from function
    // calls.
    std::vector<std::string> name_tokens =
        absl::StrSplit(name, '.', absl::SkipEmpty());
    TF_RET_CHECK(!name_tokens.empty());
    TF_RET_CHECK(name_tokens.size() <= 2);
    auto it = func_call_attrs.find(name_tokens[0]);
    if (it == func_call_attrs.end())
      (*values)[std::string(name)] = value;
    else
      (*it->second.mutable_func()->mutable_attr())[name_tokens[1]] = value;
  }
  for (const auto& it : func_call_attrs) {
    (*values)[it.first] = it.second;
  }
  return absl::OkStatus();
}

Status SetShapeAttribute(absl::string_view name, ShapedType shaped_type,
                         AttrValueMap* values) {
  AttrValue value;
  SetTensorShapeProto(shaped_type, value.mutable_shape());

  auto result = values->insert({std::string(name), value});
  if (!result.second) {
    // This should be extremely rare as it means we are adding the same
    // attribute multiple times/have some redundancy in representing this
    // attribute.
    TensorShapeProto actual_shape = result.first->second.shape();
    // Just check via string output as we shouldn't get here and if we do they
    // should be trivially the same, else fail.
    std::string new_shape_string = value.shape().ShortDebugString();
    if (actual_shape.ShortDebugString() != new_shape_string) {
      return InvalidArgument("Expected ", new_shape_string, " '", name,
                             "' attribute but found ",
                             actual_shape.ShortDebugString());
    }
  }
  return absl::OkStatus();
}

// Converts non func AttrValue proto into an MLIR attribute. Func attribute is
// exclused in this function because the function might be renamed when the
// function definition is imported.
absl::StatusOr<Attribute> ConvertNonFuncAttributeValue(const AttrValue& value,
                                                       Builder& builder) {
  switch (value.value_case()) {
    case AttrValue::kI:
      return builder.getI64IntegerAttr(value.i());
    case AttrValue::kS:
      return builder.getStringAttr(value.s());
    case AttrValue::kF:
      return builder.getFloatAttr(builder.getF32Type(), value.f());
    case AttrValue::kB:
      return builder.getBoolAttr(value.b());
    case AttrValue::kType: {
      Type type;
      TF_RETURN_IF_ERROR(ConvertDataType(value.type(), builder, &type));
      return TypeAttr::get(type);
    }
    case AttrValue::kShape:
      return ConvertTensorShapeProto(value.shape(), builder.getContext());
    case AttrValue::kTensor:
      return ConvertTensorProto(value.tensor(), builder);
    case AttrValue::kList: {
      absl::InlinedVector<Attribute, 8> attrs;
      for (const auto& item : value.list().i())
        attrs.push_back(builder.getI64IntegerAttr(item));
      for (const auto& item : value.list().s())
        attrs.push_back(builder.getStringAttr(item));
      for (const auto& item : value.list().f())
        attrs.push_back(builder.getFloatAttr(builder.getF32Type(), item));
      for (const auto& item : value.list().b())
        attrs.push_back(builder.getBoolAttr(item));
      for (const auto& item : value.list().type()) {
        Type type;
        TF_RETURN_IF_ERROR(ConvertDataType(DataType(item), builder, &type));
        attrs.push_back(TypeAttr::get(type));
      }
      for (const auto& item : value.list().shape()) {
        TF_ASSIGN_OR_RETURN(
            auto attr, ConvertTensorShapeProto(item, builder.getContext()));
        attrs.push_back(attr);
      }
      for (const auto& item : value.list().tensor()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertTensorProto(item, builder));
        attrs.push_back(attr);
      }
      for (const auto& func_attr : value.list().func()) {
        NamedAttrList subattrs;
        for (const auto& subattr : func_attr.attr()) {
          TF_ASSIGN_OR_RETURN(auto attr,
                              ConvertAttributeValue(subattr.second, builder));
          if (subattr.first.empty())
            return InvalidArgument("empty func_attr name");
          subattrs.push_back(builder.getNamedAttr(subattr.first, attr));
        }
        attrs.push_back(FuncAttr::get(builder.getContext(), func_attr.name(),
                                      builder.getDictionaryAttr(subattrs)));
      }
      return builder.getArrayAttr(llvm::ArrayRef(attrs.begin(), attrs.end()));
    }
    case AttrValue::VALUE_NOT_SET:
      return builder.getUnitAttr();
    case AttrValue::kPlaceholder:
      return PlaceholderAttr::get(builder.getContext(), value.placeholder());
    default:
      return tensorflow::errors::Unimplemented(
          absl::StrCat("Attribute ", value.DebugString()));
  }
}

absl::StatusOr<Attribute> ConvertAttributeValue(const AttrValue& value,
                                                Builder& builder) {
  switch (value.value_case()) {
    case AttrValue::kFunc: {
      NamedAttrList attrs;
      for (const auto& func_attr : value.func().attr()) {
        if (func_attr.first.empty()) return InvalidArgument("empty attr name");
        TF_ASSIGN_OR_RETURN(auto attr,
                            ConvertAttributeValue(func_attr.second, builder));
        attrs.push_back(builder.getNamedAttr(func_attr.first, attr));
      }
      auto func_attrs = builder.getDictionaryAttr(attrs);
      return FuncAttr::get(builder.getContext(), value.func().name(),
                           func_attrs);
    }
    default:
      return ConvertNonFuncAttributeValue(value, builder);
  }
}

absl::StatusOr<tf_type::FullTypeAttr> ConvertAttribute(
    const tensorflow::FullTypeDef& full_type, Builder& builder) {
  using FullTypeAttr = ::mlir::tf_type::FullTypeAttr;

  SmallVector<FullTypeAttr> args;
  for (const tensorflow::FullTypeDef& it : full_type.args()) {
    TF_ASSIGN_OR_RETURN(FullTypeAttr arg, ConvertAttribute(it, builder));
    args.push_back(arg);
  }

  Attribute attr;
  switch (full_type.attr_case()) {
    case tensorflow::FullTypeDef::AttrCase::kS:
      attr = builder.getStringAttr(full_type.s());
      break;
    case tensorflow::FullTypeDef::AttrCase::kI:
      attr = builder.getI64IntegerAttr(full_type.i());
      break;
    case tensorflow::FullTypeDef::ATTR_NOT_SET:
      break;
    default:
      return InvalidArgument("Unsupported attr kind in FullType");
  }
  IntegerAttr type_id_attr =
      mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 32),
                             static_cast<int32_t>(full_type.type_id()));
  return FullTypeAttr::get(builder.getContext(), type_id_attr, args, attr);
}

absl::StatusOr<tensorflow::FullTypeDef> ConvertAttribute(
    tf_type::FullTypeAttr full_type) {
  using FullTypeDef = tensorflow::FullTypeDef;

  FullTypeDef ret;
  for (tf_type::FullTypeAttr it : full_type.getArgs()) {
    TF_ASSIGN_OR_RETURN(*ret.add_args(), ConvertAttribute(it));
  }

  if (full_type.getAttr()) {
    bool converted = llvm::TypeSwitch<Attribute, bool>(full_type.getAttr())
                         .Case<StringAttr>([&](StringAttr sattr) {
                           ret.set_s(sattr.str());
                           return true;
                         })
                         .Case<IntegerAttr>([&](IntegerAttr iattr) {
                           ret.set_i(iattr.getInt());
                           return true;
                         })
                         .Default([&](Attribute attr) { return false; });
    if (!converted)
      return InvalidArgument("Unsupported attr kind in FullType:",
                             mlir::debugString(full_type.getAttr()));
  }

  ret.set_type_id(
      static_cast<tensorflow::FullTypeId>(full_type.getTypeId().getInt()));

  return ret;
}

absl::StatusOr<ArrayAttr> ConvertHandleData(
    Builder builder,
    const tensorflow::protobuf::RepeatedPtrField<
        tensorflow::ResourceHandleProto_DtypeAndShape>& handle_data) {
  SmallVector<Attribute> dtype_and_shape;
  for (const auto& handle : handle_data) {
    if (handle.dtype() == tensorflow::DT_INVALID)
      return InvalidArgument("Invalid dtype for handle_data");
    Type dtype;
    TF_RETURN_IF_ERROR(ConvertDataType(handle.dtype(), builder, &dtype));
    TF_ASSIGN_OR_RETURN(
        ShapeAttr shape,
        ConvertTensorShapeProto(handle.shape(), builder.getContext()));
    TensorType handle_type;
    if (shape.hasRank()) {
      handle_type = RankedTensorType::get(shape.getShape(), dtype);
    } else {
      handle_type = UnrankedTensorType::get(dtype);
    }
    dtype_and_shape.push_back(TypeAttr::get(handle_type));
  }
  return builder.getArrayAttr(dtype_and_shape);
}

Status ConvertHandleData(ArrayAttr handle_data_arr,
                         tensorflow::OpDef::ArgDef* arg) {
  if (!handle_data_arr) return {};
  for (auto handle_data_attr : handle_data_arr.getAsRange<TypeAttr>()) {
    TensorType handle_type =
        mlir::dyn_cast<TensorType>(handle_data_attr.getValue());
    if (!handle_type) {
      return InvalidArgument("Expected an array of tensor types, but got ",
                             debugString(handle_data_arr));
    }
    auto* handle_data = arg->add_handle_data();
    if (handle_type.hasRank()) {
      ConvertToTensorShapeProto(handle_type.getShape(),
                                handle_data->mutable_shape());
    } else {
      handle_data->mutable_shape()->set_unknown_rank(true);
    }
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(handle_type.getElementType(), &dtype));
    handle_data->set_dtype(dtype);
  }
  return {};
}

}  // namespace tfg
}  // namespace mlir
