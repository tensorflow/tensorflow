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

#include "tensorflow/compiler/mlir/tensorflow/utils/convert_attr.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

// Converts non func AttrValue proto into an MLIR attribute. Func attribute is
// exclused in this function because the function might be renamed when the
// function definition is imported.
StatusOr<mlir::Attribute> ConvertNonFuncAttributeValue(const AttrValue& value,
                                                       mlir::Builder* builder) {
  switch (value.value_case()) {
    case AttrValue::kI:
      return builder->getI64IntegerAttr(value.i());
    case AttrValue::kS:
      return builder->getStringAttr(value.s());
    case AttrValue::kF:
      return builder->getFloatAttr(builder->getF32Type(), value.f());
    case AttrValue::kB:
      return builder->getBoolAttr(value.b());
    case AttrValue::kType: {
      mlir::Type type;
      TF_RETURN_IF_ERROR(ConvertDataType(value.type(), *builder, &type));
      return mlir::TypeAttr::get(type);
    }
    case AttrValue::kShape:
      return ConvertTensorShapeProto(value.shape(), builder->getContext());
    case AttrValue::kTensor:
      return ConvertTensorProto(value.tensor(), builder);
    case AttrValue::kList: {
      absl::InlinedVector<mlir::Attribute, 8> attrs;
      for (const auto& item : value.list().i())
        attrs.push_back(builder->getI64IntegerAttr(item));
      for (const auto& item : value.list().s())
        attrs.push_back(builder->getStringAttr(item));
      for (const auto& item : value.list().f())
        attrs.push_back(builder->getFloatAttr(builder->getF32Type(), item));
      for (const auto& item : value.list().b())
        attrs.push_back(builder->getBoolAttr(item));
      for (const auto& item : value.list().type()) {
        mlir::Type type;
        TF_RETURN_IF_ERROR(ConvertDataType(DataType(item), *builder, &type));
        attrs.push_back(mlir::TypeAttr::get(type));
      }
      for (const auto& item : value.list().shape()) {
        TF_ASSIGN_OR_RETURN(
            auto attr, ConvertTensorShapeProto(item, builder->getContext()));
        attrs.push_back(attr);
      }
      for (const auto& item : value.list().tensor()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertTensorProto(item, builder));
        attrs.push_back(attr);
      }
      if (!value.list().func().empty()) {
        return tensorflow::errors::Unimplemented(
            absl::StrCat("Attribute ", value.DebugString()));
      }
      return builder->getArrayAttr(
          llvm::makeArrayRef(attrs.begin(), attrs.end()));
    }
    case AttrValue::VALUE_NOT_SET:
      return builder->getUnitAttr();
    // kPlaceholder is not implemented.
    default:
      return tensorflow::errors::Unimplemented(
          absl::StrCat("Attribute ", value.DebugString()));
  }
}

StatusOr<mlir::Attribute> ConvertAttributeValue(const AttrValue& value,
                                                mlir::Builder* builder) {
  switch (value.value_case()) {
    case AttrValue::kFunc: {
      // TODO(b/156546237): Unify kFunc/NameAttrList attribute representation.
      // Currently kFunc/NameAttrList attributes in a kList/repeated AttrValue
      // will not use this representation.
      mlir::NamedAttrList attrs;
      for (const auto& func_attr : value.func().attr()) {
        TF_ASSIGN_OR_RETURN(auto attr,
                            ConvertAttributeValue(func_attr.second, builder));
        attrs.push_back(builder->getNamedAttr(func_attr.first, attr));
      }
      auto func_attrs = builder->getDictionaryAttr(attrs);
      return mlir::TF::FuncAttr::get(builder->getContext(), value.func().name(),
                                     func_attrs);
    }
    default:
      return ConvertNonFuncAttributeValue(value, builder);
  }
}

}  // namespace tensorflow
