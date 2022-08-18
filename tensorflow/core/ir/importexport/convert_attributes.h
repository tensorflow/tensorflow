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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_EXPORT_UTILS_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_EXPORT_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/platform/statusor.h"

namespace mlir {
namespace tfg {

// Convert the list of MLIR Attributes `attrs` to the `tensorflow::AttrValueMap`
// `values`.
tensorflow::Status ConvertAttributes(ArrayRef<NamedAttribute> attrs,
                                     ArrayRef<StringRef> attrs_to_ignore,
                                     bool remove_ref_type,
                                     tensorflow::AttrValueMap* values);

// Convert the MLIR attribute `attr` and return a `tensorflow::AttrValue`.
tensorflow::StatusOr<tensorflow::AttrValue> ConvertAttribute(Attribute attr);

tensorflow::Status SetShapeAttribute(absl::string_view name,
                                     ShapedType shaped_type,
                                     tensorflow::AttrValueMap* values);

// Converts an MLIR shaped type to a TensorFlow shape attribute.
ShapeAttr ConvertTypeToTensorShapeAttr(const Type& type);

/// Import from TensorFlow to MLIR

// Converts non func AttrValue proto into an MLIR attribute. Func attribute is
// exclused in this function because the function might be renamed when the
// function definition is imported.
tensorflow::StatusOr<Attribute> ConvertNonFuncAttributeValue(
    const tensorflow::AttrValue& value, Builder& builder);

// Converts all kinds of AttrValue proto into an MLIR attribute.
tensorflow::StatusOr<Attribute> ConvertAttributeValue(
    const tensorflow::AttrValue& value, Builder& builder);

// Convert the MLIR FullTyoe attribute `attr` and return a
// `tensorflow::FullTypeDef`.
tensorflow::StatusOr<tensorflow::FullTypeDef> ConvertAttribute(
    tf_type::FullTypeAttr full_type);

// Converts fulltype proto to attribute.
tensorflow::StatusOr<::mlir::tf_type::FullTypeAttr> ConvertAttribute(
    const tensorflow::FullTypeDef& full_type, Builder& builder);

// Convert an array of handle data (pairs of data types and shapes) to an array
// attribute of tensor types.
tensorflow::StatusOr<ArrayAttr> ConvertHandleData(
    Builder builder,
    const tensorflow::protobuf::RepeatedPtrField<
        tensorflow::ResourceHandleProto_DtypeAndShape>& handle_data);

// Convert an array of handle data into the `handle_data` field of the provided
// ArgDef. Each entry of the array is expected to be a TensorType.
tensorflow::Status ConvertHandleData(ArrayAttr handle_data_arr,
                                     tensorflow::OpDef::ArgDef* arg);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_EXPORT_UTILS_H_
