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

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {
namespace tf_type {
class FullTypeAttr;
}  // namespace tf_type

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
    const tensorflow::AttrValue& value, Builder& builder,
    TFGraphDialect* tfgDialect);

// Converts all kinds of AttrValue proto into an MLIR attribute.
tensorflow::StatusOr<Attribute> ConvertAttributeValue(
    const tensorflow::AttrValue& value, Builder& builder,
    TFGraphDialect* tfgDialect);

// Convert the MLIR FullTyoe attribute `attr` and return a
// `tensorflow::FullTypeDef`.
tensorflow::StatusOr<tensorflow::FullTypeDef> ConvertAttribute(
    tf_type::FullTypeAttr full_type);

// Converts fulltype proto to attribute.
tensorflow::StatusOr<::mlir::tf_type::FullTypeAttr> ConvertAttribute(
    const tensorflow::FullTypeDef& full_type, Builder& builder,
    TFGraphDialect* tfgDialect);

// Certain load-bearing TF attributes are promoted to TFG attributes by dropping
// the underscore and adding a `tfg.` prefix. This is so that transformations
// are made aware that these are attributes that must be handled with care.
StringRef PromoteToTFGAttribute(StringRef tf_attr_name);
// Prepare promoted TFG attributes for export by converting them back to their
// original names.
StringRef PrepareTFGAttributeForExport(StringRef tfg_attr_name);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_EXPORT_UTILS_H_
