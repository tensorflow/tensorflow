/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {

// Ensure an attribute named attr_name exists and it is of type AttrType.
// If so, sets the `out_attr` pointer to point to the casted attribute.
template <typename AttrType>
bool EnsureAttribute(const DictionaryAttr& composite_attributes,
                     const std::string& attr_name, AttrType* out_attr) {
  Attribute attr = composite_attributes.get(attr_name);
  if (!mlir::isa_and_nonnull<AttrType>(attr)) {
    return false;
  }
  if (AttrType content = mlir::dyn_cast<AttrType>(attr)) {
    *out_attr = content;
    return true;
  } else {
    return false;
  }
}

// Changes a DenseIntElementsAttr **containing I64** elements to an I32 Vector.
bool DenseI64AttrToI32Vector(const DenseIntElementsAttr& dense_attr,
                             std::vector<int32_t>* out_vec);

// Gets boolean from composite attrs if it exists.
std::optional<bool> GetBoolFromCompositeAttr(
    const DictionaryAttr& composite_attrs, llvm::StringRef attr_name);

// Given a DictionaryAttr, checks if it has a DenseIntElementsAttr attribute
// with the name attr_name. If so, extracts its values and stores as a vector
// of int32_t elements.
// Note: This assumes the DenseIntElementsAttr has its values stored as int64_t.
bool GetI32VectorFromDenseI64CompositeAttr(
    const DictionaryAttr& composite_attrs, const std::string& attr_name,
    std::vector<int32_t>* out_vec);

// Get a DenseIntElementsAttr of type I64 and convert it to an I32 attribute.
DenseIntElementsAttr DenseI64AttrToI32Attr(
    const DenseIntElementsAttr& dense_attr, PatternRewriter& builder);

// Returns a NHWC shaped type from an NCHW shaped type op.
// For example- Given a Composite op that wraps a core.aten.avg_pool2d, this
// returns the return type of the tfl.average_pool_2d emitted. Note that the
// aten.avg_pool2d works with the NCHW layout while tfl.average_pool_2d assumes
// NHWC.
ShapedType GetNhwcReturnTypeFromNchw(Operation* old_op);

}  // namespace odml

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_COMPOSITE_UTILS_H_
