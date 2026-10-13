/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/tensor_ir/transforms/type_conversion.h"

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::nv_tensor_ir::xla {

TensorIrTypeConverter::TensorIrTypeConverter() {
  // Pass-through fallback for types that do not need conversion (e.g. floats).
  addConversion([](mlir::Type type) { return type; });

  // Map signless IntegerType of width 8/16/32/64 to signed IntegerType.
  // i1 remains unchanged. Unsigned integers remain unchanged.
  addConversion([](mlir::IntegerType type) -> mlir::Type {
    if (type.isSignless() && (type.getWidth() == 8 || type.getWidth() == 16 ||
                              type.getWidth() == 32 || type.getWidth() == 64)) {
      return mlir::IntegerType::get(type.getContext(), type.getWidth(),
                                    mlir::IntegerType::Signed);
    }
    return type;
  });

  // Map RankedTensorType recursively: same shape, converted element type,
  // encoding dropped (nv_tensor_ir rejects non-null encodings).
  addConversion(
      [this](mlir::RankedTensorType type) -> std::optional<mlir::Type> {
        mlir::Type elem_type = convertType(type.getElementType());
        if (!elem_type) {
          return std::nullopt;
        }
        return mlir::RankedTensorType::get(type.getShape(), elem_type);
      });

  // Convert FunctionType: convert all input and result types.
  addConversion([this](mlir::FunctionType type) -> std::optional<mlir::Type> {
    llvm::SmallVector<mlir::Type> inputs;
    llvm::SmallVector<mlir::Type> results;
    if (failed(convertTypes(type.getInputs(), inputs)) ||
        failed(convertTypes(type.getResults(), results))) {
      return std::nullopt;
    }
    return mlir::FunctionType::get(type.getContext(), inputs, results);
  });
}

}  // namespace mlir::nv_tensor_ir::xla
