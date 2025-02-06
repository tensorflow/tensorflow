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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONSTANT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONSTANT_UTILS_H_

#include "absl/status/statusor.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "tsl/platform/statusor.h"

namespace mlir {
namespace TFL {

// Returns a Constant op with a single value.
absl::StatusOr<arith::ConstantOp> CreateConstOpWithSingleValue(
    PatternRewriter* rewriter, Location loc, ShapedType shaped_type, int value);

// Returns a Constant op with a splat vector value.
absl::StatusOr<arith::ConstantOp> CreateConstOpWithVectorValue(
    PatternRewriter* rewriter, Location loc, ShapedType shaped_type, int value);

}  // namespace TFL
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONSTANT_UTILS_H_
