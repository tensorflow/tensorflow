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

#include "xla/service/gpu/fusions/triton/xla_triton_ops.h"

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"  // IWYU pragma: keep
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"  // IWYU pragma: keep
#include "mlir/IR/ValueRange.h"
#include "xla/service/gpu/fusions/triton/xla_triton_dialect.cc.inc"

namespace xla {
namespace triton {

void XlaTritonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/service/gpu/fusions/triton/xla_triton_ops.cc.inc"
      >();
}

::llvm::LogicalResult SparseDotOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  return ::llvm::success();
}

::llvm::LogicalResult SparseDotOp::verify() { return ::llvm::success(); }

}  // namespace triton
}  // namespace xla

#define GET_OP_CLASSES
#include "xla/service/gpu/fusions/triton/xla_triton_ops.cc.inc"
