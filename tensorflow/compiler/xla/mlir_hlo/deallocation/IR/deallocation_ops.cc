/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "deallocation/IR/deallocation_ops.h"

#include "deallocation/IR/deallocation_dialect.cc.inc"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define GET_TYPEDEF_CLASSES
#include "deallocation/IR/deallocation_typedefs.cc.inc"
#undef GET_TYPEDEF_CLASSES

namespace mlir {
namespace deallocation {

void DeallocationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "deallocation/IR/deallocation_ops.cc.inc"
#undef GET_OP_LIST
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "deallocation/IR/deallocation_typedefs.cc.inc"
#undef GET_TYPEDEF_LIST
      >();
}

void OwnOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  Value memref) {
  return build(odsBuilder, odsState,
               OwnershipIndicatorType::get(odsBuilder.getContext()), memref);
}

void NullOp::build(OpBuilder& odsBuilder, OperationState& odsState) {
  return build(odsBuilder, odsState,
               OwnershipIndicatorType::get(odsBuilder.getContext()));
}

}  // namespace deallocation
}  // namespace mlir

#define GET_OP_CLASSES
#include "deallocation/IR/deallocation_ops.cc.inc"
