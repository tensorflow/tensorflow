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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_UTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace mhlo {

// Builds body for reduce op by using the template binary op as the
// reducer op.
template <typename Op>
void BuildReduceBody(Type element_type, Region* body, OpBuilder* builder) {
  OpBuilder::InsertionGuard guard(*builder);
  Block* block = builder->createBlock(body);

  // Block arguments are scalars of the given element type.
  Type type = RankedTensorType::get(/*shape=*/{}, element_type);
  Location loc = body->getLoc();
  block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));

  auto reducer =
      builder->create<Op>(loc, block->getArgument(0), block->getArgument(1));
  builder->create<ReturnOp>(loc, reducer.getResult());
}

ConstOp GetScalarConstOfType(Type ty, Location loc, int64_t raw_value,
                             OpBuilder* builder);

ConstOp GetScalarNegZeroOfType(Type ty, Location loc, OpBuilder* builder);

// Converts an ArrayAttr to a 1D 64-bit dense elements attribute.
DenseIntElementsAttr GetI64ElementsAttr(ArrayAttr attr);
DenseIntElementsAttr GetI64ElementsAttr(llvm::ArrayRef<int64_t> values,
                                        Builder* builder);

}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_UTILS_H_
