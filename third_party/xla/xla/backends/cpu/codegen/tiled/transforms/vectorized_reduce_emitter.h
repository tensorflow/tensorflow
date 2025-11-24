/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_VECTORIZED_REDUCE_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_VECTORIZED_REDUCE_EMITTER_H_

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace xla::cpu {

// Create a vectorized reduction of the given source vector.
//
// The implementation is as follows:
// 1. If the reduction dimension is only the most minor we convert it into a
//    nested scf.loop of horizonal reductions and if the body of the reduce is a
//    single binary operation that is supported by ReductionOp we use that,
//    otherwise we simply loop over the scalar values.
// 2. If the reduction dimensions does not include the most minor dimension, we
//    loop over the reductions dimensions and apply the body with vectorized
//    inputs.
// 3. If the dimensions are a combindation of minor & non-minor dimensions we
//    simply apply strategy 2 followed by strategy 1.
mlir::Value EmitVectorizedReduction(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType result_type,
    mlir::TypedValue<mlir::RankedTensorType> source, mlir::Value init_value,
    llvm::ArrayRef<int64_t> reduction_dims, mlir::Block& body);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TILED_TRANSFORMS_VECTORIZED_REDUCE_EMITTER_H_
