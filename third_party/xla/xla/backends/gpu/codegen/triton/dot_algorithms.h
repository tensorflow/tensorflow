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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_DOT_ALGORITHMS_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_DOT_ALGORITHMS_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla {
namespace gpu {
namespace triton {

// Carries named `Value`s corresponding to `dot` operands. This includes an
// accumulator.
struct DotOperands {
  ::mlir::Value lhs;
  ::mlir::Value rhs;
  ::mlir::Value accumulator;
};

// Carries named `Value`s corresponding to `scaled-dot` operands. This includes
// an accumulator and their respective scaling factors.
struct ScaledDotOperands {
  ::mlir::Value lhs;
  ::mlir::Value rhs;
  ::mlir::Value lhs_scale;
  ::mlir::Value rhs_scale;
  ::mlir::Value accumulator;
};

// Returns the type to use for accumulation for the given `dot` instruction.
// This also handles the case where the algorithm is `ALG_UNSET`.
absl::StatusOr<::mlir::Type> GetDotAccumulatorType(
    EmitterLocOpBuilder b, const HloDotInstruction& dot);

// Emits a single-tile dot, considering the given `dot` instruction's algorithm
// and operand precisions. Raises an `UnimplementedError` if the algorithm is
// not supported.
absl::StatusOr<::mlir::Value> EmitSingleTileDot(EmitterLocOpBuilder b,
                                                const HloDotInstruction& dot,
                                                DotOperands dot_operands);

// Emits a single-tile scaled-dot, considering the given `scaled-dot`
// instruction's operand precisions. Raises an `InvalidArgumentError` if the
// operand types are not supported.
absl::StatusOr<::mlir::Value> EmitSingleTileScaledDot(
    EmitterLocOpBuilder b, const HloScaledDotInstruction& scaled_dot,
    ScaledDotOperands dot_operands);

namespace internal {
absl::StatusOr<mlir::triton::ScaleDotElemType> GetScaleDotElemType(
    mlir::Type value);

}  // namespace internal

}  // namespace triton
}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_DOT_ALGORITHMS_H_
