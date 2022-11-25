/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DOT_OP_EMITTER_INTERNAL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DOT_OP_EMITTER_INTERNAL_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"

// -----------------------------------------------------------------------------
// INTERNAL HEADER.
//
// This file exposes internal implementation details from dot_op_emitter.cc for
// unit tests.  Please do not depend on this!
//
// -----------------------------------------------------------------------------

namespace xla {
namespace cpu {
namespace internal {

// Represents a dot operation.  We use this in lieu of an `HloInstruction`
// because we want to be able to create this for the "inner" dot operation in a
// batch dot, for which there is no separate HLO instruction.
struct DotInfo {
  Shape lhs_shape;
  Shape rhs_shape;
  Shape result_shape;
  DotDimensionNumbers dim_nums;

  explicit DotInfo(const HloInstruction& instr) {
    CHECK_EQ(instr.opcode(), HloOpcode::kDot);
    lhs_shape = instr.operand(0)->shape();
    rhs_shape = instr.operand(1)->shape();
    result_shape = instr.shape();
    dim_nums = instr.dot_dimension_numbers();
  }
};

// Dictates how a dot operation is implemented.
enum class DotImplementationStrategy {
  // The dot operation is lowered into LLVM IR that implements a naive nested
  // loop that computes the result one element at a time.  This is our
  // "fallback"; we don't really want this to kick in for any non-trival dot
  // operation.
  kNaiveLlvmIr,

  // The dot operation is lowered into LLVM IR that implements a tiled
  // Matrix*Vector operation.  This strategy also allows fusing in a bias add
  // into the dot.  The matrix can be row major or column major, both are
  // supported.
  kTiledLlvmIrGemv,

  // The dot operation is lowered into LLVM IR that implements a tiled
  // Matrix*Matrix operation.  No fusions are supported.  The two inputs
  // and the output have to be row major.
  kTiledLlvmIrGemm,

  // The dot operation is lowered into a call into an Eigen routine.  No fusions
  // are supported today.  The two inputs and the output have to be row major.
  // However, we do allow transposing either the LHS or the RHS as part of the
  // GEMM -- we expose this flexibility as flexibility in the contraction
  // dimensions, but we can also see this as flexibility in the input layouts.
  kEigen,
};

// Returns the implementation strategy for a dot with the configuration
// `dot_info`.
DotImplementationStrategy GetDotImplementationStrategy(
    const HloModuleConfig& config, const DotInfo& dot_info,
    const TargetMachineFeatures& target_machine_features);
}  // namespace internal
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DOT_OP_EMITTER_INTERNAL_H_
