/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_
#define XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace cpu {

bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features);

// Computes the minimum alignment guaranteed for a tensor of shape `shape` on
// the target machine.
int64_t GetMinimumAlignmentForArray(
    const Shape& shape, const TargetMachineFeatures& target_machine_features);

// Dynamic loop bounds are specified as an array of dimension index
// [start, limit) pairs of ir values (one for each partitioned outer dimension).
//
// EX: Let 'shape' = [8, 16, 32], with the loop bounds of the two-most major
//     dimensions dynamic. Then 'dynamic_loop_bounds' will contain the
//     following ir values for the two most-major dimensions:
//       [dim0_index_start_ir_value, dim0_index_limit_ir_value]
//       [dim1_index_start_ir_value, dim1_index_limit_ir_value]
//
// See IrFunction and ParallelLoopEmitter for details.
using DynamicLoopBounds = std::vector<std::pair<llvm::Value*, llvm::Value*>>;

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_
