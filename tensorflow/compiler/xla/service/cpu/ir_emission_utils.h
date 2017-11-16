/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace cpu {

bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution);

bool PotentiallyImplementedAsEigenDot(const HloInstruction& dot);

enum class DotInLlvmIrProfitable { kYes, kNo, kWithColumnMajorRhs };

// Returns a value to indicate if (and under what conditions) will lowering
// |dot| as a untiled LLVM IR dot operation be profitable over calling into
// Eigen or emitting a tiled LLVM IR implementation.  Possible return values
// are:
//
//  * DotInLlvmIrProfitable::kYes - always profitable.
//  * DotInLlvmIrProfitable::kNo - never profitable.
//  * DotInLlvmIrProfitable::kWithColumnMajorRhs - only if we can manage to make
//    the Rhs layout column major.
DotInLlvmIrProfitable ProfitableToImplementDotInUntiledLlvmIr(
    const HloInstruction& dot);

// Returns true to indicate that we can generate a tiled LLVM IR implementation
// for |dot|.
bool ProfitableToImplementDotInTiledLlvmIr(const HloInstruction& dot);

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_
