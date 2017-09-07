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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_GPU_WHILE_TRANSFORMER_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_GPU_WHILE_TRANSFORMER_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace gpu {

// Runs an analysis of the while loop instruction 'while_hlo' (and its
// associated sub-computations) to determine if it can be transformed into an
// equivalent "for" loop with the following "for" loop parameters:
//
// *) 'loop_start': loop induction variable starting value.
// *) 'loop_limit': loop induction variable limit value.
// *) 'loop_increment': loop induction variable per-iteration increment value.
//
// Returns an std::tuple = (loop_start, loop_limit, loop_increment) on success.
// The values in the returned tuple are values extracted from the 'while_hlo'
// operand (and its sub-computations) during analysis.
// Returns an error status on failure.
StatusOr<std::tuple<int64, int64, int64>> CanTransformWhileToFor(
    const HloInstruction* while_hlo);

}  // namespace gpu
}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_GPU_WHILE_TRANSFORMER_H_
