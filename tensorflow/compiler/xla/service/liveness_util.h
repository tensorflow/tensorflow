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

// A collection of utilities on the HLO graph.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LIVENESS_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LIVENESS_UTIL_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Returns true if 'user' cannot possibly use the buffer at 'index' in
// 'operand'. Returns false otherwise.
//
// REQUIRES: 'operand' is an operand of 'user'.
bool DoesNotUseOperandBuffer(const HloInstruction* operand,
                             const ShapeIndex& index,
                             const HloInstruction* user,
                             const TuplePointsToAnalysis& points_to_analysis);

// Returns true if 'user' (at 'user_index') can share a buffer with its operand
// 'operand' (at 'operand_index'). Returns false otherwise. Optionally takes a
// points-to analysis argument. Without the analysis, the result is more
// conservative (returns false more often).
//
// REQUIRES: 'operand' is an operand of 'user'.
bool CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index,
    const TuplePointsToAnalysis* points_to_analysis = nullptr);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LIVENESS_UTIL_H_
