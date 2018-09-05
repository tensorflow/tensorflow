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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

// Returns the precise trip count of the loop if it's statically known,
// nullopt otherwise. max_value_returned limits the number of steps that are
// evaluated while trying to brute force a loop trip count, trip counts larger
// than max_value_returned result in nullopt.
absl::optional<int64> ComputeWhileLoopTripCount(HloInstruction *while_op,
                                                int64 max_value_returned = 128);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_
