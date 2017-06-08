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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_USER_COMPUTATION_FLAGS_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_USER_COMPUTATION_FLAGS_H_

// Legacy flags for XLA's user_computation module.

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Append to *flag_list flags definitions associated with XLA's user_computation
// module.
void AppendUserComputationFlags(std::vector<tensorflow::Flag>* flag_list);

typedef struct {
  // Eliminate implicit broadcast on when lowering user computation to HLO
  // instructions, use explicit broadcast instead.
  bool xla_eliminate_hlo_implicit_broadcast;
} UserComputationFlags;

// Return a pointer to the UserComputationFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
UserComputationFlags* GetUserComputationFlags();

}  // namespace legacy_flags
}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_USER_COMPUTATION_FLAGS_H_
