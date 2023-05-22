#ifndef TENSORFLOW_COMPILER_XLA_FRONTEND_ATTRIBUTES_H_
#define TENSORFLOW_COMPILER_XLA_FRONTEND_ATTRIBUTES_H_
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
#include <string>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Attribute which indicates that an in-place instruction has disjoint read
// and write regions w.r.t aliased input/output buffers.
inline constexpr char kXlaDisjointReadWriteRegions[] =
    "_xla_disjoint_read_write_regions";

// Set frontend attribute on 'instruction' which indices that in-place
// 'instruction' has disjoint read/write buffer regions.
void SetDisjointReadWriteRegionsAttr(HloInstruction* instruction);

// Returns 'true' if in-place 'instruction' has the kXlaDisjointReadWriteRegions
// frontend attribute set (returns false otherwise).
bool HasDisjointReadWriteRegionsAttr(HloInstruction* instruction);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_FRONTEND_ATTRIBUTES_H_
