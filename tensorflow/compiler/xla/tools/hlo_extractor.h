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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_HLO_EXTRACTOR_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_HLO_EXTRACTOR_H_

#include <functional>
#include <memory>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"

namespace xla {

// Define HloSelector, which is a lambda that, given an HLO
// instruction, returns true if selected, otherwise return false.
using HloSelector = std::function<bool(const HloInstruction*)>;

// Creates a new HLO module rooted with an entry computation rooted at the given
// instruction.
//
//  By default (height == -1), the new computation includes all transitive
//  operands of `root`.  If you specify a different height, the new computation
//  will include all instructions <= `height` hops away from `root`.
//  Instructions at the boundary are replaced by parameters.
//
// The `hlo_selector` will return true/false for each hlo instruction. If false
// is returned, the corresponding instruction and its predecessors will not be
// included in the extracted hlo module
std::unique_ptr<HloModule> ExtractModule(HloInstruction* instruction,
                                         int64_t height = -1,
                                         HloSelector hlo_selector = nullptr);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_HLO_EXTRACTOR_H_
