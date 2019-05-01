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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_INPLACE_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_INPLACE_UTIL_H_

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

#include <set>
namespace xla {
namespace poplarplugin {
struct CompilerAnnotations;

using OperandIndexes = std::vector<int64>;
using InplaceInstructions = absl::flat_hash_set<const HloInstruction*>;
using InplaceWorkList = absl::flat_hash_map<HloInstruction*, bool>;

enum class HloInstructionType {
  // A kGetTupleElement instruction is inplace if and only if it's a unique
  // access to a tensor in a tuple and all other users of the tuple are
  // GetTupleElement ops too.
  kInplaceGetTupleElement = 0,
  // An instruction is inplace read/write when the output is input tensor(s)
  // which are modified by this instruction.
  kInplaceReadWrite,
  // An instruction is inplace read-only when the output is reference to the
  // input tensor(s) which are not modified by this instruction.
  kInplaceReadOnly,
  // An instruction is not inplace when the output does not alias any of the
  // inputs.
  kNotInplace,
};

// Internal representations of the Types of instructions.
class HloInstructionDescription {
 public:
  HloInstructionDescription(const HloInstruction* inst);

  // Get the HloInstructionType.
  const HloInstructionType& GetType() const;

  // Get the inplace operands.
  const OperandIndexes& GetInplaceOperandIndexes() const;

  // Checks if the type is kInplaceReadWrite or kInplaceReadOnly.
  bool IsInplaceType() const;

  static bool IsInplace(HloInstruction* inst,
                        HloReachabilityMap* reachability_map,
                        InplaceWorkList& worklist,
                        const InplaceInstructions& inplace_instructions);

  const std::string ToString() const;

 private:
  HloInstructionDescription();

  OperandIndexes inplace_operands_;

  HloInstructionType type_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif
