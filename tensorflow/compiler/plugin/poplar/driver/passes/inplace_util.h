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

namespace xla {
namespace poplarplugin {
struct CompilerAnnotations;
namespace InplaceUtil {
using OperandIndexes = absl::InlinedVector<uint64, 1>;
using InplaceInstructions = absl::flat_hash_set<const HloInstruction*>;

// Internal representations of the Types of instructions.
// An instruction is either an inplace op or not inplace op.

// Base Hlo instruction op type.
class HloInstructionDescription {
 public:
  // Returns true if given description is of InplaceHloInstructionDescription
  // type given a instruction.
  virtual bool IsInPlaceType(const HloInstruction*);

 protected:
  HloInstructionDescription();
};

// An instruction which has no aliasing between its input and output tensors.
class NotInplaceHloInstructionDescription : public HloInstructionDescription {
 public:
  NotInplaceHloInstructionDescription();
};

// An instruction which overwrites the output tensors of all inplace_operands
// instructions.
class InplaceHloInstructionDescription : public HloInstructionDescription {
 public:
  InplaceHloInstructionDescription();
  InplaceHloInstructionDescription(
      const OperandIndexes& inplace_operand_indexes);
  bool IsInPlaceType(const HloInstruction*);
  const OperandIndexes GetInplaceOperandIndexes() const;

 private:
  OperandIndexes inplace_operand_indexes_;
};

std::unique_ptr<HloInstructionDescription> GetHloInstructionDescription(
    const HloInstruction* inst);

// A function which is used to decide whether an instruction is of inplace or
// view changing or not-inplace type given our backend implementation of these
// ops in Poplar.
bool IsInPlace(HloInstruction* inst, HloReachabilityMap* reachability_map);
}  // namespace InplaceUtil

}  // namespace poplarplugin
}  // namespace xla

#endif
