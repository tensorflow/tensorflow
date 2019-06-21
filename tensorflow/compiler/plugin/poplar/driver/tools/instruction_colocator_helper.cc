/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/instruction_colocator_helper.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include "absl/memory/memory.h"
#include "absl/types/optional.h"

namespace xla {
namespace poplarplugin {

InstructionColocatorHelper::InstructionColocatorHelper() : id_(GetNextID()) {}

int64 InstructionColocatorHelper::GetID() const { return id_; }

int64 InstructionColocatorHelper::GetNextID() {
  static int64 id = 0;
  return id++;
}

bool InstructionColocatorHelper::CanColocateExtra(
    const HloInstruction* a, const HloInstruction* b) const {
  return true;
}

bool InstructionColocatorHelper::CanColocate(const HloInstruction* a,
                                             const HloInstruction* b) const {
  if (!CanColocate(a) || !CanColocate(b)) {
    return false;
  }
  // We don't support tuple shapes.
  if (a->shape().IsTuple() || b->shape().IsTuple()) {
    return false;
  }
  // Make sure a and b have compitable sharding.
  if (!a->has_compatible_sharding(b)) {
    return false;
  }
  return CanColocateExtra(a, b);
}

bool InstructionColocatorHelperPtrComparator::operator()(
    const InstructionColocatorHelper* const& lhs,
    const InstructionColocatorHelper* const& rhs) const {
  if (rhs == nullptr) {
    // Nothing compares less than nullptr.
    return false;
  }
  if (lhs == nullptr) {
    return true;
  }
  return lhs->GetID() < rhs->GetID();
}

namespace {
// Manager for all the colocators.
class InstructionColocatorHelperManager {
 public:
  static InstructionColocatorHelperManager& GetInstance() {
    static InstructionColocatorHelperManager instance;
    return instance;
  }

  void AddInstructionColocatorHelper(
      std::unique_ptr<InstructionColocatorHelper> colocator) {
    colocators.push_back(std::move(colocator));
    colocators_refs.push_back(colocators.back().get());
  }

  const std::vector<const InstructionColocatorHelper*>&
  GetAllInstructionColocatorHelpers() const {
    return colocators_refs;
  }

 private:
  InstructionColocatorHelperManager() {}

  std::vector<std::unique_ptr<InstructionColocatorHelper>> colocators;
  std::vector<const InstructionColocatorHelper*> colocators_refs;
};

// Registrar
class InstructionColocatorHelperRegistrar {
 public:
  InstructionColocatorHelperRegistrar(
      std::unique_ptr<InstructionColocatorHelper> colocator) {
    InstructionColocatorHelperManager::GetInstance()
        .AddInstructionColocatorHelper(std::move(colocator));
  }

  InstructionColocatorHelperRegistrar() = delete;
};

#define REGISTER_INSTRUCTION_COLLOCATOR_HELPER(colocator)              \
  namespace {                                                          \
  static InstructionColocatorHelperRegistrar                           \
      registrar__colocator__##colocator##__object(                     \
          std::unique_ptr<InstructionColocatorHelper>(new colocator)); \
  }

// Colocator helper which is used to combine multiple all reduce instructions.
class AllReduceColocatorHelper : public InstructionColocatorHelper {
 public:
  AllReduceColocatorHelper() : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return inst->opcode() == HloOpcode::kAllReduce;
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_all_reduce_buffer_size;
  }

 protected:
  bool CanColocateExtra(const HloInstruction* a,
                        const HloInstruction* b) const override {
    // Make sure the same to_apply() computation is used.
    return *a->to_apply() == *b->to_apply();
  }
};

// Colocator helper which is used to combine multiple inter IPU copies.
class InterIpuCopyColocatorHelper : public InstructionColocatorHelper {
 public:
  InterIpuCopyColocatorHelper() : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsInterIpuCopy(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_inter_ipu_copies_buffer_size;
  }
};

// Colocator helper which is used to combine multiple gradient accumulations and
// all reduce instructions.
class StatefulGradientAccumulationAllReduceColocatorHelper
    : public InstructionColocatorHelper {
 public:
  StatefulGradientAccumulationAllReduceColocatorHelper()
      : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return DynCast<HloStatefulGradientAccumulateAndAllReduce>(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_all_reduce_buffer_size;
  }

 protected:
  bool CanColocateExtra(const HloInstruction* a,
                        const HloInstruction* b) const override {
    auto a_cast = Cast<HloStatefulGradientAccumulateAndAllReduce>(a);
    auto b_cast = Cast<HloStatefulGradientAccumulateAndAllReduce>(b);
    // Make accumulate the same number of batches.
    return a_cast->MiniBatchesToAccumulate() ==
           b_cast->MiniBatchesToAccumulate();
  }
};

}  // namespace

REGISTER_INSTRUCTION_COLLOCATOR_HELPER(InterIpuCopyColocatorHelper)
REGISTER_INSTRUCTION_COLLOCATOR_HELPER(AllReduceColocatorHelper)
REGISTER_INSTRUCTION_COLLOCATOR_HELPER(
    StatefulGradientAccumulationAllReduceColocatorHelper)

const std::vector<const InstructionColocatorHelper*>&
GetAllInstructionColocatorHelpers() {
  return InstructionColocatorHelperManager::GetInstance()
      .GetAllInstructionColocatorHelpers();
}

absl::optional<const InstructionColocatorHelper*> GetInstructionColocatorHelper(
    const HloInstruction* inst) {
  for (auto colocator : GetAllInstructionColocatorHelpers()) {
    if (colocator->CanColocate(inst)) {
      return colocator;
    }
  }
  return absl::nullopt;
}

bool CanColocate(const HloInstruction* a, const HloInstruction* b) {
  auto colocator = GetInstructionColocatorHelper(a);
  return colocator ? (*colocator)->CanColocate(a, b) : false;
}

#undef REGISTER_INSTRUCTION_COLLOCATOR_HELPER
}  // namespace poplarplugin
}  // namespace xla
