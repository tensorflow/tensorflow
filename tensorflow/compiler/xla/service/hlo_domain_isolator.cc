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

#include "tensorflow/compiler/xla/service/hlo_domain_isolator.h"

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

namespace {

StatusOr<bool> RunInternal(HloModule* module,
                           HloDomainIsolator::DomainCreator* creator) {
  int64 added_domains = 0;
  for (HloComputation* computation : module->computations()) {
    // Walk in post order and place all the required kDomain instructions.
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kDomain) {
        continue;
      }
      for (HloInstruction* operand : instruction->unique_operands()) {
        // When applying multiple domains, we could end up stacking more than
        // one in one edge, so here we want to build the effective
        // (kDomain-less) instruction->operand edge.
        HloInstruction* root = operand;
        while (root->opcode() == HloOpcode::kDomain) {
          root = root->mutable_operand(0);
        }
        // Check whether a kDomain is necessary between instruction and operand.
        HloInstruction* domain = (*creator)(instruction, root, operand);
        if (domain != nullptr) {
          VLOG(4) << "New domain: " << domain->ToString();
          // Call ReplaceUseWithDifferentShape even though the shapes are
          // expected to match to avoid an expensive shape check between the
          // original and the new instruction.
          TF_RETURN_IF_ERROR(
              operand->ReplaceUseWithDifferentShape(instruction, domain));
          ++added_domains;
        }
      }
    }
  }
  VLOG(3) << "Added " << added_domains << " kDomain instructions";
  return added_domains > 0;
}

}  // namespace

HloDomainIsolator::HloDomainIsolator(DomainCreatorFactory creator_factory)
    : creator_factory_(std::move(creator_factory)) {}

StatusOr<bool> HloDomainIsolator::Run(HloModule* module) {
  DomainCreator creator = creator_factory_();
  return RunInternal(module, &creator);
}

}  // namespace xla
