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

#include "tensorflow/compiler/xla/tools/hlo_extractor.h"

#include <stdio.h>
#include <unistd.h>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace {

// Visitor that build a new HLO module with an entry computation and a root that
// is provided to the visit function. Only HLOs that are reachable from the new
// root instruction are included in the new module.
//
// The constructor allows specifying a set of boundary HLOs to prune the HLO
// graph. HLOs at the boundary are replaced with parameters. Can be nullptr
// which means no boundary, i.e. no HLOs are replaced with parameters.
class ExtractionVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  explicit ExtractionVisitor(
      const HloModule& old_module,
      absl::flat_hash_set<const HloInstruction*>* boundary)
      : old_module_(old_module),
        module_(absl::make_unique<HloModule>("extracted", config_)),
        clone_context_(module_.get()),
        builder_("entry_computation"),
        boundary_(boundary) {}

  Status HandleParameter(const HloInstruction* parameter) override {
    // Entry parameters need renumbering.
    auto new_parameter = HloInstruction::CreateParameter(
        parameter_number_++, parameter->shape(), parameter->name());
    clone_context_.MapInstruction(parameter, new_parameter.get());
    builder_.AddInstruction(std::move(new_parameter));
    return OkStatus();
  }

  Status DefaultAction(const HloInstruction* hlo) override {
    // Replace instructions at the boundary with parameters, but leave constants
    // untouched.
    if (boundary_ != nullptr && boundary_->count(hlo) > 0) {
      auto new_parameter = HloInstruction::CreateParameter(
          parameter_number_, hlo->shape(), hlo->name());
      parameter_number_++;
      clone_context_.MapInstruction(hlo, new_parameter.get());
      builder_.AddInstruction(std::move(new_parameter));
      return OkStatus();
    }
    std::vector<HloInstruction*> new_operands;
    for (auto operand : hlo->operands()) {
      new_operands.push_back(clone_context_.GetInstruction(operand));
    }
    auto instruction =
        hlo->CloneWithNewOperands(hlo->shape(), new_operands, &clone_context_);
    builder_.AddInstruction(std::move(instruction));
    return OkStatus();
  }

  Status FinishVisit(const HloInstruction* /*root*/) override {
    module_->AddEntryComputation(builder_.Build());
    // Rename HLOs so that their name matches the original. By default,
    // HLOs get new unique names when adding a new entry computation to
    // a module.
    for (auto computation : old_module_.MakeComputationPostOrder()) {
      for (auto old_instruction : computation->MakeInstructionPostOrder()) {
        if (auto new_instruction =
                clone_context_.FindInstruction(old_instruction)) {
          new_instruction->SetAndSanitizeName(old_instruction->name());
        }
      }
    }
    return OkStatus();
  }

  HloModule* module() { return module_.get(); }

  std::unique_ptr<HloModule> ConsumeModule() { return std::move(module_); }

 private:
  const HloModule& old_module_;
  HloModuleConfig config_;
  std::unique_ptr<HloModule> module_;
  HloCloneContext clone_context_;
  HloComputation::Builder builder_;
  absl::flat_hash_set<const HloInstruction*>* boundary_;
  int64_t parameter_number_ = 0;
};

void ComputeBoundary(const HloInstruction* root, int64_t limit,
                     absl::flat_hash_set<const HloInstruction*>* boundary) {
  std::deque<const HloInstruction*> worklist;
  absl::flat_hash_map<const HloInstruction*, int64_t> visited;
  worklist.push_back(root);
  visited.emplace(root, 0);
  while (!worklist.empty()) {
    auto hlo = worklist.front();
    worklist.pop_front();
    int64_t hops = visited[hlo];
    if (hops > limit) {
      boundary->insert(hlo);
      continue;
    }
    for (const HloInstruction* operand : hlo->operands()) {
      if (visited.count(operand)) {
        continue;
      }
      worklist.push_back(operand);
      visited.emplace(operand, hops + 1);
    }
  }
}

}  // namespace

std::unique_ptr<HloModule> ExtractModule(HloInstruction* instruction,
                                         int64_t height) {
  absl::flat_hash_set<const HloInstruction*> boundary;
  if (height != -1) {
    ComputeBoundary(instruction, height, &boundary);
  }
  ExtractionVisitor visitor(*instruction->GetModule(), &boundary);
  CHECK(instruction->Accept(&visitor).ok());

  // The first pass may leave unused parameter instructions. Do another
  // extraction pass to remove unused parameters. This is done because
  // HloComputation does not allow removing parameters after the computation has
  // been built.
  ExtractionVisitor cleanup_visitor(*visitor.module(), /*boundary=*/nullptr);
  TF_CHECK_OK(visitor.module()->entry_computation()->root_instruction()->Accept(
      &cleanup_visitor));

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision=*/true);
  TF_CHECK_OK(verifier.Run(cleanup_visitor.module()).status());
  return cleanup_visitor.ConsumeModule();
}

}  // namespace xla
