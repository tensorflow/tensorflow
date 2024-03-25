/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/tools/hlo_extractor.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/status.h"

namespace xla {
namespace {

// Visitor that build a new HLO module with an given root HLO instruction. Only
// HLOs that are reachable from the new root instruction are included in the new
// module.
//
// The constructor allows specifying a set of boundary HLOs to prune the HLO
// graph. HLOs at the boundary are replaced with parameters. Can be nullptr
// which means no boundary.
//
// This visitor keeps a map `old_computations_to_builders_` that maps the
// original computations to computation builders. When visiting a root
// instruction of a computation, we know that all the instructions in that
// computation have been visited and will not be visited again (since visitor is
// post-order DFS that tracks the visited instructions), so we build the
// computation and and put it in the `clone_context_`. When we visit the users
// of this computation (in the original HLO module), it would be replaced with
// the newly-built computation (in the extracted HLO module).
class ExtractionVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  explicit ExtractionVisitor(
      const HloInstruction* root_instruction,
      absl::flat_hash_set<const HloInstruction*>* boundary,
      ExtractSelector extract_selector,
      ReplaceTypeSelector replace_type_selector)
      : root_instruction_(root_instruction),
        old_module_(root_instruction->GetModule()),
        module_(std::make_unique<HloModule>(
            "extracted", config_,
            std::make_unique<CompilationEnvironments>(
                old_module_->comp_envs()))),
        clone_context_(module_.get()),
        boundary_(boundary),
        extract_selector_(extract_selector),
        replace_type_selector_(replace_type_selector) {
    // Initialize the computation builder for every computations.
    for (auto computation : old_module_->computations()) {
      old_computations_to_builders_.insert(
          {computation,
           std::make_unique<HloComputation::Builder>(computation->name())});
    }

    // Initialize the parameter counter for every computations.
    for (auto computation : old_module_->computations()) {
      parameter_numbers_[computation] = 0;
    }
  }

  Status HandleParameter(const HloInstruction* parameter) override {
    // Entry parameters need renumbering.
    return ReplaceWithParameter(parameter);
  }

  Status DefaultAction(const HloInstruction* hlo) override {
    // Replace the following two types of instructions with parameters/constants
    // (1) the instructions at the boundary with (2) the instructions that are
    // not selected by the hlo_selector.
    if ((boundary_ != nullptr && boundary_->contains(hlo) > 0) ||
        (extract_selector_ != nullptr && !extract_selector_(hlo))) {
      if (replace_type_selector_ != nullptr) {
        switch (replace_type_selector_(hlo)) {
          case ReplaceType::kReplaceConst:
            return ReplaceWithConstant(hlo);
          case ReplaceType::kReplaceParam:
            CHECK(hlo->parent() == root_instruction_->parent())
                << "Replacing instructions at non-entry computation with "
                   "parameters is not supported.";
            return ReplaceWithParameter(hlo);
          case ReplaceType::kReplaceZeroBroadcast:
            return ReplaceWithConstantBroadcast(
                hlo, ReplaceType::kReplaceZeroBroadcast);
          case ReplaceType::kReplaceRandomBroadcast:
            return ReplaceWithConstantBroadcast(
                hlo, ReplaceType::kReplaceRandomBroadcast);
          default:
            QCHECK(false) << "Unsupported replacement type";
        }
      }

      return ReplaceWithParameter(hlo);
    }

    // Clone the visiting hlo and add it to computation builder.
    std::vector<HloInstruction*> new_operands;
    for (auto operand : hlo->operands()) {
      new_operands.push_back(clone_context_.GetInstruction(operand));
    }
    auto instruction =
        hlo->CloneWithNewOperands(hlo->shape(), new_operands, &clone_context_);

    auto it = old_computations_to_builders_.find(hlo->parent());
    CHECK(it != old_computations_to_builders_.end());
    auto builder = it->second.get();
    builder->AddInstruction(std::move(instruction));

    // If the visiting `hlo` is the root instruction of a computation (except
    // for the root of the entry computation), we can build the new computation
    // now and put it in `clone_context_`. The entry computation would be built
    // in `FinishVisit()` when all the instructions are visited.
    if (hlo->IsRoot() && hlo != root_instruction_) {
      CHECK(clone_context_.FindComputation(hlo->parent()) == nullptr);
      auto new_computation = module_->AddEmbeddedComputation(builder->Build());
      clone_context_.MapComputation(hlo->parent(), new_computation);
    }

    return OkStatus();
  }

  Status FinishVisit(const HloInstruction* /*root*/) override {
    // Create the entry computation for the extracted module.
    auto new_entry_computation = module_->AddEntryComputation(
        old_computations_to_builders_.at(root_instruction_->parent())->Build());
    clone_context_.MapComputation(root_instruction_->parent(),
                                  new_entry_computation);

    // Rename HLOs so that their name matches the original. By default,
    // HLOs get new unique names when adding a new entry computation to
    // a module.
    for (auto computation : old_module_->MakeComputationPostOrder()) {
      for (auto old_instruction : computation->MakeInstructionPostOrder()) {
        if (auto new_instruction =
                clone_context_.FindInstruction(old_instruction)) {
          new_instruction->SetAndSanitizeName(old_instruction->name());
        }
      }
    }
    // For the extra created instructions (e.g., the ones created when replacing
    // with broadcasted zeros), we make sure they have unique names without
    // breaking the matches made at above code.
    for (HloInstruction* instruction : extra_created_instructions_) {
      module_->SetAndUniquifyInstrName(instruction, instruction->name());
    }

    return OkStatus();
  }

  HloModule* module() { return module_.get(); }

  std::unique_ptr<HloModule> ConsumeModule() { return std::move(module_); }

 private:
  // Replace the `hlo` with Constant of the same shape.
  Status ReplaceWithConstant(const HloInstruction* hlo) {
    absl::StatusOr<Literal> literal_status = MakeFakeLiteral(hlo->shape());
    TF_CHECK_OK(literal_status.status());
    auto new_const =
        HloInstruction::CreateConstant(std::move(literal_status.value()));
    clone_context_.MapInstruction(hlo, new_const.get());
    auto it = old_computations_to_builders_.find(hlo->parent());
    CHECK(it != old_computations_to_builders_.end());
    auto builder = it->second.get();
    builder->AddInstruction(std::move(new_const));
    return OkStatus();
  }

  // Replace the `hlo` with Parameter of the same shape.
  Status ReplaceWithParameter(const HloInstruction* hlo) {
    CHECK(parameter_numbers_.contains(hlo->parent()));
    auto new_parameter = HloInstruction::CreateParameter(
        parameter_numbers_.at(hlo->parent())++, hlo->shape(), hlo->name());
    clone_context_.MapInstruction(hlo, new_parameter.get());
    CHECK(old_computations_to_builders_.contains(hlo->parent()));
    auto builder = old_computations_to_builders_[hlo->parent()].get();
    builder->AddInstruction(std::move(new_parameter));
    return OkStatus();
  }

  // Helper to create constant instruction (that return a constant tensor) of
  // the given shape. If the shape is of tuple type, we recursively reuse/create
  // constant instruction for each of its sub-type. If it is not tuple type, we
  // just create a constant and broadcast it to the desired shape.
  // Currently the constant could be either a zero or a random number, depending
  // on `replace_type`.
  HloInstruction* ReplaceWithConstantBroadcastHelper(
      const Shape& shape, HloComputation::Builder* builder,
      ReplaceType replace_type) {
    if (shape.IsTuple()) {
      // If it is a tuple, recursively create a zero instruction.
      std::vector<HloInstruction*> tuple_operands;
      for (const auto& subshape : shape.tuple_shapes()) {
        tuple_operands.push_back(ReplaceWithConstantBroadcastHelper(
            subshape, builder, replace_type));
      }
      auto zero_tuple =
          builder->AddInstruction(HloInstruction::CreateTuple(tuple_operands));
      extra_created_instructions_.push_back(zero_tuple);
      return zero_tuple;
    } else {
      // If not a tuple, we need to create a zero constant of
      // `shape.element_type()`, and then broadcast it into the shape we want.

      // Create a constant of `shape.element_type()`. The constant could be
      // either a zero or a random number, depending on `replace_type`.
      Shape constant_shape = ShapeUtil::MakeShape(shape.element_type(), {});
      HloInstruction* constant_instruction;
      CHECK(replace_type == ReplaceType::kReplaceZeroBroadcast ||
            replace_type == ReplaceType::kReplaceRandomBroadcast);
      if (replace_type == ReplaceType::kReplaceZeroBroadcast) {
        constant_instruction =
            builder->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(constant_shape.element_type())));
      } else {
        absl::StatusOr<Literal> literal_status =
            MakeFakeLiteral(constant_shape);
        TF_CHECK_OK(literal_status.status());
        constant_instruction = builder->AddInstruction(
            HloInstruction::CreateConstant(std::move(literal_status.value())));
      }
      extra_created_instructions_.push_back(constant_instruction);

      // Broadcast `constant_instruction` to create an hlo of the desired
      // shape.
      auto broadcast_constant_instruction = builder->AddInstruction(
          HloInstruction::CreateBroadcast(shape, constant_instruction, {}));
      extra_created_instructions_.push_back(broadcast_constant_instruction);
      return broadcast_constant_instruction;
    }
  }

  // Replace with `hlo` with a broadcasted constant of the same shape. The
  // constant could be either a zero or a random number, depending on
  // `replace_type`.
  Status ReplaceWithConstantBroadcast(const HloInstruction* hlo,
                                      ReplaceType replace_type) {
    CHECK(replace_type == ReplaceType::kReplaceZeroBroadcast ||
          replace_type == ReplaceType::kReplaceRandomBroadcast);
    CHECK(old_computations_to_builders_.contains(hlo->parent()));
    auto builder = old_computations_to_builders_[hlo->parent()].get();
    HloInstruction* zero_broadcast =
        ReplaceWithConstantBroadcastHelper(hlo->shape(), builder, replace_type);
    clone_context_.MapInstruction(hlo, zero_broadcast);
    return OkStatus();
  }

  const HloInstruction* root_instruction_;
  HloModule* old_module_;
  HloModuleConfig config_;
  std::unique_ptr<HloModule> module_;
  HloCloneContext clone_context_;
  // Map from the old (i.e., original) computations to the builders (that build
  // the new computations in the extracted module).
  absl::flat_hash_map<const HloComputation*,
                      std::unique_ptr<HloComputation::Builder>>
      old_computations_to_builders_;
  // Keep track of the number of parameters of each computation, as the counter
  // is necessary to create a valid Parameter op.
  absl::flat_hash_map<const HloComputation*, int> parameter_numbers_;
  absl::flat_hash_set<const HloInstruction*>* boundary_;
  ExtractSelector extract_selector_;
  ReplaceTypeSelector replace_type_selector_;
  std::vector<HloInstruction*> extra_created_instructions_;
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

std::unique_ptr<HloModule> ExtractModule(
    const HloInstruction* instruction, int64_t height,
    ExtractSelector extract_selector, ReplaceTypeSelector replace_type_selector,
    bool cross_computation) {
  QCHECK(height == -1 || !cross_computation)
      << "Boundary cannnot be calculated across the computations.";

  absl::flat_hash_set<const HloInstruction*> boundary;
  if (height != -1) {
    ComputeBoundary(instruction, height, &boundary);
  }
  ExtractionVisitor visitor(instruction, &boundary, extract_selector,
                            replace_type_selector);

  TF_CHECK_OK(instruction->Accept(&visitor, /*call_finish_visit=*/true,
                                  /*ignore_control_predecessors=*/false,
                                  /*cross_computation=*/cross_computation));

  // The first pass may leave unused parameter instructions in the entry
  // computation. Do another extraction pass to remove unused parameters in the
  // entry computation. This is done because HloComputation does not allow
  // removing parameters after the computation has been built.
  ExtractionVisitor cleanup_visitor(
      visitor.module()->entry_computation()->root_instruction(),
      /*boundary=*/nullptr,
      /*extract_selector=*/nullptr,
      /*replace_type_selector=*/nullptr);

  TF_CHECK_OK(visitor.module()->entry_computation()->root_instruction()->Accept(
      &cleanup_visitor, /*call_finish_visit=*/true,
      /*ignore_control_predecessors=*/false,
      /*cross_computation=*/false));

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision=*/true);
  TF_CHECK_OK(verifier.Run(cleanup_visitor.module()).status());
  return cleanup_visitor.ConsumeModule();
}

}  // namespace xla
