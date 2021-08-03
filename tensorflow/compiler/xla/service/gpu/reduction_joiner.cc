/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/reduction_joiner.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

namespace {

namespace m = match;

// Returns whether the computation is addition.
bool IsAddition(HloComputation* computation) {
  return Match(computation->root_instruction(),
               m::AddAnyOrder(m::Parameter(0), m::Parameter(1)));
}

class ReductionSumVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleAdd(HloInstruction* add) override {
    HloInstruction *p1, *p2;
    HloInstruction *r1, *r2;

    auto reduction_matcher = [](HloInstruction** reduction,
                                HloInstruction** input) {
      return m::Op(reduction)
          .WithOpcode(HloOpcode::kReduce)
          .WithOperand(0, m::Op(input))
          .WithOperand(1, m::ConstantScalar(0));
    };

    if (Match(add, m::Add(reduction_matcher(&r1, &p1),
                          reduction_matcher(&r2, &p2)))) {
      if (!(IsAddition(r1->called_computations()[0]) &&
            IsAddition(r2->called_computations()[0]) &&
            r1->dimensions() == r2->dimensions() &&
            r1->dimensions().size() == 1)) {
        return Status::OK();
      }

      int reduced_dim = r1->dimensions(0);
      StatusOr<HloInstruction*> new_concat =
          CreateMatchedConcatenation(p1, p2, reduced_dim);
      TF_RETURN_IF_ERROR(new_concat.status());

      VLOG(1) << "Inside computation: " << add->parent()->name();
      VLOG(1) << "Generated concatenation: " << (*new_concat)->ToString();

      StatusOr<Shape> reduce_shape = ShapeInference::InferReduceShape(
          {&(*new_concat)->shape(), &r1->operand(1)->shape()}, {reduced_dim},
          r1->called_computations()[0]->ComputeProgramShape());
      TF_RETURN_IF_ERROR(reduce_shape.status());

      std::unique_ptr<HloInstruction> new_reduction =
          HloInstruction::CreateReduce(*reduce_shape, *new_concat,
                                       r1->mutable_operand(1), {reduced_dim},
                                       r1->called_computations()[0]);
      VLOG(1) << "Replacing: " << add->ToString()
              << " with: " << new_reduction->ToString();

      TF_RETURN_IF_ERROR(
          ReplaceWithNewInstruction(add, std::move(new_reduction)));
    }

    return Status::OK();
  }

 private:
  StatusOr<HloInstruction*> CreateMatchedConcatenation(HloInstruction* lhs,
                                                       HloInstruction* rhs,
                                                       int reduced_dim) {
    if (absl::optional<absl::Span<HloInstruction* const>> lhs_concat_operands =
            MatchConcatenation(lhs, reduced_dim)) {
      return ExtendMatchedConcatenation(rhs, *lhs_concat_operands, reduced_dim);
    } else if (absl::optional<absl::Span<HloInstruction* const>>
                   rhs_concat_operands = MatchConcatenation(rhs, reduced_dim)) {
      return ExtendMatchedConcatenation(lhs, *rhs_concat_operands, reduced_dim);
    } else {
      return MakeConcatHlo({lhs, rhs}, reduced_dim);
    }
  }

  StatusOr<HloInstruction*> ExtendMatchedConcatenation(
      HloInstruction* p, absl::Span<HloInstruction* const> matched,
      int reduced_dim) {
    std::vector<HloInstruction*> c;
    c.push_back(p);
    c.insert(c.end(), matched.begin(), matched.end());
    return MakeConcatHlo(c, reduced_dim);
  }

  absl::optional<absl::Span<HloInstruction* const>> MatchConcatenation(
      HloInstruction* i, int dimension) {
    if (i->opcode() == HloOpcode::kConcatenate &&
        i->concatenate_dimension() == dimension) {
      return i->operands();
    }
    return absl::nullopt;
  }
};

// Verify that the reductions we are creating aren't blocking fusion and aren't
// being split.
class ConcatReductionsSplitterVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleReduce(HloInstruction* outer_reduce) override {
    // If we have a concatenation over reductions which is too large, let's try
    // to split it back.
    HloInstruction* concat = outer_reduce->mutable_operand(0);
    HloInstruction* init = outer_reduce->mutable_operand(1);
    HloComputation* reduction_op = outer_reduce->called_computations()[0];
    if (concat->opcode() != HloOpcode::kConcatenate ||
        !IsAddition(reduction_op) || outer_reduce->dimensions().size() != 1 ||
        !Match(init, m::ConstantScalar(0))) {
      return Status::OK();
    }
    int reduced_dim = outer_reduce->dimensions()[0];

    if (concat->operand_count() > kArgLimit) {
      VLOG(1) << "Matched a concat to split: " << concat->ToString()
              << " from module: " << outer_reduce->parent()->parent()->name()
              << " computation: " << outer_reduce->parent()->name();
      VLOG(1) << "Reduction over concat is: " << outer_reduce->ToString();

      // TODO(cheshire): Avoid having to create a vector of mutable operands.
      std::vector<HloInstruction*> operands;
      operands.reserve(concat->operand_count());
      for (int i = 0; i < concat->operand_count(); i++) {
        operands.push_back(concat->mutable_operand(i));
      }

      HloParameterGroups groups = GroupParameters(operands);

      std::vector<HloInstruction*> new_reductions;
      for (const std::vector<HloInstruction*>& group : groups) {
        TF_ASSIGN_OR_RETURN(HloInstruction * new_concat,
                            MakeConcatHlo(group, reduced_dim));
        new_reductions.push_back(outer_reduce->parent()->AddInstruction(
            HloInstruction::CreateReduce(outer_reduce->shape(), new_concat,
                                         init, {reduced_dim}, reduction_op)));
        VLOG(1) << "Generated one of reductions: "
                << new_reductions.back()->ToString();
      }
      HloInstruction* variadic_sum = CreateVariadicSum(new_reductions);
      VLOG(1) << "Generated sum over reductions: " << variadic_sum->ToString();
      TF_RETURN_IF_ERROR(outer_reduce->parent()->ReplaceInstruction(
          outer_reduce, variadic_sum));
      changed_ = true;
    }

    return Status::OK();
  }

 private:
  using HloParameterGroups = std::vector<std::vector<HloInstruction*>>;

  HloParameterGroups GroupParameters(
      const std::vector<HloInstruction*>& params) {
    HloParameterGroups out;
    std::vector<HloInstruction*>* cursor = nullptr;
    for (HloInstruction* p : params) {
      if (cursor == nullptr || cursor->size() >= kArgLimit) {
        out.emplace_back();
        cursor = &out.back();
      }
      cursor->push_back(p);
    }
    return out;
  }

  HloInstruction* CreateVariadicSum(absl::Span<HloInstruction* const> params) {
    CHECK(!params.empty());
    HloInstruction* cursor = params[0];
    for (HloInstruction* p : params.subspan(1)) {
      cursor = p->parent()->AddInstruction(
          HloInstruction::CreateBinary(p->shape(), HloOpcode::kAdd, p, cursor));
    }
    return cursor;
  }

  const int kArgLimit = kMaxOperandsAndOutputsPerFusion - 10;
};

}  // namespace

StatusOr<bool> ReductionJoiner::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(bool changed, ReductionSumVisitor().RunOnModule(module));
  if (changed) {
    TF_RETURN_IF_ERROR(
        ConcatReductionsSplitterVisitor().RunOnModule(module).status());
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
