/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/collectives/collective_pipelining_analyzer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/gpu/dynamic_slicing_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {

using SlicedUserPaths =
    absl::InlinedVector<absl::InlinedVector<HloInstruction*, 2>, 4>;

using SlicedOperandPath = absl::InlinedVector<HloInstruction*, 8>;

// Defines the frontend attribute when compiler determines whether a collective
// is trivially pipelineable.
constexpr char kTriviallyPipelineable[] = "trivially_pipelineable";

// This is a hardcoded threshold which we check to determine whether collectives
// can be pipelined without negative side effects. The idea is we will pipeline
// only small enough collectives and leave up to combiners the decision how much
// should we combine them in order to minimize the risk of involuntary remat.
constexpr int64_t kPipeliningSizeThreshold = 1 << 30;  // 1 Gibibyte.

class CollectivePipelineAnalyzerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CollectivePipelineAnalyzerVisitor(
      std::unique_ptr<CallGraph> call_graph, const HloModuleConfig& config,
      int64_t pointer_size)
      : call_graph_(std::move(call_graph)),
        config_(config),
        pointer_size_(pointer_size) {}

  absl::Status HandleAllReduce(HloInstruction* instr) override {
    if (!ShouldProcess(*instr, pointer_size_)) {
      return absl::OkStatus();
    }

    HloInstruction* hero = instr;
    // In XLA:GPU pipeline we run this pass before `AllReduce` is transformed
    // into `ReduceScatter`. To simplify the logic matching the op with the
    // subsequent DUS we check if op is the logical ReduceScatter. If so we will
    // re-adjust the hero instruction accordingly.
    if (std::optional<ReduceScatterSpec> spec = MatchReduceScatter(
            Cast<HloAllReduceInstruction>(instr), config_.num_partitions(),
            config_.replica_count(),
            /*allow_multiple_split_dims=*/false,
            /*allow_intervening_reshape=*/true);
        spec.has_value()) {
      hero = spec->dynamic_slice;
    }
    auto paths =
        GetSlicedUserPaths(*hero, *call_graph_, allowed_formatting_ops_,
                           /*check_alignment=*/false);
    if (IsEligibleSlicedUserPath(paths)) {
      MarkAsTriviallyPipelineable(instr);
    }
    return absl::OkStatus();
  }

  absl::Status HandleReduceScatter(HloInstruction* instr) override {
    if (!ShouldProcess(*instr, pointer_size_)) {
      return absl::OkStatus();
    }

    auto paths =
        GetSlicedUserPaths(*instr, *call_graph_, allowed_formatting_ops_,
                           /*check_alignment=*/false);
    if (IsEligibleSlicedUserPath(paths)) {
      MarkAsTriviallyPipelineable(instr);
    }
    return absl::OkStatus();
  }

  absl::Status HandleAllGather(HloInstruction* instr) override {
    if (!ShouldProcess(*instr, pointer_size_)) {
      return absl::OkStatus();
    }

    auto path =
        GetSlicedOperandPaths(*instr, *call_graph_, allowed_formatting_ops_,
                              /*check_alignment=*/false);
    if (IsEligibleSlicedOperandPath(path)) {
      MarkAsTriviallyPipelineable(instr);
    }
    return absl::OkStatus();
  }

 private:
  bool IsNoOp(const HloInstruction* instr) {
    return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kTuple,
                            HloOpcode::kGetTupleElement>(instr);
  }

  void MarkAsTriviallyPipelineable(HloInstruction* instr) {
    VLOG(1) << "Identified " << instr->name() << " as trivially pipeline-able.";
    instr->add_frontend_attribute(kTriviallyPipelineable, "");
  }

  // Makes sure there is a single path from DUS to ROOT and along this path
  // there are only noops.
  bool IsEligibleSlicedUserPath(const SlicedUserPaths& user_paths) {
    if (user_paths.size() != 1) {
      return false;
    }

    HloInstruction* dus = user_paths.front().back();
    if (HloPredicateIsNotOp<HloOpcode::kDynamicUpdateSlice>(dus)) {
      return false;
    }

    if (dus->user_count() != 1) {
      return false;
    }
    // We traverse the chain from DUS to root, making sure it's a single path
    // and all of the ops are effective noops.
    HloInstruction* instr;
    for (instr = dus->users().front(); instr->user_count() == 1;
         instr = instr->users().front()) {
      if (!IsNoOp(instr)) {
        return false;
      }
    }
    return instr->IsRoot();
  }

  // Makes sure there is a single path from DS to a computation parameter and
  // along this path there are only noops.
  bool IsEligibleSlicedOperandPath(const SlicedOperandPath& operand_path) {
    const HloInstruction* hero = operand_path.front();
    if (HloPredicateIsNotOp<HloOpcode::kDynamicSlice>(hero)) {
      return false;
    }
    // We start from the first operand DS operand, is it is the buffer we're
    // slicing and has the only chance to be preceded by heavy ops.
    const HloInstruction* it;
    for (it = hero->operand(0); it->operands().size() == 1;
         it = it->operand(0)) {
      if (!IsNoOp(it)) {
        return false;
      }
    }

    return it->opcode() == HloOpcode::kParameter;
  }

  // We allow processing of ops with shapes up to `kPipeliningSizeThreshold` to
  // avoid the involuntary remat, and we also make sure we do not pipeline ops
  // in such a way that we expand the while loop tuple size (as it leads to
  // increased compilation times).
  bool ShouldProcess(const HloInstruction& instr, int64_t pointer_size) {
    VLOG(1) << "Determining eligibility of processing: " << instr.ToString();
    switch (instr.opcode()) {
      case HloOpcode::kAllReduce:
      case HloOpcode::kReduceScatter:
        return instr.operands().size() == 1 &&
               ShapeUtil::ByteSizeOf(instr.shape(), pointer_size) <
                   kPipeliningSizeThreshold;
      case HloOpcode::kAllGather:
        return ShapeUtil::ByteSizeOf(instr.shape(), pointer_size) <
               kPipeliningSizeThreshold;
      default:
        return false;
    }
    return false;
  }

  std::unique_ptr<CallGraph> call_graph_;
  const HloModuleConfig& config_;
  const HloPredicate allowed_formatting_ops_ =
      HloPredicateIsOp<HloOpcode::kReshape, HloOpcode::kConvert,
                       HloOpcode::kTranspose, HloOpcode::kBitcast,
                       HloOpcode::kGetTupleElement, HloOpcode::kTuple>;
  int64_t pointer_size_;
};

}  // namespace

bool IsTriviallyPipelineable(const HloInstruction& instr) {
  return instr.frontend_attributes().map().contains(kTriviallyPipelineable);
}

absl::StatusOr<bool> CollectivePipeliningAnalyzer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const HloModuleConfig& config = module->config();

  CollectivePipelineAnalyzerVisitor visitor(CallGraph::Build(module), config,
                                            pointer_size_);
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace xla::gpu
