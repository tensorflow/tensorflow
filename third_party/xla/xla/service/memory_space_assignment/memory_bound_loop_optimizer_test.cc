/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/memory_bound_loop_optimizer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/service/memory_space_assignment/prefetch_interval_picker.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace memory_space_assignment {
namespace {

constexpr int64_t kPointerSize = 8;

int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

int64_t SizeFunction(const BufferValue& value) {
  return ShapeSize(value.shape());
}

int64_t ReservedScopedMemoryFn(
    const HloInstruction* instruction,
    const absl::flat_hash_set<std::pair<int, ShapeIndex>>&
        operands_in_alternate_memory,
    const absl::flat_hash_set<ShapeIndex>& outputs_in_alternate_memory) {
  return 0;
}

class MemoryBoundLoopOptimizerTest : public HloTestBase {
 public:
  MemoryBoundLoopOptimizerTest() = default;

 protected:
  const int64_t kAlternateMemorySpace = 1;
  const int64_t kDefaultMemorySpace = 0;

  absl::Status Initialize(const HloModule* module,
                          uint64_t alternate_memory_size = 256) {
    HloCostAnalysis::Options options;
    MemoryBoundLoopOptimizerOptions optimizer_options;
    optimizer_options.set_enabled(true);
    optimizer_options.set_desired_copy_ratio(0.7);
    optimizer_options.set_allow_unsatisfied_fully_pipelined_prefetch(false);
    optimizer_options.set_min_num_iterations(3.0);
    options_.memory_bound_loop_optimizer_options = optimizer_options;
    cost_analysis_options_.alternate_mem_bandwidth_bytes_per_second = 128;
    cost_analysis_options_.async_copy_bandwidth_bytes_per_second = 32;
    cost_analysis_options_.pipeline_overhead_window_size_mib = 1;
    options.shape_size = ShapeSize;
    options.set_flops_per_second(16);
    options.set_bytes_per_second(32);
    options.set_transcendentals_per_second(16);
    hlo_cost_analysis_ = std::make_unique<HloCostAnalysis>(options);
    TF_RETURN_IF_ERROR(
        module->entry_computation()->Accept(hlo_cost_analysis_.get()));
    hlo_cost_analysis_costs_ =
        std::make_unique<HloCostAnalysisCosts>(*hlo_cost_analysis_);
    TF_ASSIGN_OR_RETURN(cost_analysis_,
                        CostAnalysis::Create(*hlo_cost_analysis_costs_,
                                             cost_analysis_options_, *module));
    TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module));
    TF_ASSIGN_OR_RETURN(live_range_,
                        HloLiveRange::Run(module->schedule(), *alias_analysis_,
                                          module->entry_computation()));
    return absl::OkStatus();
  }

  absl::StatusOr<MemoryBoundLoopOptimizer*> CreateOptimizer(
      int loop_start, int loop_end, const HloModule* module,
      uint64_t alternate_memory_size = 256,
      const ReservedScopedMemoryFunction& reserved_scoped_memory_fn =
          ReservedScopedMemoryFn) {
    TF_RETURN_IF_ERROR(Initialize(module, alternate_memory_size));
    MemoryBoundLoopOptimizerOptions optimizer_options;
    optimizer_options.set_enabled(true);
    optimizer_options.set_desired_copy_ratio(0.7);
    optimizer_options.set_allow_unsatisfied_fully_pipelined_prefetch(false);
    TF_ASSIGN_OR_RETURN(
        optimizer_,
        MemoryBoundLoopOptimizer::Create(
            loop_start, loop_end, alternate_memory_size, optimizer_options,
            *live_range_, *alias_analysis_, *cost_analysis_, SizeFunction,
            reserved_scoped_memory_fn));
    return optimizer_.get();
  }

  absl::StatusOr<std::unique_ptr<HloModule>> ParseAndCreateOptimizer(
      absl::string_view hlo_loop_str, uint64_t alternate_memory_size,
      int& loop_start_idx, MemoryBoundLoopOptimizer** optimizer,
      const ReservedScopedMemoryFunction& reserved_scoped_memory_fn =
          ReservedScopedMemoryFn) {
    int loop_end_idx;
    TF_ASSIGN_OR_RETURN(
        std::string module_str,
        ParseAndCreateModuleString(hlo_loop_str, loop_start_idx, loop_end_idx));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        ParseAndReturnVerifiedModule(module_str));
    TF_ASSIGN_OR_RETURN(
        *optimizer,
        CreateOptimizer(loop_start_idx, loop_end_idx, module.get(),
                        alternate_memory_size, reserved_scoped_memory_fn));
    return std::move(module);
  }

  // Parse a loop string description like the following:
  //  $op0 = f32[1,4] add(f32[1,4] $param0, f32[1,4] $prev_op4)
  //  $op1 = f32[8,4] add(f32[8,4] $param1, f32[8,4] $prev_op3)
  //  $op2 = f32[1,4] add(f32[1,4] $param2, f32[1,4] $op0)
  //  $op3 = f32[8,4] add(f32[8,4] $param3, f32[8,4] $op1)
  //  $op4 = f32[1,4] add(f32[1,4] $param4, f32[1,4] $op2)
  absl::StatusOr<std::string> ParseAndCreateModuleString(
      absl::string_view hlo_loop_str, int& loop_start_idx, int& loop_end_idx) {
    // Parse op name and types first.
    RE2 op_re("\\$op([0-9]+) += +(\\S+).*");
    std::vector<absl::string_view> ops;
    std::vector<absl::string_view> op_types;
    int begin_pos = 0;
    absl::string_view submatch[3];
    while (op_re.Match(hlo_loop_str, begin_pos, hlo_loop_str.size(),
                       RE2::UNANCHORED, submatch, /*nsubmatch=*/3)) {
      for (int i = 0; i < 3; ++i) {
        if (submatch[i].data() == nullptr) {
          VLOG(4) << "Submatch[" << i << "] = nullptr";
        } else {
          VLOG(4) << "Submatch[" << i << "] = " << submatch[i]
                  << " (idx: " << (submatch[i].data() - hlo_loop_str.data())
                  << ")";
        }
      }
      int op_num;
      if (!absl::SimpleAtoi(submatch[1], &op_num)) {
        return InvalidArgument("Op name expects to contain a number, found %s.",
                               submatch[1]);
      }
      if (op_num != ops.size()) {
        return InvalidArgument("Op number expected to be %d found %d.",
                               op_types.size(), op_num);
      }
      ops.push_back(submatch[0]);
      op_types.push_back(submatch[2]);
      begin_pos = submatch[0].data() - hlo_loop_str.data() + submatch[0].size();
    }

    RE2 param_re("([[:alnum:]]+\\[\\S*\\]) +\\$param([0-9]+)");
    std::vector<absl::string_view> param_types;
    begin_pos = 0;
    while (param_re.Match(hlo_loop_str, begin_pos, hlo_loop_str.size(),
                          RE2::UNANCHORED, submatch, /*nsubmatch=*/3)) {
      for (int i = 0; i < 3; ++i) {
        if (submatch[i].data() == nullptr) {
          VLOG(4) << "Submatch[" << i << "] = nullptr";
        } else {
          VLOG(4) << "Submatch[" << i << "] = " << submatch[i]
                  << " (idx: " << (submatch[i].data() - hlo_loop_str.data())
                  << ")";
        }
      }
      int param_num;
      if (!absl::SimpleAtoi(submatch[2], &param_num)) {
        return InvalidArgument(
            "Param name expects to contain a number, found %s.", submatch[2]);
      }
      while (param_num >= param_types.size()) {
        param_types.push_back({});
      }
      param_types[param_num] = submatch[1];

      begin_pos = submatch[0].data() - hlo_loop_str.data() + submatch[0].size();
    }

    RE2 root_re("ROOT \\$root += +tuple\\((.*)\\)");
    absl::string_view root_values;
    if (root_re.Match(hlo_loop_str, 0, hlo_loop_str.size(), RE2::UNANCHORED,
                      submatch, /*nsubmatch=*/2)) {
      for (int i = 0; i < 2; ++i) {
        if (submatch[i].data() == nullptr) {
          VLOG(4) << "Submatch[" << i << "] = nullptr";
        } else {
          VLOG(4) << "Submatch[" << i << "] = " << submatch[i]
                  << " (idx: " << (submatch[i].data() - hlo_loop_str.data())
                  << ")";
        }
      }
      root_values = submatch[1];
    }

    for (absl::string_view op_type : op_types) {
      VLOG(4) << "op_type: " << op_type;
    }
    for (absl::string_view param_type : param_types) {
      VLOG(4) << "param_type: " << param_type;
    }

    std::string hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY Entry {
)";
    int total_instructions = 0;
    for (absl::string_view param_prefix : {"prev_", "", "next_"}) {
      for (int i = 0; i < param_types.size(); ++i) {
        int parameter_number = total_instructions;
        absl::StrAppend(&hlo_string, "  ", param_prefix, "param", i, " = ",
                        param_types[i], " parameter(", parameter_number,
                        ")  // ", total_instructions++, "\n");
      }
    }

    for (int i = 0; i < op_types.size(); ++i) {
      int parameter_number = total_instructions;
      absl::StrAppend(&hlo_string, "  ", "prev_prev_op", i, " = ", op_types[i],
                      " parameter(", parameter_number, ")  // ",
                      total_instructions++, "\n");
    }

    std::string new_root_values;
    auto print_ops =
        [&](const std::vector<std::pair<const absl::string_view, std::string>>&
                replacements) {
          for (int i = 0; i < ops.size(); ++i) {
            absl::StrAppend(&hlo_string, "  ",
                            absl::StrReplaceAll(ops[i], replacements), "  // ",
                            total_instructions++, "\n");
          }
          if (!root_values.empty()) {
            absl::StrAppend(&new_root_values,
                            new_root_values.empty() ? "" : ", ",
                            absl::StrReplaceAll(root_values, replacements));
          }
        };

    std::vector<std::pair<const absl::string_view, std::string>>
        prev_replacements;
    prev_replacements.push_back({"$prev_op", "prev_prev_op"});
    prev_replacements.push_back({"$op", "prev_op"});
    prev_replacements.push_back({"$param", "prev_param"});
    absl::StrAppend(&hlo_string, "  // Prev iteration body:\n");
    print_ops(prev_replacements);

    loop_start_idx = total_instructions;
    std::vector<std::pair<const absl::string_view, std::string>> replacements;
    replacements.push_back({"$", ""});
    absl::StrAppend(&hlo_string, "  // Loop body:\n");
    print_ops(replacements);
    loop_end_idx = total_instructions;

    std::vector<std::pair<const absl::string_view, std::string>>
        next_replacements;
    next_replacements.push_back({"$prev_op", "op"});
    next_replacements.push_back({"$op", "next_op"});
    next_replacements.push_back({"$param", "next_param"});
    absl::StrAppend(&hlo_string, "  // Next iteration body:\n");
    print_ops(next_replacements);

    absl::StrAppend(&hlo_string, "  ROOT root = tuple(", new_root_values,
                    ")\n");
    absl::StrAppend(&hlo_string, "}");

    VLOG(1) << hlo_string;
    return hlo_string;
  }

  absl::StatusOr<std::unique_ptr<PresetAssignments>> RunMsa(
      HloModule* module, uint64_t alternate_memory_size = 256) {
    options_.max_size_in_bytes = alternate_memory_size;
    options_.alignment_in_bytes = 8;
    options_.verify = true;

    options_.alternate_memory_space = kAlternateMemorySpace;

    if (!cost_analysis_) {
      TF_RETURN_IF_ERROR(Initialize(module, alternate_memory_size));
    }
    CostAnalysis::Cache cache;
    MemoryBoundednessBufferIntervalComparator comparator(*cost_analysis_,
                                                         &cache);
    options_.buffer_interval_comparator = &comparator;
    CostAnalysisPrefetchIntervalPicker prefetch_interval_picker(
        CostAnalysisPrefetchIntervalPicker(
            *cost_analysis_, /*min_overlap_to_async_copy_ratio=*/0.8,
            /*preferred_overlap_to_async_copy_ratio=*/1.5,
            /*max_overlap_to_mem_size_async_copy_ratio=*/10.0,
            /*mem_size_bytes=*/alternate_memory_size));
    options_.prefetch_interval_picker = &prefetch_interval_picker;

    auto size_fn = [](const BufferValue& buffer) {
      return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
    };
    options_.size_fn = size_fn;

    auto is_allowed_in_alternate_mem = [](const HloValue& value) {
      // Check if the value belongs to the entry computation.
      HloInstruction* instruction = value.instruction();
      HloComputation* computation = instruction->parent();
      bool in_entry_computation =
          (computation == computation->parent()->entry_computation());
      if (in_entry_computation &&
          instruction->opcode() == HloOpcode::kParameter) {
        return false;
      }
      return true;
    };
    options_.is_allowed_in_alternate_mem_fn = is_allowed_in_alternate_mem;
    options_.max_outstanding_prefetches = -1;
    options_.max_outstanding_evictions = -1;
    options_.allocate_across_sequential_calls = true;
    options_.cost_analysis = cost_analysis_.get();

    std::unique_ptr<PresetAssignments> preset_assignments =
        MemorySpaceAssignment::Run(module, *live_range_, *alias_analysis_,
                                   options_)
            .value();
    return preset_assignments;
  }

  absl::Status VerifyMsaEquivalence(
      HloModule* module, bool expect_unsupported_allocations = false) {
    // Create a map indexed by instruction number and operand number.
    absl::flat_hash_map<std::pair<int, int>, const Allocation*> allocation_map;
    for (const MemoryBoundLoopOptimizer::LoopValue& value :
         optimizer_->loop_values()) {
      // Skip verification for unsupported allocations as they will go through
      // the usual MSA algorithm and may actually get an alternate memory
      // allocation.
      if (!value.IsAllocationTypeSupported()) {
        continue;
      }
      for (const auto& allocation : value.allocations) {
        for (const HloUse& use : allocation->uses()) {
          absl::string_view inst_name = use.instruction->name();
          TF_RET_CHECK(absl::StartsWith(inst_name, "op"));
          int inst_number;
          TF_RET_CHECK(absl::SimpleAtoi(inst_name.substr(2), &inst_number));
          allocation_map[{inst_number, use.operand_number}] = allocation.get();
        }
      }
    }

    auto get_inst_prefix_in_iter = [](int iteration) {
      switch (iteration) {
        case 0:
          return "prev_";
        case 1:
          return "";
        case 2:
          return "next_";
        default:
          LOG(FATAL) << "Invalid iteration " << iteration;
          return "INVALID";
      }
    };

    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                        HloAliasAnalysis::Run(module));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloLiveRange> live_range,
                        HloLiveRange::Run(module->schedule(), *alias_analysis,
                                          module->entry_computation()));
    const auto& flattened_instructions =
        live_range->flattened_instruction_sequence().instructions();
    for (int iteration = 1; iteration < 3; ++iteration) {
      for (int inst_number = 0; inst_number < optimizer_->loop_size();
           ++inst_number) {
        HloInstruction* inst = FindInstruction(
            module, absl::StrCat(get_inst_prefix_in_iter(iteration), "op",
                                 inst_number));
        for (int operand_number = 0; operand_number < 2; ++operand_number) {
          const HloInstruction* operand = inst->operand(operand_number);
          LOG(INFO) << inst->name() << ", operand " << operand_number;
          if (!allocation_map.contains({inst_number, operand_number})) {
            TF_RET_CHECK(expect_unsupported_allocations);
            continue;
          }
          const Allocation* allocation =
              allocation_map.at({inst_number, operand_number});
          if (!allocation->is_copy_allocation()) {
            // We don't expect a prefetch here.
            EXPECT_NE(operand->opcode(), HloOpcode::kCopyDone);
            int expected_memory_space =
                allocation->memory_space() == MemorySpace::kDefault
                    ? kDefaultMemorySpace
                    : kAlternateMemorySpace;
            EXPECT_EQ(operand->shape().layout().memory_space(),
                      expected_memory_space);
          } else {
            EXPECT_EQ(allocation->memory_space(), MemorySpace::kAlternate);
            TF_RET_CHECK(operand->opcode() == HloOpcode::kCopyDone);
            const CopyAllocation* copy_allocation =
                static_cast<const CopyAllocation*>(allocation);
            if (copy_allocation->copy_done_schedule_before() != inst_number) {
              // The only case where the copy done schedule before is not the
              // same as this use would be that this use is not the first use of
              // the copy allocation.
              EXPECT_NE(allocation->uses().front(),
                        (HloUse{inst, operand_number}));
              continue;
            }
            int expected_copy_start_iteration = iteration;
            if (copy_allocation->copy_start_schedule_after() ==
                    optimizer_->loop_size() &&
                copy_allocation->copy_done_schedule_before() == 0) {
              expected_copy_start_iteration -= 2;
            } else if (copy_allocation->copy_start_schedule_after() + 1 >=
                       copy_allocation->copy_done_schedule_before()) {
              expected_copy_start_iteration -= 1;
            }

            if (expected_copy_start_iteration >= 0) {
              const HloInstruction* expected_copy_start_schedule_after =
                  FindInstruction(
                      module,
                      absl::StrCat(
                          get_inst_prefix_in_iter(
                              expected_copy_start_iteration),
                          "op", copy_allocation->copy_start_schedule_after()));
              LOG(INFO) << "Expected copy start schedule after: "
                        << expected_copy_start_schedule_after->name();
              const HloInstruction* copy_start = operand->operand(0);
              TF_RET_CHECK(copy_start->opcode() == HloOpcode::kCopyStart);
              // Find the instruction before this copy start that is not an
              // async copy or gte or parameter.
              int copy_start_idx =
                  live_range->instruction_schedule().at(copy_start);
              const HloInstruction* copy_start_schedule_after = nullptr;
              for (int i = copy_start_idx - 1; i >= 0; --i) {
                HloOpcode opcode = flattened_instructions.at(i)->opcode();
                if (opcode != HloOpcode::kCopyStart &&
                    opcode != HloOpcode::kCopyDone &&
                    opcode != HloOpcode::kGetTupleElement &&
                    opcode != HloOpcode::kParameter) {
                  copy_start_schedule_after = flattened_instructions.at(i);
                  break;
                }
              }
              TF_RET_CHECK(copy_start_schedule_after != nullptr);
              EXPECT_EQ(copy_start_schedule_after,
                        expected_copy_start_schedule_after);
            }
          }
        }
      }
    }
    return absl::OkStatus();
  }

 private:
  Options options_;
  CostAnalysisOptions cost_analysis_options_;
  std::unique_ptr<HloCostAnalysis> hlo_cost_analysis_;
  std::unique_ptr<HloCostAnalysisCosts> hlo_cost_analysis_costs_;
  std::unique_ptr<CostAnalysis> cost_analysis_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  std::unique_ptr<HloLiveRange> live_range_;
  std::unique_ptr<MemoryBoundLoopOptimizer> optimizer_;
};

TEST_F(MemoryBoundLoopOptimizerTest, SimplePrefetch) {
  absl::string_view hlo_loop_str = R"(
    $op0 = f32[1,4] add(f32[1,4] $prev_op3, f32[1,4] $prev_op4)
    $op1 = f32[1,4] add(f32[1,4] $prev_op4, f32[1,4] $op0)
    $op2 = f32[1,4] add(f32[1,4] $op0, f32[1,4] $op1)
    $op3 = f32[1,4] add(f32[1,4] $op1, f32[1,4] $op2)
    $op4 = f32[1,4] add(f32[1,4] $param0, f32[1,4] $op3)
    ROOT $root = tuple($op4, $param0)
  )";
  int loop_start_idx;
  MemoryBoundLoopOptimizer* optimizer;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndCreateOptimizer(hlo_loop_str,
                                                  /*alternate_memory_size=*/128,
                                                  loop_start_idx, &optimizer));

  optimizer->Optimize();
  absl::flat_hash_set<HloUse> seen_uses;
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    LOG(INFO) << loop_value.ToString();
    if (loop_value.hlo_values.front()
            ->defining_position()
            .instruction->name() == "param0") {
      EXPECT_TRUE(loop_value.allocations.back()->is_copy_allocation());
    }
    for (const auto& allocation : loop_value.allocations) {
      for (const HloUse& use : allocation->uses()) {
        EXPECT_FALSE(seen_uses.contains(use)) << use.ToString();
        seen_uses.insert(use);
      }
    }
  }

  // Ensure all of the uses in the loop have an associated use.
  for (absl::string_view inst_name : {"op0", "op1", "op2", "op3", "op4"}) {
    HloInstruction* inst =
        module->entry_computation()->GetInstructionWithName(inst_name);
    EXPECT_TRUE(seen_uses.contains(HloUse{inst, 0})) << inst_name;
    EXPECT_TRUE(seen_uses.contains(HloUse{inst, 1})) << inst_name;
  }
}

// Specify a ReservedScopedMemoryFunction to the loop optimizer that causes each
// HLO to reserve the entire alternate memory. If the loop optimizer is
// correctly accounting for reserved scoped memory, it should not put any
// allocations in alternate memory, which we test.
TEST_F(MemoryBoundLoopOptimizerTest, ReservedScopedMemory) {
  absl::string_view hlo_loop_str = R"(
    $op0 = f32[1,4] add(f32[1,4] $prev_op3, f32[1,4] $prev_op4)
    $op1 = f32[1,4] add(f32[1,4] $prev_op4, f32[1,4] $op0)
    $op2 = f32[1,4] add(f32[1,4] $op0, f32[1,4] $op1)
    $op3 = f32[1,4] add(f32[1,4] $op1, f32[1,4] $op2)
    $op4 = f32[1,4] add(f32[1,4] $param0, f32[1,4] $op3)
    ROOT $root = tuple($op4, $param0)
  )";
  int loop_start_idx;
  MemoryBoundLoopOptimizer* optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndCreateOptimizer(
          hlo_loop_str,
          /*alternate_memory_size=*/128, loop_start_idx, &optimizer,
          [](const HloInstruction*,
             const absl::flat_hash_set<std::pair<int, ShapeIndex>>&,
             const absl::flat_hash_set<ShapeIndex>&) { return 128; }));

  optimizer->Optimize();
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    LOG(INFO) << "Loop value: " << loop_value.ToString();
    for (const auto& allocation : loop_value.allocations) {
      ASSERT_NE(static_cast<int64_t>(allocation->memory_space()),
                kAlternateMemorySpace);
    }
  }
}

// Check that a spurious GetTupleElement instruction in a later iteration of a
// loop does not cause MSA to CHECK fail, when identifying loops. Prior to the
// change instroduced with this test, IdentifyAndOptimizeMemoryBoundLoops()
// would recognize 4 iterations to the loop thinking that gte is a repeat of
// op2. Doing so triggers the CHECKs introduced by the change that added this
// test to fail. So, the point of this test is to verfiy that we do not check
// fail.
TEST_F(MemoryBoundLoopOptimizerTest, GetTupleElement) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[1,4] parameter(0)
    p1 = f32[1,4] parameter(1)
    p2 = f32[1,4] parameter(2)
    p3 = f32[1,4] parameter(3)
    p4 = f32[1,4] parameter(4)
    p5 = f32[1,4] parameter(5)
    p6 = f32[1,4] parameter(6)
    tupleparam = (f32[1,4], f32[1,4]) parameter(7)

    // Iteration 0
    op1 = tanh(p0)
    op2 = tanh(p1)
    op3 = tanh(op2)
    op4 = add(op1, op3)

    // Iteration 1
    op5 = tanh(p2)
    op6 = tanh(p3)
    op7 = tanh(op6)
    op8 = add(op5, op7)

    // Iteration 2
    op9 = tanh(p4)
    op10 = tanh(p5)
    op11 = tanh(op10)
    op12 = add(op9, op11)

    // Not an iteration
    op13 = tanh(p6)
    gte = get-tuple-element(tupleparam), index=1
    op14 = tanh(gte)
    op15 = tanh(op14)
    op16 = add(op13, op15)

    ROOT root = tuple(tupleparam, op4, op8, op12, op16)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  TF_ASSERT_OK_AND_ASSIGN(auto preset_assignments, RunMsa(module.get()));
}

TEST_F(MemoryBoundLoopOptimizerTest, NoAlternateMem) {
  absl::string_view hlo_loop_str = R"(
    $op0 = f32[1,4] add(f32[1,4] $prev_op3, f32[1,4] $prev_op4)
    $op1 = f32[1,4] add(f32[1,4] $prev_op4, f32[1,4] $op0)
    $op2 = f32[1,4] add(f32[1,4] $op0, f32[1,4] $op1)
    $op3 = f32[1,4] add(f32[1,4] $op1, f32[1,4] $op2)
    $op4 = f32[1,4] add(f32[1,4] $param0, f32[1,4] $op3)
    ROOT $root = tuple($op4, $param0)
  )";
  int loop_start_idx;
  MemoryBoundLoopOptimizer* optimizer;
  // Set alternate memory size to zero so nothing should be in the alternate
  // memory. We still expect to find an allocation for all uses.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndCreateOptimizer(hlo_loop_str,
                                                  /*alternate_memory_size=*/0,
                                                  loop_start_idx, &optimizer));

  optimizer->Optimize();
  absl::flat_hash_set<HloUse> seen_uses;
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    LOG(INFO) << loop_value.ToString();
    for (const auto& allocation : loop_value.allocations) {
      EXPECT_EQ(allocation->memory_space(), MemorySpace::kDefault);
      for (const HloUse& use : allocation->uses()) {
        EXPECT_FALSE(seen_uses.contains(use)) << use.ToString();
        seen_uses.insert(use);
      }
    }
  }

  // Ensure all of the uses in the loop have an associated use.
  for (absl::string_view inst_name : {"op0", "op1", "op2", "op3", "op4"}) {
    HloInstruction* inst =
        module->entry_computation()->GetInstructionWithName(inst_name);
    EXPECT_TRUE(seen_uses.contains(HloUse{inst, 0})) << inst_name;
    EXPECT_TRUE(seen_uses.contains(HloUse{inst, 1})) << inst_name;
  }
}

TEST_F(MemoryBoundLoopOptimizerTest, PrefetchFifoOrderWithOverlap) {
  // Test for enforcing FIFO order of prefetches. There are three parameters
  // that will be prefetched (param0, param1, and param2). param2 is one eighth
  // the size of the other parameters and is scheduled later in the loop. So, we
  // expect the allocation algorithm to initially allocate param2's prefetch
  // with a short live range (since copying it doesn't take very long), but then
  // as we try to prefetch param0 and param1, we will wrap around into the
  // previous iterations and would need to "early force" param2's prefetch to be
  // scheduled earlier to enforce the FIFO order.
  //
  // alternate_mem_bytes_per_second = 128
  // default_mem_bytes_per_second = 32
  // flops_per_second = 16
  // f32[1,4] add: flops: 4, bytes: 48, compute elapsed: 0.25
  //    - All default memory elapsed: 1.5
  //    - All alternate memory elapsed: 0.375
  // f32[8,4] add: flops: 32, bytes: 384, compute elapsed: 2
  //    - All default memory elapsed: 12
  //    - All alternate memory elapsed: 3
  // f32[1,4] copy: bytes: 16, memory elapsed: 0.5
  // f32[8,4] copy: bytes: 128, memory elapsed: 4
  absl::string_view hlo_loop_str = R"(
    $op0 = f32[1,4] add(f32[1,4] $prev_op13, f32[1,4] $prev_op14)
    $op1 = f32[8,4] add(f32[8,4] $param0, f32[8,4] $param1)
    $op2 = f32[1,4] add(f32[1,4] $prev_op14, f32[1,4] $op0)
    $op3 = f32[1,4] add(f32[1,4] $op0, f32[1,4] $op2)
    $op4 = f32[1,4] add(f32[1,4] $op2, f32[1,4] $op3)
    $op5 = f32[1,4] add(f32[1,4] $op3, f32[1,4] $op4)
    $op6 = f32[1,4] add(f32[1,4] $op4, f32[1,4] $op5)
    $op7 = f32[1,4] add(f32[1,4] $op5, f32[1,4] $op6)
    $op8 = f32[1,4] add(f32[1,4] $op6, f32[1,4] $op7)
    $op9 = f32[1,4] add(f32[1,4] $op7, f32[1,4] $op8)
    $op10 = f32[1,4] add(f32[1,4] $op8, f32[1,4] $op9)
    $op11 = f32[1,4] add(f32[1,4] $op9, f32[1,4] $op10)
    $op12 = f32[1,4] add(f32[1,4] $op10, f32[1,4] $op11)
    $op13 = f32[1,4] add(f32[1,4] $op11, f32[1,4] $op12)
    $op14 = f32[1,4] add(f32[1,4] $param2, f32[1,4] $op13)
  )";

  int loop_start_idx;
  MemoryBoundLoopOptimizer* optimizer;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndCreateOptimizer(hlo_loop_str,
                                                  /*alternate_memory_size=*/512,
                                                  loop_start_idx, &optimizer));

  optimizer->Optimize();
  // We expect the prefetches to be scheduled this way:
  //
  //
  // param0 or param1:
  // ===========>       =====================================>
  // param1 or param0:
  // ===========>                                           ===
  //           ==============================================>
  // param2:
  // =====>    ========================================>    ===
  //  13 14| 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14| 0  1
  //  prev |                  loop                      | next
  //
  // Temporaries:
  //  +======+
  //     +=========+
  //        +=========+
  //              +======+
  //                 +======+
  //                    +======+
  //                       +======+
  //                          +======+
  //                             +======+
  //                                +======+
  //                                   +======+
  //                                      +======+
  //                                         +======+
  //                                            +===+
  //                                               +======+
  //                                                  +=========+
  //  13 14| 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14| 0  1
  //  prev |                  loop                      | next
  std::vector<const CopyAllocation*> prefetches;
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    if (!loop_value.allocations.empty() &&
        loop_value.allocations.back()->is_copy_allocation()) {
      prefetches.push_back(static_cast<const CopyAllocation*>(
          loop_value.allocations.back().get()));
    }
  }
  EXPECT_EQ(prefetches.size(), 3);
  bool seen_overlap = false;
  bool seen_nonoverlap = false;
  for (const CopyAllocation* prefetch : prefetches) {
    const HloUse& use = *prefetch->uses().begin();
    if (use.instruction->name() == "op14") {
      EXPECT_EQ(prefetch->copy_done_schedule_before(), 14);
      EXPECT_EQ(prefetch->copy_start_schedule_after(), 0);
    } else {
      ASSERT_EQ(use.instruction->name(), "op1");
      EXPECT_EQ(prefetch->copy_done_schedule_before(), 1);
      if (prefetch->copy_start_schedule_after() == 0) {
        EXPECT_FALSE(seen_overlap);
        seen_overlap = true;
      } else {
        EXPECT_GT(prefetch->copy_start_schedule_after(), 1);
        EXPECT_FALSE(seen_nonoverlap);
        seen_nonoverlap = true;
      }
    }
  }
  // We expect to fully saturate the default memory bandwidth. Total default
  // memory accesses:
  //   param0 (128 B) + param1 (128 B) + op1 (128 B) + param2 (16 B) = 400 B
  // execution time:
  //  400 B / 32 B/s = 12.5 s.
  EXPECT_EQ(optimizer->CalculateExecutionTime(), 12.5);

  // Check the memory used at each point of the loop.
  const std::vector<int64_t>& remaining_memory = optimizer->remaining_memory();
  // Time 0: 3 temporaries (16 B) + param0 (128 B) + param1 (128 B)
  EXPECT_EQ(remaining_memory.at(0), 512 - (3 * 16 + 128 + 128));
  // Time 1: 2 temporaries (16 B) + 2*param0 (128 B) + param1 (128 B)
  //         + param2 (16 B)
  EXPECT_EQ(remaining_memory.at(1), 512 - (2 * 16 + 2 * 128 + 128 + 16));
  // Times 2 and 3: 3 temporaries (16 B) + param0 (128 B) + param2 (16 B)
  EXPECT_EQ(remaining_memory.at(2), 512 - (3 * 16 + 128 + 16));
  EXPECT_EQ(remaining_memory.at(3), 512 - (3 * 16 + 128 + 16));
  // Times 4 to 13: 3 temporaries (16 B) + param0 (128 B) + param1 (128 B)
  //                + param2 (16 B)
  for (int i = 4; i <= 13; ++i) {
    EXPECT_EQ(remaining_memory.at(i), 512 - (3 * 16 + 128 + 128 + 16));
  }
  // Time 14: 2 temporaries (16 B) + param0 (128 B) + param1 (128 B)
  //          + param2 (16 B)
  EXPECT_EQ(remaining_memory.at(14), 512 - (2 * 16 + 128 + 128 + 16));
}

TEST_F(MemoryBoundLoopOptimizerTest, PrefetchFifoOrderWithoutOverlap) {
  // Same as the test above, except the size of alternate memory is less than
  // 384, which is the minimum amount needed to keep the three 128-byte sized
  // parameters alive (one of the parameters would need to be overlapped with
  // the previous iteration, so counts 2X). In that case, we won't be able to
  // fully saturate the bandwidth.
  //
  // alternate_mem_bytes_per_second = 128
  // default_mem_bytes_per_second = 32
  // flops_per_second = 16
  // f32[1,4] add: flops: 4, bytes: 48, compute elapsed: 0.25
  //    - All default memory elapsed: 1.5
  //    - All alternate memory elapsed: 0.375
  // f32[8,4] add: flops: 32, bytes: 384, compute elapsed: 2
  //    - All default memory elapsed: 12
  //    - All alternate memory elapsed: 3
  // f32[1,4] copy: bytes: 16, memory elapsed: 0.5
  // f32[8,4] copy: bytes: 128, memory elapsed: 4
  absl::string_view hlo_loop_str = R"(
    $op0 = f32[1,4] add(f32[1,4] $prev_op13, f32[1,4] $prev_op14)
    $op1 = f32[8,4] add(f32[8,4] $param0, f32[8,4] $param1)
    $op2 = f32[1,4] add(f32[1,4] $prev_op14, f32[1,4] $op0)
    $op3 = f32[1,4] add(f32[1,4] $op0, f32[1,4] $op2)
    $op4 = f32[1,4] add(f32[1,4] $op2, f32[1,4] $op3)
    $op5 = f32[1,4] add(f32[1,4] $op3, f32[1,4] $op4)
    $op6 = f32[1,4] add(f32[1,4] $op4, f32[1,4] $op5)
    $op7 = f32[1,4] add(f32[1,4] $op5, f32[1,4] $op6)
    $op8 = f32[1,4] add(f32[1,4] $op6, f32[1,4] $op7)
    $op9 = f32[1,4] add(f32[1,4] $op7, f32[1,4] $op8)
    $op10 = f32[1,4] add(f32[1,4] $op8, f32[1,4] $op9)
    $op11 = f32[1,4] add(f32[1,4] $op9, f32[1,4] $op10)
    $op12 = f32[1,4] add(f32[1,4] $op10, f32[1,4] $op11)
    $op13 = f32[1,4] add(f32[1,4] $op11, f32[1,4] $op12)
    $op14 = f32[1,4] add(f32[1,4] $param2, f32[1,4] $op13)
  )";

  int loop_start_idx;
  MemoryBoundLoopOptimizer* optimizer;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndCreateOptimizer(hlo_loop_str,
                                                  /*alternate_memory_size=*/350,
                                                  loop_start_idx, &optimizer));

  optimizer->Optimize();
  // We expect the prefetches to be scheduled this way:
  //
  //
  // param0 or param1:
  // ===========>       =====================================>
  // param2:
  // =====>             ===============================>
  //  13 14| 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14| 0  1
  //  prev |                  loop                      | next
  std::vector<const CopyAllocation*> prefetches;
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    if (!loop_value.allocations.empty() &&
        loop_value.allocations.back()->is_copy_allocation()) {
      prefetches.push_back(static_cast<const CopyAllocation*>(
          loop_value.allocations.back().get()));
    }
  }
  EXPECT_EQ(prefetches.size(), 2);
  std::optional<int> expected_op14_copy_start_time;
  for (const CopyAllocation* prefetch : prefetches) {
    const HloUse& use = *prefetch->uses().begin();
    if (use.instruction->name() == "op1") {
      EXPECT_EQ(prefetch->copy_done_schedule_before(), 1);
      EXPECT_GT(prefetch->copy_start_schedule_after(), 1);
      expected_op14_copy_start_time = prefetch->copy_start_schedule_after();
    }
  }
  EXPECT_TRUE(expected_op14_copy_start_time.has_value());
  for (const CopyAllocation* prefetch : prefetches) {
    const HloUse& use = *prefetch->uses().begin();
    if (use.instruction->name() == "op14") {
      EXPECT_EQ(prefetch->copy_done_schedule_before(), 14);
      EXPECT_EQ(prefetch->copy_start_schedule_after(),
                *expected_op14_copy_start_time);
    }
  }
  // We expect not to fully saturate the default memory bandwidth.
  EXPECT_GT(optimizer->CalculateExecutionTime(), 12.5);
}

TEST_F(MemoryBoundLoopOptimizerTest, PrefetchFifoOrderWithOverlap2) {
  // Same as PrefetchFifoOrderWithOverlap, except the instructions are shifted
  // earlier by one such that param0 and param1 are used by op0. This tests that
  // we are accounting for overlaps for prefetches that span three iterations.
  //
  // alternate_mem_bytes_per_second = 128
  // default_mem_bytes_per_second = 32
  // flops_per_second = 16
  // f32[1,4] add: flops: 4, bytes: 48, compute elapsed: 0.25
  //    - All default memory elapsed: 1.5
  //    - All alternate memory elapsed: 0.375
  // f32[8,4] add: flops: 32, bytes: 384, compute elapsed: 2
  //    - All default memory elapsed: 12
  //    - All alternate memory elapsed: 3
  // f32[1,4] copy: bytes: 16, memory elapsed: 0.5
  // f32[8,4] copy: bytes: 128, memory elapsed: 4
  absl::string_view hlo_loop_str = R"(
    $op0 = f32[8,4] add(f32[8,4] $param0, f32[8,4] $param1)
    $op1 = f32[1,4] add(f32[1,4] $prev_op13, f32[1,4] $prev_op14)
    $op2 = f32[1,4] add(f32[1,4] $prev_op14, f32[1,4] $op1)
    $op3 = f32[1,4] add(f32[1,4] $op1, f32[1,4] $op2)
    $op4 = f32[1,4] add(f32[1,4] $op2, f32[1,4] $op3)
    $op5 = f32[1,4] add(f32[1,4] $op3, f32[1,4] $op4)
    $op6 = f32[1,4] add(f32[1,4] $op4, f32[1,4] $op5)
    $op7 = f32[1,4] add(f32[1,4] $op5, f32[1,4] $op6)
    $op8 = f32[1,4] add(f32[1,4] $op6, f32[1,4] $op7)
    $op9 = f32[1,4] add(f32[1,4] $op7, f32[1,4] $op8)
    $op10 = f32[1,4] add(f32[1,4] $op8, f32[1,4] $op9)
    $op11 = f32[1,4] add(f32[1,4] $op9, f32[1,4] $op10)
    $op12 = f32[1,4] add(f32[1,4] $op10, f32[1,4] $op11)
    $op13 = f32[1,4] add(f32[1,4] $param2, f32[1,4] $op12)
    $op14 = f32[1,4] add(f32[1,4] $op12, f32[1,4] $op13)
  )";

  int loop_start_idx;
  MemoryBoundLoopOptimizer* optimizer;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndCreateOptimizer(hlo_loop_str,
                                                  /*alternate_memory_size=*/512,
                                                  loop_start_idx, &optimizer));

  optimizer->Optimize();
  // We expect the prefetches to be scheduled this way:
  //
  //
  // param0 or param1:
  // ========>       =====================================> ===
  // param1 or param0:
  // ========>                                           ======
  //        ==============================================>
  // param2:
  // ==>    ========================================>    ======
  //  13 14| 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14| 0  1
  //  prev |                  loop                      | next
  std::vector<const CopyAllocation*> prefetches;
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    if (!loop_value.allocations.empty() &&
        loop_value.allocations.back()->is_copy_allocation()) {
      prefetches.push_back(static_cast<const CopyAllocation*>(
          loop_value.allocations.back().get()));
    }
  }
  EXPECT_EQ(prefetches.size(), 3);
  bool seen_overlap = false;
  bool seen_nonoverlap = false;
  for (const CopyAllocation* prefetch : prefetches) {
    const HloUse& use = *prefetch->uses().begin();
    if (use.instruction->name() == "op13") {
      EXPECT_EQ(prefetch->copy_done_schedule_before(), 13);
      EXPECT_EQ(prefetch->copy_start_schedule_after(), 14);
    } else {
      ASSERT_EQ(use.instruction->name(), "op0");
      EXPECT_EQ(prefetch->copy_done_schedule_before(), 0);
      if (prefetch->copy_start_schedule_after() == 14) {
        EXPECT_FALSE(seen_overlap);
        seen_overlap = true;
      } else {
        EXPECT_LT(prefetch->copy_start_schedule_after(), 14);
        EXPECT_FALSE(seen_nonoverlap);
        seen_nonoverlap = true;
      }
    }
  }
  // We expect to fully saturate the default memory bandwidth. Total default
  // memory accesses:
  //   param0 (128 B) + param1 (128 B) + op1 (128 B) + param2 (16 B) = 400 B
  // execution time:
  //  400 B / 32 B/s = 12.5 s.
  EXPECT_EQ(optimizer->CalculateExecutionTime(), 12.5);
}

TEST_F(MemoryBoundLoopOptimizerTest, OptimizerEndToEnd) {
  absl::string_view hlo_loop_str = R"(
    $op0 = f32[1,4] add(f32[1,4] $prev_op13, f32[1,4] $prev_op14)
    $op1 = f32[8,4] add(f32[8,4] $param0, f32[8,4] $param1)
    $op2 = f32[1,4] add(f32[1,4] $prev_op14, f32[1,4] $op0)
    $op3 = f32[1,4] add(f32[1,4] $op0, f32[1,4] $op2)
    $op4 = f32[1,4] add(f32[1,4] $op2, f32[1,4] $op3)
    $op5 = f32[1,4] add(f32[1,4] $op3, f32[1,4] $op4)
    $op6 = f32[1,4] add(f32[1,4] $op4, f32[1,4] $op5)
    $op7 = f32[1,4] add(f32[1,4] $op5, f32[1,4] $op6)
    $op8 = f32[1,4] add(f32[1,4] $op6, f32[1,4] $op7)
    $op9 = f32[1,4] add(f32[1,4] $op7, f32[1,4] $op8)
    $op10 = f32[1,4] add(f32[1,4] $op8, f32[1,4] $op9)
    $op11 = f32[1,4] add(f32[1,4] $op9, f32[1,4] $op10)
    $op12 = f32[1,4] add(f32[1,4] $op10, f32[1,4] $op11)
    $op13 = f32[1,4] add(f32[1,4] $op11, f32[1,4] $op12)
    $op14 = f32[1,4] add(f32[1,4] $param2, f32[1,4] $op13)
    ROOT $root = tuple($op1, $op14)
  )";

  int loop_start_idx;
  MemoryBoundLoopOptimizer* optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndCreateOptimizer(hlo_loop_str,
                                           /*alternate_memory_size=*/1024,
                                           loop_start_idx, &optimizer));

  optimizer->Optimize();
  TF_ASSERT_OK_AND_ASSIGN(auto preset_assignments,
                          RunMsa(module.get(), /*alternate_memory_size=*/1024));

  TF_ASSERT_OK(VerifyMsaEquivalence(module.get()));
}

TEST_F(MemoryBoundLoopOptimizerTest, OptimizerEndToEndUnsupportedAllocation) {
  // op2 is a loop-carried dependency, which is currently not supported. But the
  // usual MSA algorithm should still be able to give it an alternate memory
  // allocation.
  absl::string_view hlo_loop_str = R"(
    $op0 = f32[1,4] add(f32[1,4] $prev_op3, f32[1,4] $prev_op4)
    $op1 = f32[8,4] add(f32[8,4] $param0, f32[8,4] $param1)
    $op2 = f32[1,4] add(f32[1,4] $prev_op2, f32[1,4] $op0)
    $op3 = f32[1,4] add(f32[1,4] $op0, f32[1,4] $op2)
    $op4 = f32[1,4] add(f32[1,4] $op2, f32[1,4] $op3)
    ROOT $root = tuple($op1, $op4)
  )";

  int loop_start_idx;
  MemoryBoundLoopOptimizer* optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndCreateOptimizer(hlo_loop_str,
                                           /*alternate_memory_size=*/1024,
                                           loop_start_idx, &optimizer));

  optimizer->Optimize();
  TF_ASSERT_OK_AND_ASSIGN(auto preset_assignments,
                          RunMsa(module.get(), /*alternate_memory_size=*/1024));

  TF_ASSERT_OK(VerifyMsaEquivalence(module.get(),
                                    /*expect_unsupported_allocations=*/true));

  const HloInstruction* op2 = FindInstruction(module.get(), "op2");
  EXPECT_EQ(op2->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_F(MemoryBoundLoopOptimizerTest, TempAndPinnedAllocations) {
  absl::string_view hlo_str = R"(
  HloModule module, is_scheduled=true

  while_cond {
    while_cond_param = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) parameter(0)
    ROOT p = pred[] get-tuple-element(while_cond_param), index=5
  }

  while_body {
    while_body_param = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) parameter(0)
    pinned_prev_param0 = f32[1,4] get-tuple-element(while_body_param), index=0
    next_param0 = f32[1,4] get-tuple-element(while_body_param), index=1
    prev_prev_op3 = f32[1,4] get-tuple-element(while_body_param), index=2
    prev_prev_op4 = f32[1,4] get-tuple-element(while_body_param), index=3
    prev_op0 = f32[1,4] add(f32[1,4] prev_prev_op3, f32[1,4] prev_prev_op4)
    prev_op1 = f32[1,4] add(f32[1,4] prev_prev_op4, f32[1,4] prev_op0)
    prev_op2 = f32[1,4] add(f32[1,4] prev_op0, f32[1,4] prev_op1)
    prev_op3 = f32[1,4] add(f32[1,4] prev_op1, f32[1,4] prev_op2)
    prev_op4 = f32[1,4] multiply(f32[1,4] pinned_prev_param0, f32[1,4] prev_op3)
    op0 = f32[1,4] add(f32[1,4] prev_op3, f32[1,4] prev_op4)
    op1 = f32[1,4] add(f32[1,4] prev_op4, f32[1,4] op0)
    op2 = f32[1,4] add(f32[1,4] op0, f32[1,4] op1)
    op3 = f32[1,4] add(f32[1,4] op1, f32[1,4] op2)
    op4 = f32[1,4] multiply(f32[1,4] pinned_prev_param0, f32[1,4] op3)
    next_op0 = f32[1,4] add(f32[1,4] op3, f32[1,4] op4)
    next_op1 = f32[1,4] add(f32[1,4] op4, f32[1,4] next_op0)
    next_op2 = f32[1,4] add(f32[1,4] next_op0, f32[1,4] next_op1)
    next_op3 = f32[1,4] add(f32[1,4] next_op1, f32[1,4] next_op2)
    next_op4 = f32[1,4] multiply(f32[1,4] pinned_prev_param0, f32[1,4] next_op3)
    p = pred[] get-tuple-element(while_body_param), index=5
    ROOT root = tuple(pinned_prev_param0, next_param0, prev_prev_op3, prev_prev_op4, next_op4, p)
  }

  ENTRY entry {
    p0 = f32[1,4] parameter(0)
    p1 = f32[1,4] parameter(1)
    p2 = f32[1,4] parameter(2)
    p3 = f32[1,4] parameter(3)
    p4 = pred[] parameter(4)
    copy = f32[1,4] copy(p3)
    tuple = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) tuple(p0, p1, p2, p3, copy, p4)
    while = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) while(tuple), condition=while_cond, body=while_body
    ROOT root = f32[1,4] get-tuple-element(while), index=4
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));

  TF_ASSERT_OK_AND_ASSIGN(auto optimizer,
                          CreateOptimizer(19, 24, module.get(),
                                          /*alternate_memory_size=*/512));
  optimizer->Optimize();

  const std::vector<int64_t>& remaining_memory = optimizer->remaining_memory();
  // Time 0: 3 temporaries (16 B) + 1 pinned (16 B)
  EXPECT_EQ(remaining_memory.at(0), 512 - (3 * 16 + 16));
  // Time 1: 3 temporaries (16 B) + 1 pinned (16 B)
  EXPECT_EQ(remaining_memory.at(1), 512 - (3 * 16 + 16));
  // Time 2: 3 temporaries (16 B) + 1 pinned (16 B)
  EXPECT_EQ(remaining_memory.at(2), 512 - (3 * 16 + 16));
  // Time 3: 3 temporaries (16 B) + 1 pinned (16 B)
  EXPECT_EQ(remaining_memory.at(3), 512 - (3 * 16 + 16));
  // Time 4: 2 temporaries (16 B) + 1 pinned (16 B)
  EXPECT_EQ(remaining_memory.at(4), 512 - (2 * 16 + 16));
}

TEST_F(MemoryBoundLoopOptimizerTest, NegativeSavingNotPinned) {
  absl::string_view hlo_str = R"(
  HloModule module, is_scheduled=true

  while_cond {
    while_cond_param = (f32[28,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) parameter(0)
    ROOT p = pred[] get-tuple-element(while_cond_param), index=5
  }

  while_body {
    while_body_param = (f32[28,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) parameter(0)
    pinned_prev_param0 = f32[28,4] get-tuple-element(while_body_param), index=0
    zero = s32[] constant(0)
    next_param0 = f32[1,4] get-tuple-element(while_body_param), index=1
    prev_prev_op3 = f32[1,4] get-tuple-element(while_body_param), index=2
    prev_prev_op4 = f32[1,4] get-tuple-element(while_body_param), index=3
    prev_op0 = f32[1,4] add(f32[1,4] prev_prev_op3, f32[1,4] prev_prev_op4)
    prev_op1 = f32[1,4] add(f32[1,4] prev_prev_op4, f32[1,4] prev_op0)
    prev_op2 = f32[1,4] add(f32[1,4] prev_op0, f32[1,4] prev_op1)
    prev_op3 = f32[1,4] add(f32[1,4] prev_op1, f32[1,4] prev_op2)
    pinned_slice = f32[1,4] dynamic-slice(pinned_prev_param0, zero, zero), dynamic_slice_sizes={1,4}
    prev_op4 = f32[1,4] multiply(f32[1,4] pinned_slice, f32[1,4] prev_op3)
    op0 = f32[1,4] add(f32[1,4] prev_op3, f32[1,4] prev_op4)
    op1 = f32[1,4] add(f32[1,4] prev_op4, f32[1,4] op0)
    op2 = f32[1,4] add(f32[1,4] op0, f32[1,4] op1)
    op3 = f32[1,4] add(f32[1,4] op1, f32[1,4] op2)
    pinned_slice2 = f32[1,4] dynamic-slice(pinned_prev_param0, zero, zero), dynamic_slice_sizes={1,4}
    op4 = f32[1,4] multiply(f32[1,4] pinned_slice2, f32[1,4] op3)
    next_op0 = f32[1,4] add(f32[1,4] op3, f32[1,4] op4)
    next_op1 = f32[1,4] add(f32[1,4] op4, f32[1,4] next_op0)
    next_op2 = f32[1,4] add(f32[1,4] next_op0, f32[1,4] next_op1)
    next_op3 = f32[1,4] add(f32[1,4] next_op1, f32[1,4] next_op2)
    pinned_slice3 = f32[1,4] dynamic-slice(pinned_prev_param0, zero, zero), dynamic_slice_sizes={1,4}
    next_op4 = f32[1,4] multiply(f32[1,4] pinned_slice3, f32[1,4] next_op3)
    p = pred[] get-tuple-element(while_body_param), index=5
    ROOT root = tuple(pinned_prev_param0, next_param0, prev_prev_op3, prev_prev_op4, next_op4, p)
  }

  ENTRY entry {
    p0 = f32[28,4] parameter(0)
    p1 = f32[1,4] parameter(1)
    p2 = f32[1,4] parameter(2)
    p3 = f32[1,4] parameter(3)
    p4 = pred[] parameter(4)
    copy = f32[1,4] copy(p3)
    tuple = (f32[28,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) tuple(p0, p1, p2, p3, copy, p4)
    while = (f32[28,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) while(tuple), condition=while_cond, body=while_body
    ROOT root = f32[1,4] get-tuple-element(while), index=4
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));

  TF_ASSERT_OK_AND_ASSIGN(auto optimizer,
                          CreateOptimizer(21, 27, module.get(),
                                          /*alternate_memory_size=*/512));
  optimizer->Optimize();

  const std::vector<int64_t>& remaining_memory = optimizer->remaining_memory();
  // We expect that pinned_prev_param0 would not get pinned due to negative
  // savings: 32(uses) -  28 * 16(size) = -416 Time 0: 3 temporaries (16 B) + 1
  // pinned (4 B)
  EXPECT_EQ(remaining_memory.at(0), 512 - (3 * 16 + 4));
}

TEST_F(MemoryBoundLoopOptimizerTest, OptimizerEndToEndWhileLoop) {
  absl::string_view hlo_str = R"(
HloModule module, is_scheduled=true

while_cond {
  while_cond_param = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) parameter(0)
  ROOT p = pred[] get-tuple-element(while_cond_param), index=6
}

while_body {
  while_body_param = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) parameter(0)
  prev_param0 = f32[1,4] get-tuple-element(while_body_param), index=0
  param0 = f32[1,4] get-tuple-element(while_body_param), index=1
  next_param0 = f32[1,4] get-tuple-element(while_body_param), index=2
  prev_prev_op3 = f32[1,4] get-tuple-element(while_body_param), index=3
  prev_prev_op4 = f32[1,4] get-tuple-element(while_body_param), index=4
  prev_op0 = f32[1,4] add(f32[1,4] prev_prev_op3, f32[1,4] prev_prev_op4)
  prev_op1 = f32[1,4] add(f32[1,4] prev_prev_op4, f32[1,4] prev_op0)
  prev_op2 = f32[1,4] add(f32[1,4] prev_op0, f32[1,4] prev_op1)
  prev_op3 = f32[1,4] add(f32[1,4] prev_op1, f32[1,4] prev_op2)
  prev_op4 = f32[1,4] multiply(f32[1,4] prev_param0, f32[1,4] prev_op3)
  op0 = f32[1,4] add(f32[1,4] prev_op3, f32[1,4] prev_op4)
  op1 = f32[1,4] add(f32[1,4] prev_op4, f32[1,4] op0)
  op2 = f32[1,4] add(f32[1,4] op0, f32[1,4] op1)
  op3 = f32[1,4] add(f32[1,4] op1, f32[1,4] op2)
  op4 = f32[1,4] multiply(f32[1,4] param0, f32[1,4] op3)
  next_op0 = f32[1,4] add(f32[1,4] op3, f32[1,4] op4)
  next_op1 = f32[1,4] add(f32[1,4] op4, f32[1,4] next_op0)
  next_op2 = f32[1,4] add(f32[1,4] next_op0, f32[1,4] next_op1)
  next_op3 = f32[1,4] add(f32[1,4] next_op1, f32[1,4] next_op2)
  next_op4 = f32[1,4] multiply(f32[1,4] next_param0, f32[1,4] next_op3)
  p = pred[] get-tuple-element(while_body_param), index=6
  ROOT root = tuple(prev_param0, param0, next_param0, prev_prev_op3, prev_prev_op4, next_op4, p)
}

ENTRY entry {
  p0 = f32[1,4] parameter(0)
  p1 = f32[1,4] parameter(1)
  p2 = f32[1,4] parameter(2)
  p3 = f32[1,4] parameter(3)
  p4 = f32[1,4] parameter(4)
  p5 = pred[] parameter(5)
  copy = f32[1,4] copy(p4)
  tuple = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) tuple(p0, p1, p2, p3, p4, copy, p5)
  while = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) while(tuple), condition=while_cond, body=while_body
  ROOT root = f32[1,4] get-tuple-element(while), index=5
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));

  TF_ASSERT_OK_AND_ASSIGN(auto preset_assignments,
                          RunMsa(module.get(), /*alternate_memory_size=*/512));

  // We expect operand 0 of prev_op4, op4, and next_op4 to all be prefetches of
  // same distance from the user.
  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_live_range,
                          HloLiveRange::Run(module->schedule(), *alias_analysis,
                                            module->entry_computation()));
  const HloInstruction* prev_copy_done =
      FindInstruction(module.get(), "prev_op4")->operand(0);
  const HloInstruction* copy_done =
      FindInstruction(module.get(), "op4")->operand(0);
  const HloInstruction* next_copy_done =
      FindInstruction(module.get(), "next_op4")->operand(0);
  ASSERT_EQ(prev_copy_done->opcode(), HloOpcode::kCopyDone);
  ASSERT_EQ(copy_done->opcode(), HloOpcode::kCopyDone);
  ASSERT_EQ(next_copy_done->opcode(), HloOpcode::kCopyDone);
  EXPECT_EQ(prev_copy_done->shape().layout().memory_space(),
            kAlternateMemorySpace);
  EXPECT_EQ(copy_done->shape().layout().memory_space(), kAlternateMemorySpace);
  EXPECT_EQ(next_copy_done->shape().layout().memory_space(),
            kAlternateMemorySpace);
  auto prefetch_distance = [&](const HloInstruction* copy_done) {
    return hlo_live_range->instruction_schedule().at(copy_done) -
           hlo_live_range->instruction_schedule().at(copy_done->operand(0));
  };
  EXPECT_EQ(prefetch_distance(prev_copy_done), prefetch_distance(copy_done));
  EXPECT_EQ(prefetch_distance(next_copy_done), prefetch_distance(copy_done));
}

TEST_F(MemoryBoundLoopOptimizerTest, OptimizerEndToEndNestedWhileLoopBug) {
  absl::string_view hlo_str = R"(
HloModule module, is_scheduled=true

prev_while_cond {
  prev_while_cond_param = (f32[1,4], pred[]) parameter(0)
  ROOT p = pred[] get-tuple-element(prev_while_cond_param), index=1
}

prev_while_body {
  prev_while_body_param = (f32[1,4], pred[]) parameter(0)
  prev_while_body_gte = f32[1,4] get-tuple-element(prev_while_body_param), index=0
  prev_while_body_pred = pred[] get-tuple-element(prev_while_body_param), index=1
  prev_while_body_op = f32[1,4] negate(prev_while_body_gte)
  ROOT prev_while_body_root = (f32[1,4], pred[]) tuple(prev_while_body_op, prev_while_body_pred)
}

current_while_cond {
  current_while_cond_param = (f32[1,4], pred[]) parameter(0)
  ROOT p = pred[] get-tuple-element(current_while_cond_param), index=1
}

current_while_body {
  current_while_body_param = (f32[1,4], pred[]) parameter(0)
  current_while_body_gte = f32[1,4] get-tuple-element(current_while_body_param), index=0
  current_while_body_pred = pred[] get-tuple-element(current_while_body_param), index=1
  current_while_body_op = f32[1,4] negate(current_while_body_gte)
  ROOT current_while_body_root = (f32[1,4], pred[]) tuple(current_while_body_op, current_while_body_pred)
}

next_while_cond {
  next_while_cond_param = (f32[1,4], pred[]) parameter(0)
  ROOT p = pred[] get-tuple-element(next_while_cond_param), index=1
}

next_while_body {
  next_while_body_param = (f32[1,4], pred[]) parameter(0)
  next_while_body_gte = f32[1,4] get-tuple-element(next_while_body_param), index=0
  next_while_body_pred = pred[] get-tuple-element(next_while_body_param), index=1
  next_while_body_op = f32[1,4] negate(next_while_body_gte)
  ROOT next_while_body_root = (f32[1,4], pred[]) tuple(next_while_body_op, next_while_body_pred)
}

while_cond {
  while_cond_param = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) parameter(0)
  ROOT p = pred[] get-tuple-element(while_cond_param), index=6
}

while_body {
  while_body_param = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) parameter(0)
  prev_param0 = f32[1,4] get-tuple-element(while_body_param), index=0
  param0 = f32[1,4] get-tuple-element(while_body_param), index=1
  next_param0 = f32[1,4] get-tuple-element(while_body_param), index=2
  prev_prev_op3 = f32[1,4] get-tuple-element(while_body_param), index=3
  prev_prev_op4 = f32[1,4] get-tuple-element(while_body_param), index=4
  while_pred = pred[] get-tuple-element(while_body_param), index=6
  prev_op0 = f32[1,4] add(f32[1,4] prev_prev_op3, f32[1,4] prev_prev_op4)
  prev_op1 = f32[1,4] add(f32[1,4] prev_prev_op4, f32[1,4] prev_op0)
  prev_op2 = f32[1,4] add(f32[1,4] prev_op0, f32[1,4] prev_op1)
  prev_op3 = f32[1,4] add(f32[1,4] prev_op1, f32[1,4] prev_op2)
  prev_tuple = (f32[1,4], pred[]) tuple(prev_op3, while_pred)
  prev_while = (f32[1,4], pred[]) while(prev_tuple), condition=prev_while_cond, body=prev_while_body
  prev_gte = f32[1,4] get-tuple-element(prev_while), index=0
  prev_op4 = f32[1,4] multiply(f32[1,4] prev_param0, f32[1,4] prev_gte)
  op0 = f32[1,4] add(f32[1,4] prev_op3, f32[1,4] prev_op4)
  op1 = f32[1,4] add(f32[1,4] prev_op4, f32[1,4] op0)
  op2 = f32[1,4] add(f32[1,4] op0, f32[1,4] op1)
  op3 = f32[1,4] add(f32[1,4] op1, f32[1,4] op2)
  current_tuple = (f32[1,4], pred[]) tuple(op3, while_pred)
  current_while = (f32[1,4], pred[]) while(current_tuple), condition=current_while_cond, body=current_while_body
  current_gte = f32[1,4] get-tuple-element(current_while), index=0
  op4 = f32[1,4] multiply(f32[1,4] param0, f32[1,4] current_gte)
  next_op0 = f32[1,4] add(f32[1,4] op3, f32[1,4] op4)
  next_op1 = f32[1,4] add(f32[1,4] op4, f32[1,4] next_op0)
  next_op2 = f32[1,4] add(f32[1,4] next_op0, f32[1,4] next_op1)
  next_op3 = f32[1,4] add(f32[1,4] next_op1, f32[1,4] next_op2)
  next_tuple = (f32[1,4], pred[]) tuple(next_op3, while_pred)
  next_while = (f32[1,4], pred[]) while(next_tuple), condition=next_while_cond, body=next_while_body
  next_gte = f32[1,4] get-tuple-element(next_while), index=0
  next_op4 = f32[1,4] multiply(f32[1,4] next_param0, f32[1,4] next_gte)
  ROOT root = tuple(prev_param0, param0, next_param0, prev_prev_op3, prev_prev_op4, next_op4, while_pred)
}

ENTRY entry {
  p0 = f32[1,4] parameter(0)
  p1 = f32[1,4] parameter(1)
  p2 = f32[1,4] parameter(2)
  p3 = f32[1,4] parameter(3)
  p4 = f32[1,4] parameter(4)
  p5 = pred[] parameter(5)
  copy = f32[1,4] copy(p4)
  tuple = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) tuple(p0, p1, p2, p3, p4, copy, p5)
  while = (f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], f32[1,4], pred[]) while(tuple), condition=while_cond, body=while_body
  ROOT root = f32[1,4] get-tuple-element(while), index=5
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_str));

  TF_ASSERT_OK_AND_ASSIGN(auto preset_assignments,
                          RunMsa(module.get(), /*alternate_memory_size=*/512));
}

}  // namespace
}  // namespace memory_space_assignment
}  // namespace xla
