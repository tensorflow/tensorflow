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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_SPACE_ASSIGNMENT_TEST_BASE_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_SPACE_ASSIGNMENT_TEST_BASE_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/simplifiers/instruction_hoister.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cost_modelling/op_cost.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/buffer_interval_comparator.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/service/memory_space_assignment/prefetch_interval_picker.h"
#include "xla/service/memory_space_assignment/utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace memory_space_assignment {

constexpr int64_t kPointerSize = 8;
constexpr float kDefaultMemBandwidth = 100;
constexpr float kAlternateMemBandwidth = 1000;
constexpr float kBytesPerSecond = 100;
constexpr float kFlopsPerSecond = 1000;
constexpr float kTranscendentalsPerSecond = 10;

class TestBufferIntervalComparator : public BufferIntervalComparator {
 public:
  explicit TestBufferIntervalComparator(MsaBufferIntervalCompare compare_method)
      : compare_method_(std::move(compare_method)) {}

  ~TestBufferIntervalComparator() override = default;

  std::string DescribeComparisonCriteria() const override {
    return "internal to test";
  }
  std::string CriteriaToString(
      const MsaBufferInterval& buffer_interval) override {
    return "internal to test";
  }
  bool LessThan(const MsaBufferInterval& lhs,
                const MsaBufferInterval& rhs) override {
    return compare_method_(lhs, rhs);
  }

 private:
  MsaBufferIntervalCompare compare_method_;
};

class MemorySpaceAssignmentTestBase : public HloTestBase {
 protected:
  // We use the following two memory space values to describe the default (slow
  // and large) and alternate (fast and small) memory spaces.
  const int64_t kDefaultMemorySpace = 0;
  const int64_t kAlternateMemorySpace = 1;
  const int64_t kPinnedDefaultMemorySpace = 2;

  static HloCostAnalysis::Options DefaultHloCostAnalysisOptions() {
    HloCostAnalysis::Options options;
    options.set_flops_per_second(kFlopsPerSecond);
    options.set_bytes_per_second(kBytesPerSecond);
    options.set_transcendentals_per_second(kTranscendentalsPerSecond);

    return options;
  }

  Options DefaultMemorySpaceOptions() const {
    Options options;
    options.max_size_in_bytes = 128;
    options.alignment_in_bytes = 8;
    options.verify = false;
    options.alternate_memory_space = kAlternateMemorySpace;
    options.max_outstanding_prefetches = -1;
    options.max_outstanding_evictions = -1;
    options.shape_size_fn = HloCostAnalysis::DefaultShapeSize;

    return options;
  }

  static CostAnalysisOptions DefaultCostAnalysisOptions() {
    CostAnalysisOptions options;
    options.default_mem_bandwidth_bytes_per_second = kDefaultMemBandwidth;
    options.alternate_mem_read_bandwidth_bytes_per_second =
        kAlternateMemBandwidth;
    options.alternate_mem_write_bandwidth_bytes_per_second =
        kAlternateMemBandwidth;
    return options;
  }

  static Options UpdateMaxAsyncCopies(Options options,
                                      int64_t max_async_copies) {
    options.max_outstanding_prefetches = max_async_copies;
    options.max_outstanding_evictions = max_async_copies;

    return options;
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpaceUsingCostAnalysis(
      HloModule* module,
      std::optional<Options> memory_space_options_override = std::nullopt,
      std::optional<CostAnalysisOptions> cost_analysis_options_override =
          std::nullopt,
      std::optional<HloCostAnalysis::Options> hlo_cost_options_override =
          std::nullopt,
      std::optional<MsaSortOrderOverrides> optional_msa_sort_order_overrides =
          std::nullopt) {
    HloCostAnalysis::Options hlo_cost_options = DefaultHloCostAnalysisOptions();
    if (hlo_cost_options_override) {
      hlo_cost_options = *hlo_cost_options_override;
    }

    HloCostAnalysis hlo_cost_analysis(hlo_cost_options);
    HloCostAnalysisWithAcceptState hlo_cost_analysis_wrapper(hlo_cost_analysis);
    for (HloComputation* computation : module->MakeNonfusionComputations()) {
      TF_CHECK_OK(computation->Accept(&hlo_cost_analysis));
    }
    TF_CHECK_OK(HloAliasAnalysis::Run(module).status());

    Options memory_space_options = DefaultMemorySpaceOptions();
    if (memory_space_options_override) {
      memory_space_options = *memory_space_options_override;
    }
    CostAnalysisOptions cost_analysis_options = DefaultCostAnalysisOptions();
    if (cost_analysis_options_override) {
      cost_analysis_options = *cost_analysis_options_override;
    }
    OpCostManager op_cost_manager(
        OpCostManager::Options{
            /*enable_cache=*/false,
            /*enable_analysis_logging=*/false,
        },
        OpCostManager::CalculationNode::CreateLeaf(
            "HloCostAnalysis",
            CreateHloCostAnalysisCalculator(hlo_cost_analysis_wrapper),
            /*enable_cache=*/false));

    auto status_or_cost_analysis =
        CostAnalysis::Create(op_cost_manager, cost_analysis_options, *module);
    TF_CHECK_OK(status_or_cost_analysis.status());
    auto cost_analysis = std::move(status_or_cost_analysis.value());

    memory_space_options.cost_analysis = cost_analysis.get();
    CostAnalysisPrefetchIntervalPicker prefetch_interval_picker(
        CostAnalysisPrefetchIntervalPicker(
            *cost_analysis, /*min_overlap_to_async_copy_ratio=*/0.8f,
            /*preferred_overlap_to_async_copy_ratio=*/1.5,
            /*max_overlap_to_mem_size_async_copy_ratio=*/10.0,
            /*mem_size_bytes=*/memory_space_options.max_size_in_bytes));
    MsaSortOrderOverrides msa_sort_order_overrides;
    if (optional_msa_sort_order_overrides.has_value()) {
      msa_sort_order_overrides = optional_msa_sort_order_overrides.value();
    }
    MemoryBoundednessBufferIntervalComparator comparator(
        *cost_analysis, &cache_, msa_sort_order_overrides);
    return AssignMemorySpace(
        module, memory_space_options,
        [&comparator](const MsaBufferInterval& lhs,
                      const MsaBufferInterval& rhs) {
          return comparator.LessThan(lhs, rhs);
        },
        &prefetch_interval_picker);
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, std::optional<Options> options_override = std::nullopt,
      int64_t max_prefetch_interval = 10, int64_t min_prefetch_interval = 2) {
    InstructionHoister instruction_hoister;
    TF_CHECK_OK(instruction_hoister.Run(module).status());
    InstructionCountPrefetchIntervalPicker prefetch_interval_picker(
        min_prefetch_interval, max_prefetch_interval);
    return AssignMemorySpace(module, std::move(options_override),
                             /*buffer_interval_compare=*/{},
                             &prefetch_interval_picker);
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, std::optional<Options> options_override,
      std::optional<MsaBufferIntervalCompare> buffer_interval_compare,
      PrefetchIntervalPicker* prefetch_interval_picker) {
    auto status_or = AssignMemorySpaceAndReturnStatus(
        module, std::move(options_override), std::move(buffer_interval_compare),
        prefetch_interval_picker);
    TF_EXPECT_OK(status_or.status());
    return std::move(status_or.value());
  }

  absl::StatusOr<std::unique_ptr<PresetAssignments>>
  AssignMemorySpaceAndReturnStatus(
      HloModule* module, std::optional<Options> options_override,
      std::optional<MsaBufferIntervalCompare> buffer_interval_compare,
      PrefetchIntervalPicker* prefetch_interval_picker) {
    auto size_fn = [](const BufferValue& buffer) {
      return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
    };

    auto is_allowed_in_alternate_mem = [](const HloValue& value) {
      // Check if the value belongs to the entry computation.
      HloInstruction* instruction = value.instruction();
      HloComputation* computation = instruction->parent();
      bool in_entry_computation =
          (computation == computation->parent()->entry_computation());

      return (!in_entry_computation ||
              instruction->opcode() != HloOpcode::kParameter);
    };

    // Only check parameters in default memory if the original module didn't
    // have the parameters in alternate memory.
    bool check_parameters_in_default_memory = true;
    for (const HloInstruction* parameter :
         module->entry_computation()->parameter_instructions()) {
      ShapeUtil::ForEachSubshape(
          parameter->shape(),
          [&](const Shape& subshape, const ShapeIndex& /*index*/) {
            if (subshape.has_layout() &&
                subshape.layout().memory_space() == kAlternateMemorySpace) {
              check_parameters_in_default_memory = false;
            }
          });
    }

    Options options = DefaultMemorySpaceOptions();
    if (options_override) {
      options = *options_override;
    }
    std::unique_ptr<TestBufferIntervalComparator> test_comparator;
    if (buffer_interval_compare.has_value()) {
      test_comparator = std::make_unique<TestBufferIntervalComparator>(
          *buffer_interval_compare);
      options.buffer_interval_comparator = test_comparator.get();
    }
    options.prefetch_interval_picker = prefetch_interval_picker;
    options.size_fn = size_fn;
    if (options.is_allowed_in_alternate_mem_fn == nullptr) {
      options.is_allowed_in_alternate_mem_fn = is_allowed_in_alternate_mem;
    }

    TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(module));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloLiveRange> hlo_live_range,
                        HloLiveRange::Run(module->schedule(), *alias_analysis,
                                          module->entry_computation()));

    TF_ASSIGN_OR_RETURN(std::unique_ptr<PresetAssignments> preset_assignments,
                        MemorySpaceAssignment::Run(module, *hlo_live_range,
                                                   *alias_analysis, options));
    if (check_parameters_in_default_memory) {
      CheckParametersInDefaultMemory(module);
    }
    CheckRootInDefaultMemory(module);
    CheckPresetAssignments(preset_assignments.get());
    return preset_assignments;
  }

  void CheckPresetAssignments(const PresetAssignments* preset_assignments) {
    // Ensure that the exported preset assignments point to layouts in the
    // alternate memory.  Also ensure that the positions are unique. Note that
    // we're using a std::set instead of absl::flat_hash_set because we can make
    // use of HloPosition's comparator logic instead of providing a hasher.
    std::set<HloPosition> positions_in_preset_assignments;
    for (auto& position_and_chunk : preset_assignments->chunks()) {
      HloPosition position = position_and_chunk.first;
      EXPECT_EQ(positions_in_preset_assignments.find(position),
                positions_in_preset_assignments.end());
      positions_in_preset_assignments.insert(position);
      const Shape& subshape =
          ShapeUtil::GetSubshape(position.instruction->shape(), position.index);
      EXPECT_EQ(subshape.layout().memory_space(), kAlternateMemorySpace)
          << "Exported position is not in alternate mem: "
          << position.ToString();
    }
  }

  void CheckParametersInDefaultMemory(const HloModule* module) {
    // Check that all the entry parameter subshapes are placed in default
    // memory.
    const HloComputation* entry_computation = module->entry_computation();
    for (const HloInstruction* parameter :
         entry_computation->parameter_instructions()) {
      ShapeUtil::ForEachSubshape(
          parameter->shape(),
          [&](const Shape& subshape, const ShapeIndex& /*index*/) {
            if (subshape.has_layout()) {
              EXPECT_NE(subshape.layout().memory_space(), kAlternateMemorySpace)
                  << "Parameter not in default memory: "
                  << parameter->ToString();
            }
          });
    }
  }

  void CheckRootInDefaultMemory(const HloModule* module) {
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    if (root->shape().IsArray()) {
      EXPECT_EQ(root->shape().layout().memory_space(), kDefaultMemorySpace);
    }
  }

  struct OutstandingAsyncCopies {
    int64_t max_copies;
    int64_t max_prefetches;
    int64_t max_evictions;
  };

  /*static*/ OutstandingAsyncCopies CountMaximumOutstandingAsyncCopies(
      const HloModule& module) const {
    OutstandingAsyncCopies copies{0, 0, 0};
    int64_t current_copies = 0;
    int64_t current_prefetches = 0;
    int64_t current_evictions = 0;
    for (HloInstruction* instruction : module.schedule()
                                           .sequence(module.entry_computation())
                                           .instructions()) {
      if (instruction->opcode() == HloOpcode::kCopyStart) {
        current_copies++;
        if (ShapeUtil::GetSubshape(instruction->shape(), {0})
                .layout()
                .memory_space() == kAlternateMemorySpace) {
          current_prefetches++;
        } else {
          current_evictions++;
        }
      } else if (instruction->opcode() == HloOpcode::kCopyDone) {
        current_copies--;
        if (instruction->shape().layout().memory_space() ==
            kAlternateMemorySpace) {
          current_prefetches--;
        } else {
          current_evictions--;
        }
      }
      copies.max_copies = std::max(copies.max_copies, current_copies);
      copies.max_prefetches =
          std::max(copies.max_prefetches, current_prefetches);
      copies.max_prefetches = std::max(copies.max_evictions, current_evictions);
    }
    return copies;
  }

  static int64_t GetAlternateMemoryOffset(
      const PresetAssignments& preset_assignments,
      const HloInstruction* instruction, const ShapeIndex& index = {}) {
    // Returns the offset of the assignment, -1 if it's not in the alternate
    // memory.
    const HloModule* module = instruction->GetModule();
    auto status_or_alias_analysis = HloAliasAnalysis::Run(module);
    TF_CHECK_OK(status_or_alias_analysis.status());
    auto alias_analysis = std::move(status_or_alias_analysis.value());
    HloBuffer& buffer = alias_analysis->GetUniqueBufferAt(instruction, index);
    for (auto& pos_and_chunk : preset_assignments.chunks()) {
      for (auto& value : buffer.values()) {
        if (pos_and_chunk.first == value->defining_position()) {
          return pos_and_chunk.second.offset;
        }
      }
    }
    return -1;
  }

  std::unique_ptr<HloModule> CreateEvictAndPrefetchModule() {
    HloComputation::Builder builder(TestName());
    Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
    HloInstruction* p0 =
        builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
    HloInstruction* p1 =
        builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
    HloInstruction* tanh = builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
    // tanh should be placed in the alternate memory since there isn't much
    // contention in the beginning. However, tanh has another consumer at the
    // end. So it should be kicked out to default memory and prefetched back in.
    // The graph below is meant to increase the contention to force
    // eviction/prefetch behavior.
    HloInstruction* a = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, tanh));
    HloInstruction* b = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
    HloInstruction* c = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, p0, p1));
    HloInstruction* d = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
    HloInstruction* e = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, b));
    HloInstruction* f = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, c));
    HloInstruction* g = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, d));
    HloInstruction* h = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, b, c));
    HloInstruction* i = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, b, d));
    HloInstruction* j = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, c, d));
    HloInstruction* k = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, e, f));
    HloInstruction* l = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, g, h));
    HloInstruction* m = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, i, j));
    HloInstruction* n = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, k, l));
    HloInstruction* o = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, n, m));
    // tanh is being used at the root instruction, and this should be
    // prefetched.
    HloInstruction* add = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, o, tanh));

    auto module = CreateNewVerifiedModule();
    HloComputation* computation = module->AddEntryComputation(builder.Build());

    HloSchedule schedule(module.get());
    schedule.set_sequence(computation, {p0, p1, tanh, a, b, c, d, e, f, g, h, i,
                                        j, k, l, m, n, o, add});
    TF_CHECK_OK(module->set_schedule(schedule));
    return module;
  }

  CostAnalysis::Cache cache_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_SPACE_ASSIGNMENT_TEST_BASE_H_
