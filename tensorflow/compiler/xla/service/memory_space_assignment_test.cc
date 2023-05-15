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

#include "tensorflow/compiler/xla/service/memory_space_assignment.h"

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/instruction_hoister.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using memory_space_assignment::AsynchronousCopy;
using memory_space_assignment::AsynchronousCopyOrdering;
using memory_space_assignment::AsynchronousCopyResource;
using memory_space_assignment::CostAnalysisPrefetchIntervalPicker;
using memory_space_assignment::InstructionCountPrefetchIntervalPicker;
using memory_space_assignment::MemoryBoundLoopOptimizer;
using memory_space_assignment::MemoryBoundLoopOptimizerOptions;
using memory_space_assignment::MemorySpaceAssignment;
using memory_space_assignment::MemorySpaceAssignmentCostAnalysis;
using memory_space_assignment::Options;
using memory_space_assignment::PrefetchIntervalPicker;
using memory_space_assignment::PresetAssignments;

constexpr int64_t kPointerSize = 8;
constexpr float kAsyncCopyBandwidth = 100;
constexpr float kAlternateMemBandwidth = 1000;
constexpr float kBytesPerSecond = 100;
constexpr float kFlopsPerSecond = 1000;
constexpr float kTranscendentalsPerSecond = 10;

int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

int64_t SizeFunction(const BufferValue& value) {
  return ShapeSize(value.shape());
}

class MemorySpaceAssignmentTest : public HloTestBase,
                                  public ::testing::WithParamInterface<bool> {
 protected:
  // We use the following two memory space values to describe the default (slow
  // and large) and alternate (fast and small) memory spaces.
  const int64_t kDefaultMemorySpace = 0;
  const int64_t kAlternateMemorySpace = 1;

  std::unique_ptr<PresetAssignments> AssignMemorySpaceUsingCostAnalysis(
      HloModule* module,
      std::optional<Options> memory_space_assignment_options = std::nullopt) {
    HloCostAnalysis::Options cost_options{ShapeSize};
    cost_options.set_flops_per_second(kFlopsPerSecond);
    cost_options.set_bytes_per_second(kBytesPerSecond);
    cost_options.set_transcendentals_per_second(kTranscendentalsPerSecond);
    HloCostAnalysis hlo_cost_analysis(cost_options);
    for (HloComputation* computation : module->MakeNonfusionComputations()) {
      TF_CHECK_OK(computation->Accept(&hlo_cost_analysis));
    }
    auto alias_analysis = HloAliasAnalysis::Run(module).value();

    Options options;
    if (memory_space_assignment_options.has_value()) {
      options = *memory_space_assignment_options;
    } else {
      options.async_copy_bandwidth_bytes_per_second = kAsyncCopyBandwidth;
      options.alternate_mem_bandwidth_bytes_per_second = kAlternateMemBandwidth;
    }
    auto cost_analysis = MemorySpaceAssignmentCostAnalysis::Create(
                             hlo_cost_analysis, options, *module)
                             .value();
    CostAnalysisPrefetchIntervalPicker prefetch_interval_picker(
        CostAnalysisPrefetchIntervalPicker(
            *cost_analysis, /*min_overlap_to_async_copy_ratio=*/0.8,
            /*preferred_overlap_to_async_copy_ratio=*/1.5,
            /*max_overlap_to_mem_size_async_copy_ratio=*/10.0,
            /*mem_size_bytes=*/128));
    return AssignMemorySpace(
        module, /*max_outstanding_async_copies=*/-1,
        MemorySpaceAssignment::GetMemoryBoundednessBufferIntervalCompare(
            *cost_analysis, &cache_),
        &prefetch_interval_picker, memory_space_assignment_options);
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, int64_t max_outstanding_async_copies = -1,
      int64_t max_prefetch_interval = 10, int64_t min_prefetch_interval = 2,
      std::optional<Options> options = std::nullopt) {
    InstructionHoister instruction_hoister;
    TF_CHECK_OK(instruction_hoister.Run(module).status());
    InstructionCountPrefetchIntervalPicker prefetch_interval_picker(
        min_prefetch_interval, max_prefetch_interval);
    return AssignMemorySpace(module, max_outstanding_async_copies,
                             /*buffer_interval_compare=*/{},
                             &prefetch_interval_picker, options);
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, int64_t max_outstanding_async_copies,
      std::optional<MemorySpaceAssignment::BufferIntervalCompare>
          buffer_interval_compare,
      PrefetchIntervalPicker* prefetch_interval_picker,
      std::optional<Options> memory_space_assignment_options = std::nullopt) {
    auto size_fn = [](const BufferValue& buffer) {
      return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
    };

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

    Options options;
    if (memory_space_assignment_options) {
      options = *memory_space_assignment_options;
    } else {
      options.max_size_in_bytes = 128;
      options.alignment_in_bytes = 8;
      options.verify = true;
    }

    options.alternate_memory_space = kAlternateMemorySpace;
    options.buffer_interval_compare = buffer_interval_compare;
    options.prefetch_interval_picker = prefetch_interval_picker;
    options.size_fn = size_fn;
    if (options.is_allowed_in_alternate_mem_fn == nullptr) {
      options.is_allowed_in_alternate_mem_fn = is_allowed_in_alternate_mem;
    }
    options.max_outstanding_prefetches = max_outstanding_async_copies;
    options.max_outstanding_evictions = max_outstanding_async_copies;
    options.allocate_across_sequential_calls = GetParam();

    auto alias_analysis = HloAliasAnalysis::Run(module).value();
    std::unique_ptr<HloLiveRange> hlo_live_range =
        HloLiveRange::Run(module->schedule(), *alias_analysis,
                          module->entry_computation())
            .value();

    std::unique_ptr<PresetAssignments> preset_assignments =
        MemorySpaceAssignment::Run(module, *hlo_live_range, *alias_analysis,
                                   options)
            .value();
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
      const HloModule& module) {
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

  int64_t GetAlternateMemoryOffset(const PresetAssignments& preset_assignments,
                                   const HloInstruction* instruction,
                                   const ShapeIndex& index = {}) const {
    // Returns the offset of the assignment, -1 if it's not in the alternate
    // memory.
    const HloModule* module = instruction->GetModule();
    auto alias_analysis = HloAliasAnalysis::Run(module).value();
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

  MemorySpaceAssignmentCostAnalysis::Cache cache_;
};

// For testing purposes, we define a cost analysis where we can control the
// elapsed times of each HLO and asynchronous copy.
class FakeMemorySpaceAssignmentCostAnalysis
    : public MemorySpaceAssignmentCostAnalysis {
 public:
  static StatusOr<std::unique_ptr<FakeMemorySpaceAssignmentCostAnalysis>>
  Create(const HloCostAnalysis& cost_analysis, const HloModule& module,
         const Options& options) {
    TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(&module));
    TF_ASSIGN_OR_RETURN(auto hlo_live_range,
                        HloLiveRange::Run(module.schedule(), *alias_analysis,
                                          module.entry_computation()));
    auto call_graph = CallGraph::Build(&module);
    return absl::WrapUnique(new FakeMemorySpaceAssignmentCostAnalysis(
        cost_analysis, options, std::move(alias_analysis),
        std::move(hlo_live_range), std::move(call_graph)));
  }

  float GetInstructionElapsed(
      const HloInstruction& instruction) const override {
    if (get_instruction_elapsed_override_) {
      return get_instruction_elapsed_override_(instruction);
    }
    return 1.0;
  }

  float GetInstructionElapsedInAlternateMemory(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem,
      absl::Span<const ShapeIndex> outputs_in_alternate_mem) const override {
    if (get_instruction_elapsed_in_alternate_memory_override_) {
      return get_instruction_elapsed_in_alternate_memory_override_(
          instruction, operands_in_alternate_mem, outputs_in_alternate_mem);
    }
    if (!operands_in_alternate_mem.empty()) {
      return 0.5;
    } else {
      return 1.0;
    }
  }

  float GetAsyncCopyElapsed(const Shape& shape) const override {
    if (get_async_copy_elapsed_override_) {
      return get_async_copy_elapsed_override_(shape);
    }
    return 3.0;
  }

  // The following methods can be used to override what the above API calls
  // return.
  void SetOverrideForGetInstructionElapsed(
      std::function<float(const HloInstruction&)> function) {
    get_instruction_elapsed_override_ = function;
  }
  void SetOverrideForGetInstructionElapsedInAlternateMemory(
      std::function<float(const HloInstruction&,
                          absl::Span<const std::pair<int64_t, ShapeIndex>>,
                          absl::Span<const ShapeIndex>)>
          function) {
    get_instruction_elapsed_in_alternate_memory_override_ = function;
  }
  void SetOverrideForGetAsyncCopyElapsed(
      std::function<float(const Shape&)> function) {
    get_async_copy_elapsed_override_ = function;
  }

 protected:
  FakeMemorySpaceAssignmentCostAnalysis(
      const HloCostAnalysis& cost_analysis, const Options& options,
      std::unique_ptr<HloAliasAnalysis> alias_analysis,
      std::unique_ptr<HloLiveRange> hlo_live_range,
      std::unique_ptr<CallGraph> call_graph)
      : MemorySpaceAssignmentCostAnalysis(
            cost_analysis, options, std::move(alias_analysis),
            std::move(hlo_live_range), std::move(call_graph)) {}

 private:
  std::function<float(const HloInstruction&)>
      get_instruction_elapsed_override_ = nullptr;
  std::function<float(const HloInstruction&,
                      absl::Span<const std::pair<int64_t, ShapeIndex>>,
                      absl::Span<const ShapeIndex>)>
      get_instruction_elapsed_in_alternate_memory_override_ = nullptr;
  std::function<float(const Shape&)> get_async_copy_elapsed_override_ = nullptr;
};

TEST_P(MemorySpaceAssignmentTest, ParameterOnly) {
  // A module consisting of a single parameter. Inputs/outputs are currently
  // excluded from memory space assignment.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
}

TEST_P(MemorySpaceAssignmentTest, Simple) {
  // A simple module with a few simple instructions. Expect this to be
  // transformed with CopyStart and CopyDone instructions inserted after inputs
  // and before outputs.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p1));
  HloInstruction* sub = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, add, sub));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, add, sub, mul});
  TF_CHECK_OK(module->set_schedule(schedule));

  auto preset_assignments = AssignMemorySpace(module.get());

  // Inputs and outputs are currently placed in the default memory. Everything
  // else should be in the alternate memory.
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  EXPECT_THAT(mul, op::ShapeWithLayout(shape));
  EXPECT_THAT(add, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(sub, op::ShapeWithLayout(shape_in_alternate_mem));

  // Make sure the preset assignments is sane.
  EXPECT_EQ(preset_assignments->chunks().size(), 3);
  EXPECT_EQ(preset_assignments->assignment_informations().size(), 1);
  // Ensure the offset assigned to add and sub are different.
  EXPECT_NE(preset_assignments->chunks()[0].second.offset,
            preset_assignments->chunks()[1].second.offset);
}

TEST_P(MemorySpaceAssignmentTest, NegateChain) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[2], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_P(MemorySpaceAssignmentTest, FilterUpdatePreferredPrefetchTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  auto config = "op_size_gte:24:op_size_lte:24:prefetch_eagerness:0.5";
  TF_ASSERT_OK_AND_ASSIGN(
      options.filter_update_preferred_prefetches,
      memory_space_assignment::FilterUpdatePreferredPrefetch::
          ParseFilterUpdatePreferredPrefetches(config));
  AssignMemorySpace(module.get(), -1, 10, 2, options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[6], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_P(MemorySpaceAssignmentTest, FilterUpdateConfigExactMatchBeforeTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  auto config =
      "instruction_name_exact:add:op_number_exact:1:put_before_instruction:"
      "negate.3";
  TF_ASSERT_OK_AND_ASSIGN(
      options.filter_update_preferred_prefetches,
      memory_space_assignment::FilterUpdatePreferredPrefetch::
          ParseFilterUpdatePreferredPrefetches(config));
  AssignMemorySpace(module.get(), -1, 10, 2, options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[5], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_P(MemorySpaceAssignmentTest, FilterUpdateConfigExactMatchAfterTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  auto config =
      "instruction_name_exact:add:op_number_exact:1:put_after_instruction:"
      "negate.1";
  TF_ASSERT_OK_AND_ASSIGN(
      options.filter_update_preferred_prefetches,
      memory_space_assignment::FilterUpdatePreferredPrefetch::
          ParseFilterUpdatePreferredPrefetches(config));
  AssignMemorySpace(module.get(), -1, 10, 2, options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[4], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_P(MemorySpaceAssignmentTest, FilterUpdateConfigExactMatchTooLateTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  auto config =
      "instruction_name_exact:add:op_number_exact:1:put_after_instruction:"
      "negate.5";
  TF_ASSERT_OK_AND_ASSIGN(
      options.filter_update_preferred_prefetches,
      memory_space_assignment::FilterUpdatePreferredPrefetch::
          ParseFilterUpdatePreferredPrefetches(config));
  AssignMemorySpace(module.get(), -1, 10, 2, options);

  // Ensure the Async copy is not scheduled.
  EXPECT_THAT(add, op::Add(op::Negate(), op::Parameter(1)));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, FilterUpdateConfigPrecedenceTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  auto config =
      "op_size_gte:24:op_size_lte:24:prefetch_eagerness:0.5;instruction_"
      "name_exact:add:op_number_exact:1:put_after_instruction:negate.1";
  TF_ASSERT_OK_AND_ASSIGN(
      options.filter_update_preferred_prefetches,
      memory_space_assignment::FilterUpdatePreferredPrefetch::
          ParseFilterUpdatePreferredPrefetches(config));
  AssignMemorySpace(module.get(), -1, 10, 2, options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[6], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_P(MemorySpaceAssignmentTest, FilterUpdateConfigExactMatchPrecedenceTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  auto config =
      "instruction_name_exact:add:op_number_exact:1:put_after_instruction:"
      "negate.1;op_size_gte:24:op_size_lte:24:prefetch_eagerness:0.5";
  TF_ASSERT_OK_AND_ASSIGN(
      options.filter_update_preferred_prefetches,
      memory_space_assignment::FilterUpdatePreferredPrefetch::
          ParseFilterUpdatePreferredPrefetches(config));
  AssignMemorySpace(module.get(), -1, 10, 2, options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[4], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_P(MemorySpaceAssignmentTest, FilterUpdatePreferredPrefetchNoMatchTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  auto config = "op_size_gte:25:op_size_lte:24:prefetch_eagerness:0.5";
  TF_ASSERT_OK_AND_ASSIGN(
      options.filter_update_preferred_prefetches,
      memory_space_assignment::FilterUpdatePreferredPrefetch::
          ParseFilterUpdatePreferredPrefetches(config));
  AssignMemorySpace(module.get(), -1, 10, 2, options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3},
                                          /*minor_to_major=*/{1, 0},
                                          /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[2], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetch) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get());

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Add(op::Add(),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::AsyncCopy(kDefaultMemorySpace,
                                          kAlternateMemorySpace, op::Tanh()))));
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies0) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/0);

  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_prefetches, 0);
  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_evictions, 0);
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies1) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/1);

  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_prefetches, 1);
  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_evictions, 1);
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies2) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/2);

  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_prefetches, 2);
  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_evictions, 2);
}

// TODO(berkin): This test is broken with some prefetch timing improvements.
TEST_P(MemorySpaceAssignmentTest,
       DISABLED_DontEvictWhenThereIsDefaultMemAllocation) {
  // This test is the same as EvictAndPrefetchLimitAsyncCopies1, except we check
  // that there is no eviction if not necessary (due to an existing allocation
  // in default memory).
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  // tanh should be placed in the alternate memory since there isn't much
  // contention in the beginning. However, tanh has another consumer at the end.
  // So it should be kicked out to default memory and prefetched back in.  The
  // graph below is meant to increase the contention to force eviction/prefetch
  // behavior.
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

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/1);

  // We expect the second argument to multiply is prefetched c.
  EXPECT_THAT(f, op::Multiply(op::Add(), op::CopyDone()));
  // We make sure that the second argument to this multiply is not evicted
  // CopyDone but is the original c.
  EXPECT_THAT(h, op::Multiply(op::Subtract(), op::Multiply()));
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetchAndPrefetch) {
  // Test for a memory corruption bug involving evict/prefetch/prefetch pattern,
  // where the last prefetch copied from the original buffer in alternate buffer
  // instead of evicted buffer.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
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
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, o, tanh));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, add0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* negate8 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate7));
  HloInstruction* negate9 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate8));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate9, tanh));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation,
      {p0,      p1,      tanh,    a,       b,       c,       d,       e,
       f,       g,       h,       i,       j,       k,       l,       m,
       n,       o,       add0,    negate0, negate1, negate2, negate3, negate4,
       negate5, negate6, negate7, negate8, negate9, add1});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  // Check that both prefetches (add0 and add1) prefetch from the eviction
  // instead of tanh, which will be placed in the alternate memory directly.
  EXPECT_THAT(
      add0,
      op::Add(op::Add(),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::AsyncCopy(kDefaultMemorySpace,
                                          kAlternateMemorySpace, op::Tanh()))));
  EXPECT_THAT(
      add1,
      op::Add(op::Negate(),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::AsyncCopy(kDefaultMemorySpace,
                                          kAlternateMemorySpace, op::Tanh()))));
}

TEST_P(MemorySpaceAssignmentTest, While) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  // Free cond_param[] (16 bytes), Alloc PRED[] (1 byte)
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_data_increment =
      body_builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}})));
  HloInstruction* body_data_mul =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, body_data, body_data));
  HloInstruction* body_data_add =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data, body_data_increment));
  HloInstruction* body_data_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data_add, body_data_mul));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data_next, body_iter_next}));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_iter"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_data"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({data, iter}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(body_computation,
                        {body_param, body_iter, body_data, body_iter_increment,
                         body_iter_next, body_data_increment, body_data_mul,
                         body_data_add, body_data_next, body_out});
  schedule.set_sequence(entry_computation, {iter, data, tuple, while_op});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  // Ensure the tuple value and buffers used in the while instruction are
  // exempted from using the alternate memory when allocating across sequential
  // calls is disabled. However, body_data_mul is independent and can be safely
  // be placed in the alternate memory.
  const bool allocate_across_sequential_calls = GetParam();
  if (!allocate_across_sequential_calls) {
    EXPECT_THAT(tuple, op::ShapeWithLayout(tuple_shape));
    EXPECT_THAT(data, op::ShapeWithLayout(shape));
    EXPECT_THAT(iter, op::ShapeWithLayout(scalar_shape));
    EXPECT_THAT(body_data, op::ShapeWithLayout(shape));
    EXPECT_THAT(body_iter, op::ShapeWithLayout(scalar_shape));
    EXPECT_THAT(cond_iter, op::ShapeWithLayout(scalar_shape));
  }
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(body_data_mul, op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, Tuple) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape inner_tuple_shape = ShapeUtil::MakeTupleShape({shape});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({shape, shape, inner_tuple_shape});
  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));
  HloInstruction* p2 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(inner_tuple_shape, p, 2));
  HloInstruction* p2_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p2, 0));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, add, p2_0));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p, p0, negate0, negate1, negate2, negate3, negate4, negate5,
                    negate6, p1, add, p2, p2_0, mul});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_THAT(
      mul,
      op::Multiply(op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::GetTupleElement())),
                   op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                 op::GetTupleElement(op::GetTupleElement()))));
}

TEST_P(MemorySpaceAssignmentTest, Bitcast) {
  // Bitcasts can cause the position in the alternate memory to appear multiple
  // times in the preset assignments. This test ensure the preset assignments
  // refer to unique positions.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(param_shape, negate));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(param_shape, HloOpcode::kAdd, bitcast, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate, bitcast, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  bitcast = add->mutable_operand(0);
  EXPECT_EQ(bitcast->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(bitcast->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, Bitcast2) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, bitcast, negate4));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, bitcast, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_EQ(add->operand(0)->shape().layout().memory_space(),
            kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, Bitcast3) {
  HloComputation::Builder builder(TestName());
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(F32, {3, 2});
  Shape shape3 = ShapeUtil::MakeShape(F32, {1, 6});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape1, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate3));
  HloInstruction* bitcast1 =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape1, p1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, bitcast1, negate4));
  HloInstruction* bitcast2 =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape3, p1));
  HloInstruction* bitcast3 =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape2, bitcast2));
  HloInstruction* bitcast4 =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape2, add));
  HloInstruction* mul = builder.AddInstruction(HloInstruction::CreateBinary(
      shape2, HloOpcode::kMultiply, bitcast3, bitcast4));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, negate0, negate1, negate2, negate3, negate4,
                         bitcast1, add, bitcast2, bitcast3, bitcast4, mul});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  // We expect one bitcast on the LHS of multiply since bitcast(bitcast(foo)) is
  // converted to bitcast(foo).
  EXPECT_THAT(
      mul,
      op::Multiply(
          op::Bitcast(op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                    op::Parameter(1))),
          op::Bitcast(op::Add(
              op::Bitcast(op::AsyncCopy(kAlternateMemorySpace,
                                        kDefaultMemorySpace, op::Parameter(1))),
              op::Negate()))));
  EXPECT_EQ(add->operand(0)->shape().layout().memory_space(),
            kAlternateMemorySpace);
  EXPECT_EQ(add->shape().layout().memory_space(), kAlternateMemorySpace);
  // bitcast2 will no longer have a consumer and should get DCE'd, so we don't
  // care about its memory space.
  EXPECT_EQ(mul->operand(0)->shape().layout().memory_space(),
            kAlternateMemorySpace);
  EXPECT_EQ(mul->operand(1)->shape().layout().memory_space(),
            kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, BitcastTuple) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder fusion_builder("fusion");
  HloInstruction* fusion_param = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion_element0 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 0));
  HloInstruction* fusion_element1 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 1));
  fusion_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion_element0, fusion_element1));
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p1));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({bitcast, p0}));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kCustom, {tuple}, fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, negate0, negate1, negate2, negate3, negate4,
                         bitcast, tuple, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, BitcastGetTupleElementTuple) {
  // This test pattern was encountered in
  // //third_party/tensorflow/compiler/xla/tests:slice_test and was causing a
  // breakage when there is a GetTupleElement(Tuple(Bitcast())) pattern. Also
  // added a GetTupleElement(GetTupleElement(Tuple(Tuple(Bitcast())))) pattern.
  absl::string_view hlo_string = R"(
  HloModule DoIt_S64_10_0_5_1.3, is_scheduled=true

  ENTRY %DoIt_S64_10_0_5_1.3 (p0.1: (u32[10], u32[10])) -> (u32[5], u32[5]) {
    %p0.1 = (u32[10]{0:T(128)}, u32[10]{0:T(128)}) parameter(0)
    %get-tuple-element.1 = u32[10]{0:T(128)} get-tuple-element((u32[10]{0:T(128)}, u32[10]{0:T(128)}) %p0.1), index=1
    %bitcast.1 = u32[5]{0:T(128)} bitcast(u32[10]{0:T(128)} %get-tuple-element.1)
    %get-tuple-element = u32[10]{0:T(128)} get-tuple-element((u32[10]{0:T(128)}, u32[10]{0:T(128)}) %p0.1), index=0
    %bitcast = u32[5]{0:T(128)} bitcast(u32[10]{0:T(128)} %get-tuple-element)
    %tuple.1 = (u32[5]{0:T(128)}, u32[5]{0:T(128)}) tuple(u32[5]{0:T(128)} %bitcast, u32[5]{0:T(128)} %bitcast.1)
    %tuple.3 = ((u32[5]{0:T(128)}, u32[5]{0:T(128)}), (u32[5]{0:T(128)}, u32[5]{0:T(128)})) tuple(%tuple.1, %tuple.1)
    %get-tuple-element.4 = u32[5]{0:T(128)} get-tuple-element((u32[5]{0:T(128)}, u32[5]{0:T(128)}) %tuple.1), index=0
    %get-tuple-element.5 = (u32[5]{0:T(128)}, u32[5]{0:T(128)}) get-tuple-element(%tuple.3), index=0
    %get-tuple-element.6 = u32[5]{0:T(128)} get-tuple-element((u32[5]{0:T(128)}, u32[5]{0:T(128)}) %get-tuple-element.5), index=1
    %copy.2 = u32[5]{0:T(128)} copy(u32[5]{0:T(128)} %get-tuple-element.4)
    %copy.3 = u32[5]{0:T(128)} copy(u32[5]{0:T(128)} %get-tuple-element.6)
    ROOT %tuple.2 = (u32[5]{0:T(128)}, u32[5]{0:T(128)}) tuple(u32[5]{0:T(128)} %copy.2, u32[5]{0:T(128)} %copy.3)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, GetSimplifiedOperandBug) {
  // Test case for a bug finding Bitcasts in GTE(Tuple(...)) pattern.
  absl::string_view hlo_string = R"(
  HloModule sort.16, is_scheduled=true

  ENTRY %sort.16 (param.0.1: s32[1], param.1.2: f32[1], param.2.3: u32[1], param.3.4: s32[1]) -> (s32[1], f32[1], u32[1], s32[1]) {
    %param.3.4 = s32[1]{0:T(128)} parameter(3)
    %param.2.3 = u32[1]{0:T(128)} parameter(2)
    %param.1.2 = f32[1]{0:T(128)} parameter(1)
    %param.0.1 = s32[1]{0:T(128)} parameter(0)
    %tuple.1 = (s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %param.0.1, f32[1]{0:T(128)} %param.1.2, u32[1]{0:T(128)} %param.2.3, s32[1]{0:T(128)} %param.3.4)
    %get-tuple-element.4 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=0
    %get-tuple-element.5 = f32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=1
    %get-tuple-element.6 = u32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=2
    %get-tuple-element.7 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=3
    %copy.4 = s32[1]{0:T(128)} copy(s32[1]{0:T(128)} %get-tuple-element.4)
    %copy.5 = f32[1]{0:T(128)} copy(f32[1]{0:T(128)} %get-tuple-element.5)
    %copy.6 = u32[1]{0:T(128)} copy(u32[1]{0:T(128)} %get-tuple-element.6)
    %copy.7 = s32[1]{0:T(128)} copy(s32[1]{0:T(128)} %get-tuple-element.7)
    ROOT %tuple.2 = (s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %copy.4, f32[1]{0:T(128)} %copy.5, u32[1]{0:T(128)} %copy.6, s32[1]{0:T(128)} %copy.7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, BitcastMultiUse) {
  // When there is a pattern where a bitcast has multiple uses (negate0 and add)
  // and one is in the default memory and the other is in alternate memory, they
  // both need their own bitcast.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p1"));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, bitcast));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, bitcast, negate4));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, bitcast, negate0, negate1, negate2,
                                      negate3, negate4, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0->operand(0), op::ShapeWithLayout(shape));
  EXPECT_THAT(add->operand(0), op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, BitcastMultiUseTuple) {
  // Same as BitcastMultUse but the second use is a tuple.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder fusion_builder("fusion");
  HloInstruction* fusion_param = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion_element0 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 0));
  HloInstruction* fusion_element1 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 1));
  fusion_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion_element0, fusion_element1));
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p1"));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, bitcast));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({bitcast, negate4}));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kCustom, {tuple}, fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, bitcast, negate0, negate1, negate2,
                                      negate3, negate4, tuple, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0->operand(0), op::ShapeWithLayout(shape));
  EXPECT_THAT(fusion->operand(0)->operand(0),
              op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, BitcastScheduleBug) {
  // Bitcasts can force asynchronous copies to be scheduled too early, possibly
  // leading to memory corruption.
  //  Bug:
  //    p0------------------>neg-->neg-->neg ... -->neg-->neg-->neg->add
  //                                                                 /
  //    p1->cs->cd->bitcast-----------------------------------------+
  //
  //  Expected:
  //    p0-->neg-->neg-->neg ... -->neg-->neg-->neg------------->add
  //                                                             /
  //    p1--------------------->cs----------------->cd->bitcast-+
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p1));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* negate8 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate7));
  HloInstruction* negate9 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate8));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, bitcast, negate9));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p0, p1, bitcast, negate0, negate1, negate2, negate3,
                    negate4, negate5, negate6, negate7, negate8, negate9, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/4);

  EXPECT_EQ(add->operand(0)->shape().layout().memory_space(),
            kAlternateMemorySpace);
  const auto& instructions =
      module->schedule().sequence(module->entry_computation()).instructions();
  for (int i = 0; i < instructions.size(); ++i) {
    // Expect that there is a negate before and after the CopyStart and there is
    // a negate before CopyDone.
    if (instructions.at(i)->opcode() == HloOpcode::kCopyStart) {
      EXPECT_EQ(instructions.at(i - 1)->opcode(), HloOpcode::kNegate);
      EXPECT_EQ(instructions.at(i + 1)->opcode(), HloOpcode::kNegate);
    } else if (instructions.at(i)->opcode() == HloOpcode::kCopyDone) {
      EXPECT_EQ(instructions.at(i - 1)->opcode(), HloOpcode::kNegate);
    }
  }
}

TEST_P(MemorySpaceAssignmentTest, AddDependency) {
  // Make sure add-dependency is not optimized away.
  absl::string_view hlo_string = R"(
  HloModule AddDependency, is_scheduled=true

  ENTRY %AddDependency (p: f32[3]) -> f32[3] {
    %p = f32[3]{0} parameter(0)
    %neg0 = f32[3]{0} negate(f32[3]{0} %p)
    %neg1 = f32[3]{0} negate(f32[3]{0} %neg0)
    %neg2 = f32[3]{0} negate(f32[3]{0} %neg1)
    %neg3 = f32[3]{0} negate(f32[3]{0} %neg2)
    %neg4 = f32[3]{0} negate(f32[3]{0} %neg3)
    %neg5 = f32[3]{0} negate(f32[3]{0} %neg4)
    %neg6 = f32[3]{0} negate(f32[3]{0} %neg5)
    %token0 = token[] after-all()
    %add_dep = f32[3]{0} add-dependency(f32[3]{0} %p, token[] %token0)
    ROOT %add = f32[3]{0} add(f32[3]{0} %add_dep, f32[3]{0} %neg6)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Add(op::AddDependency(), op::Negate()));
}

TEST_P(MemorySpaceAssignmentTest, WhileAllocationBug) {
  // This test is carefully crafted to include two multiply ops sized [4,3] in a
  // while body. For testing purposes, we have provided a BufferIntervalCompare
  // such that first multiply, then tanh, then other HloValues will be
  // allocated. The memory is sized just enough to fit two [4,3] buffers.
  // Because the multiplies in the while body are going to be allocated in the
  // alternate memory first, the tanh that is fed inside the while loop should
  // not be placed in the alternate memory. Otherwise, we will corrupt memory.
  absl::string_view hlo_string = R"(
  HloModule WhileAllocationBug, is_scheduled=true

  %WhileBody (body_param: (f32[4,3], f32[])) -> (f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[]) %body_param), index=1
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[]) %body_param), index=0
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %get-tuple-element.2)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[] %add)
  }

  %WhileCond (cond_param: (f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[]) %cond_param), index=1
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_iter: f32[4,3], param_data: f32[], p2: f32[4,3]) -> f32[4,3] {
    %param_data = f32[] parameter(1)
    %param_iter = f32[4,3]{1,0} parameter(0)
    %p2 = f32[4,3]{1,0} parameter(2)
    %tanh = f32[4,3]{1,0} tanh(f32[4,3]{1,0} %param_iter)
    %neg0 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %p2)
    %neg1 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg0)
    %neg2 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg1)
    %neg3 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg2)
    %neg4 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg3)
    %neg5 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg4)
    %neg6 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg5)
    %add.4 = f32[4,3]{1,0} add(f32[4,3]{1,0} %neg6, f32[4,3]{1,0} %tanh)
    %tuple.1 = (f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %tanh, f32[] %param_data)
    %while = (f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[]) %while), index=0
    ROOT %add.3 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.3, f32[4,3]{1,0} %add.4)
  }
  )";

  MemorySpaceAssignment::BufferIntervalCompare buffer_interval_compare =
      [](const MemorySpaceAssignment::BufferInterval& a,
         const MemorySpaceAssignment::BufferInterval& b) {
        bool a_is_mul =
            a.buffer->defining_instruction()->opcode() == HloOpcode::kMultiply;
        bool b_is_mul =
            b.buffer->defining_instruction()->opcode() == HloOpcode::kMultiply;
        if (a_is_mul && !b_is_mul) {
          return true;
        }
        if (!a_is_mul && b_is_mul) {
          return false;
        }
        bool a_is_tanh =
            a.buffer->defining_instruction()->opcode() == HloOpcode::kTanh;
        bool b_is_tanh =
            b.buffer->defining_instruction()->opcode() == HloOpcode::kTanh;
        if (a_is_tanh && !b_is_tanh) {
          return true;
        }
        if (!a_is_tanh && b_is_tanh) {
          return false;
        }
        return a.buffer->id() < b.buffer->id();
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    buffer_interval_compare, &prefetch_interval_picker);

  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      const Shape& while_subshape =
          ShapeUtil::GetSubshape(instruction->shape(), {0});
      // We expect shape {0} to either be in default memory for the entire while
      // loop or there has to be an eviction within the while loop.
      if (while_subshape.layout().memory_space() == kAlternateMemorySpace) {
        const HloInstruction* body_param =
            instruction->while_body()->parameter_instruction(0);
        const HloInstruction* gte = nullptr;
        for (const HloInstruction* user : body_param->users()) {
          if (user->opcode() == HloOpcode::kGetTupleElement &&
              user->tuple_index() == 0) {
            gte = user;
            break;
          }
        }
        EXPECT_NE(gte, nullptr);
        const HloInstruction* copy_start = nullptr;
        for (const HloInstruction* user : gte->users()) {
          if (user->opcode() == HloOpcode::kCopyStart) {
            copy_start = user;
            break;
          }
        }
        EXPECT_NE(copy_start, nullptr);
        const Shape& copy_start_subshape =
            ShapeUtil::GetSubshape(copy_start->shape(), {0});

        EXPECT_NE(copy_start_subshape.layout().memory_space(),
                  kAlternateMemorySpace);
      }
    }
  }
}

TEST_P(MemorySpaceAssignmentTest, ConsecutiveWhileLoops) {
  absl::string_view hlo_string = R"(
  HloModule WhileAllocationBug, is_scheduled=true

  %WhileBody (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %get-tuple-element.3)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  %WhileBody2 (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %get-tuple-element.3)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond2 (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_data: f32[4,3], param_iter: f32[], p2: f32[4,3]) -> f32[4,3] {
    %param_iter = f32[] parameter(1)
    %param_data = f32[4,3]{1,0} parameter(0)
    %p2 = f32[4,3]{1,0} parameter(2)
    %neg0 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %p2)
    %neg1 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg0)
    %neg2 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg1)
    %neg3 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg2)
    %neg4 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg3)
    %neg5 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg4)
    %neg6 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg5)
    %add.4 = f32[4,3]{1,0} add(f32[4,3]{1,0} %neg6, f32[4,3]{1,0} %p2)
    %tuple.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.4, f32[4,3]{1,0} param_data, f32[] %param_iter)
    %while = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.4 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=0
    %add.3 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.4, f32[4,3]{1,0} %add.4)
    %get-tuple-element.5 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=1
    %tuple.2 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.3, f32[4,3]{1,0} get-tuple-element.5, f32[] %param_iter)
    %while.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.2), condition=%WhileCond2, body=%WhileBody2
    %get-tuple-element.6 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while.1), index=0
    ROOT %add.5 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.6, f32[4,3]{1,0} %add.3)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, WhileLiveRangeBug) {
  // Tests against while live ranges being incorrect and the verifier
  // complaining about a conflict.
  absl::string_view hlo_string = R"(
  HloModule WhileAllocationBug, is_scheduled=true

  %WhileBody (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %neg10 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %get-tuple-element.2)
    %neg11 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg10)
    %neg12 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg11)
    %neg13 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg12)
    %neg14 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg13)
    %neg15 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg14)
    %neg16 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg15)
    %neg17 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg16)
    %neg18 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg17)
    %neg19 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg18)
    %neg20 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg19)
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %neg20, f32[4,3]{1,0} %neg20)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} get-tuple-element.3, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_data: f32[4,3], param_iter: f32[], p2: f32[4,3]) -> f32[4,3] {
    %param_iter = f32[] parameter(1)
    %param_data = f32[4,3]{1,0} parameter(0)
    %p2 = f32[4,3]{1,0} parameter(2)
    %neg0 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %p2)
    %neg1 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg0)
    %neg2 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg1)
    %neg3 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg2)
    %neg4 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg3)
    %neg5 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg4)
    %neg6 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg5)
    %add.4 = f32[4,3]{1,0} add(f32[4,3]{1,0} %neg6, f32[4,3]{1,0} %p2)
    %tuple.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.4, f32[4,3]{1,0} param_data, f32[] %param_iter)
    %while = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.4 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=0
    %get-tuple-element.5 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=1
    %add.3 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.4, f32[4,3]{1,0} %add.4)
    ROOT %add.5 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.5, f32[4,3]{1,0} %add.3)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, ConsecutiveWhileLoopsOneBuffer) {
  // Tests against a bug when there are consecutive while loops with one buffer
  // (the value doesn't change in the buffer), the parameter can be colored in
  // the alternate memory space.
  absl::string_view hlo_string = R"(
  HloModule WhileAllocationBug, is_scheduled=true

  %WhileBody (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %neg10 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %get-tuple-element.2)
    %neg11 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg10)
    %neg12 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg11)
    %neg13 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg12)
    %neg14 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg13)
    %neg15 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg14)
    %neg16 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg15)
    %neg17 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg16)
    %neg18 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg17)
    %neg19 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg18)
    %neg20 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg19)
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %neg20, f32[4,3]{1,0} %neg20)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} get-tuple-element.3, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  %WhileBody2 (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %neg10 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %get-tuple-element.2)
    %neg11 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg10)
    %neg12 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg11)
    %neg13 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg12)
    %neg14 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg13)
    %neg15 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg14)
    %neg16 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg15)
    %neg17 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg16)
    %neg18 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg17)
    %neg19 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg18)
    %neg20 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg19)
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %neg20, f32[4,3]{1,0} %neg20)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} get-tuple-element.3, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond2 (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_data: f32[4,3], param_iter: f32[], p2: f32[4,3]) -> f32[4,3] {
    %param_iter = f32[] parameter(1)
    %param_data = f32[4,3]{1,0} parameter(0)
    %p2 = f32[4,3]{1,0} parameter(2)
    %neg0 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %p2)
    %neg1 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg0)
    %neg2 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg1)
    %neg3 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg2)
    %neg4 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg3)
    %neg5 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg4)
    %neg6 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg5)
    %add.4 = f32[4,3]{1,0} add(f32[4,3]{1,0} %neg6, f32[4,3]{1,0} %p2)
    %tuple.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.4, f32[4,3]{1,0} param_data, f32[] %param_iter)
    %while = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.4 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=0
    %add.3 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.4, f32[4,3]{1,0} %add.4)
    %tuple.2 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.3, f32[4,3]{1,0} param_data, f32[] %param_iter)
    %while.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.2), condition=%WhileCond2, body=%WhileBody2
    %get-tuple-element.5 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while.1), index=0
    %get-tuple-element.6 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while.1), index=1
    ROOT %add.5 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.5, f32[4,3]{1,0} %get-tuple-element.6)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, WhileCondAliasBug) {
  // While loop is the root of the entry computation. We should ensure the
  // output of the entry computation remains to be in default memory space.
  // Test from //third_party/tensorflow/compiler/xla/tests:while_test
  // WhileTest.WhileWithPrngScalarResult.
  absl::string_view hlo_string = R"(
  HloModule WhileWithPrngScalarResult.18, is_scheduled=true

  %fused_computation (param_0.1: s32[6], param_1.3: s32[1], param_2.3: s32[5]) -> s32[6] {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  %body.3 (prev.4: s32[6]) -> s32[6] {
    %constant.7 = s32[]{:T(128)} constant(100)
    %constant.6 = s32[]{:T(128)} constant(0)
    %constant.5 = s32[1]{0:T(128)} constant({1})
    %prev.4 = s32[6]{0:T(128)} parameter(0)
    %rng.8 = s32[5]{0:T(128)} rng(s32[]{:T(128)} %constant.6, s32[]{:T(128)} %constant.7), distribution=rng_uniform
    %neg = s32[1]{0:T(128)} negate(s32[1]{0:T(128)} %constant.5)
    ROOT %fusion = s32[6]{0:T(128)} fusion(s32[6]{0:T(128)} %prev.4, s32[1]{0:T(128)} %neg, s32[5]{0:T(128)} %rng.8), kind=kLoop, calls=%fused_computation
  }

  %WhileWithPrngScalarResult.11 (prev.12: s32[6]) -> pred[] {
    %constant.15 = s32[]{:T(128)} constant(1)
    %prev.12 = s32[6]{0:T(128)} parameter(0)
    %bitcast.1 = s32[1]{0:T(128)} bitcast(s32[6]{0:T(128)} %prev.12)
    %bitcast = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %bitcast.1)
    ROOT %compare.16 = pred[]{:T(128)} compare(s32[]{:T(128)} %constant.15, s32[]{:T(128)} %bitcast), direction=GT
  }

  ENTRY %WhileWithPrngScalarResult.18 () -> s32[6] {
    %constant.1 = s32[]{:T(128)} constant(0)
    %broadcast.2 = s32[6]{0:T(128)} broadcast(s32[]{:T(128)} %constant.1), dimensions={}
    ROOT %while.17 = s32[6]{0:T(128)} while(s32[6]{0:T(128)} %broadcast.2), condition=%WhileWithPrngScalarResult.11, body=%body.3
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, WhileInPlaceBuffer) {
  // Ensure that a dynamic update slice within a while loop is able to get an
  // alternate memory allocation.
  absl::string_view hlo_string = R"(
  HloModule Module, is_scheduled=true

  fused_computation {
    param0 = f32[2,3] parameter(0)
    constant.1 = f32[] constant(0)
    broadcast = f32[2,1] broadcast(constant.1), dimensions={}
    constant.3 = s32[] constant(0)
    ROOT dynamic-update-slice.5 = f32[2,3] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
  }

  %WhileBody (body_param: (f32[2,3], f32[2,3], f32[])) -> (f32[2,3], f32[2,3], f32[]) {
    %body_param = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[2,3]{1,0} get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[2,3]{1,0} get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %body_param), index=1
    %fusion = f32[2,3]{1,0} fusion(get-tuple-element.3), kind=kLoop, calls=fused_computation
    %multiply = f32[2,3]{1,0} multiply(f32[2,3]{1,0} %get-tuple-element.2, f32[2,3]{1,0} %fusion)
    ROOT %tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) tuple(f32[2,3]{1,0} %multiply, f32[2,3]{1,0} %fusion, f32[] %get-tuple-element.1)
  }

  %WhileCond (cond_param: (f32[2,3], f32[2,3], f32[])) -> pred[] {
    %cond_param = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_data: f32[2,3], param_iter: f32[], p2: f32[2,3]) -> f32[2,3] {
    %param_iter = f32[] parameter(1)
    %param_data = f32[2,3]{1,0} parameter(0)
    %p2 = f32[2,3]{1,0} parameter(2)
    %copy1 = f32[2,3]{1,0} copy(param_data)
    %copy2 = f32[2,3]{1,0} copy(p2)
    %tuple.1 = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) tuple(f32[2,3]{1,0} copy1, f32[2,3]{1,0} copy2, f32[] %param_iter)
    %while = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) while((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.4 = f32[2,3]{1,0} get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %while), index=0
    ROOT %copy3 = f32[2,3]{1,0} copy(get-tuple-element.4)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  const HloInstruction* while_op =
      module->entry_computation()->GetInstructionWithName("while");
  if (GetParam()) {
    EXPECT_EQ(
        ShapeUtil::GetSubshape(while_op->shape(), {1}).layout().memory_space(),
        kAlternateMemorySpace);
  }
}

TEST_P(MemorySpaceAssignmentTest, WhileSharedBufferVerificationBug) {
  // Tests a spurious verification failure when a while has the same value
  // passed in twice (copy0) and that value is evicted within the while loop.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=3
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = f32[3]{0} get-tuple-element(p0), index=2
    gte3 = pred[] get-tuple-element(p0), index=3
    add = f32[3]{0} add(gte0, gte0)
    negate0 = f32[3]{0} negate(add)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    negate15 = f32[3]{0} negate(negate14)
    negate16 = f32[3]{0} negate(negate15)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, gte0, negate16, gte3)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(copy0, copy0, copy1, p1)
    while = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    ROOT gte = f32[3]{0} get-tuple-element(while), index=2
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, b228599972) {
  absl::string_view hlo_string = R"(
HloModule entry, is_scheduled=true

fused_computation {
  %p0 = f32[2,3]{1,0} parameter(0)
  %result0 = f32[2,3]{1,0} copy(%p0)
  %result1 = f32[2,3]{1,0} copy(%p0)
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(%result0, %result1)
}

ENTRY entry {
  %p0 = f32[2,3]{1,0} parameter(0)
  %p1 = f32[2,3]{1,0} parameter(1)
  %unused = (f32[2,3]{1,0}, f32[2,3]{1,0}) fusion(%p0), kind=kLoop, calls=%fused_computation
  %unused.0 = f32[2,3]{1,0} get-tuple-element(%unused), index=0
  %unused.1 = f32[2,3]{1,0} get-tuple-element(%unused), index=1
  %negate.0 = f32[2,3]{1,0} negate(f32[2,3]{1,0} %unused.0)
  %negate.1 = f32[2,3]{1,0} negate(f32[2,3]{1,0} %unused.1)

  ROOT %result = f32[2,3]{1,0} negate(%p1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, b172243149) {
  // Tests for the failure in b/172243149, where if we skip processing
  // non-copy allocations that are in default memory can actually cause
  // failures. In this case, the problem tensor is copy0, where it is fed to
  // both negate, while, and add0. The copy0->negate dependency can be allocated
  // in the alternate memory. Then the algorithm attempts to place the
  // copy0->while edge in the alternate memory, but since this value isn't used
  // in the while loop, it won't get an alternate memory allocation. Finally for
  // the copy0->add0 edge, the algorithm will actually replace it with
  // while{0}->add0, since this is equivalent and while is defined later than
  // copy0. However, if we actually skip processing this while{0}->add0
  // allocation, we won't replace this edge, and will end up with the
  // copy0->add0 edge, which illegally extends the lifetime of the alternate
  // memory buffer in copy0.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=3
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = f32[3]{0} get-tuple-element(p0), index=2
    gte3 = pred[] get-tuple-element(p0), index=3
    add = f32[3]{0} add(gte1, gte2)
    negate0 = f32[3]{0} negate(add)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    negate15 = f32[3]{0} negate(negate14)
    negate16 = f32[3]{0} negate(negate15)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, add, negate16, gte3)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    copy2 = f32[3]{0} copy(p0)
    negate = f32[3]{0} negate(copy0)
    tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(copy0, copy1, copy2, p1)
    while = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[3]{0} get-tuple-element(while), index=2
    add0 = f32[3]{0} add(negate, copy0)
    ROOT add1 = f32[3]{0} add(add0, gte)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, ControlPredecessorsBug) {
  // Having control_predecessors on an HLO was preventing us from DCEing an op
  // that doesn't have any users (tuple.1). The scheduler assumes the graph is
  // fully DCEed, which causes some instructions not to be scheduled.
  absl::string_view hlo_string = R"(
  HloModule sort.16, is_scheduled=true

  ENTRY %sort.16 (param.0.1: s32[1], param.1.2: f32[1], param.2.3: u32[1], param.3.4: s32[1]) -> (s32[1], f32[1], u32[1], s32[1]) {
    %param.3.4 = s32[1]{0:T(128)} parameter(3)
    %param.2.3 = u32[1]{0:T(128)} parameter(2)
    %param.1.2 = f32[1]{0:T(128)} parameter(1)
    %param.0.1 = s32[1]{0:T(128)} parameter(0)
    %tuple.1 = (s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %param.0.1, f32[1]{0:T(128)} %param.1.2, u32[1]{0:T(128)} %param.2.3, s32[1]{0:T(128)} %param.3.4), control-predecessors={%param.0.1}
    %get-tuple-element.4 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=0
    %get-tuple-element.5 = f32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=1
    %get-tuple-element.6 = u32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=2
    %get-tuple-element.7 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=3
    %copy.4 = s32[1]{0:T(128)} copy(s32[1]{0:T(128)} %get-tuple-element.4)
    %copy.5 = f32[1]{0:T(128)} copy(f32[1]{0:T(128)} %get-tuple-element.5)
    %copy.6 = u32[1]{0:T(128)} copy(u32[1]{0:T(128)} %get-tuple-element.6)
    %copy.7 = s32[1]{0:T(128)} copy(s32[1]{0:T(128)} %get-tuple-element.7)
    ROOT %tuple.2 = (s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %copy.4, f32[1]{0:T(128)} %copy.5, u32[1]{0:T(128)} %copy.6, s32[1]{0:T(128)} %copy.7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, ConditionalShouldBeAllocatedInAlternateMem) {
  // Checks if simple conditionals get alternate memory allocations.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg1 = f32[3]{0} negate(gte)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    // Check that copy and gtes got alternate memory allocations.
    auto copy =
        module->GetComputationWithName("entry")->GetInstructionWithName("copy");
    EXPECT_EQ(copy->shape().layout().memory_space(), kAlternateMemorySpace);
    auto neg1 = module->GetComputationWithName("true_computation")
                    ->GetInstructionWithName("neg1");
    auto neg1_operand = neg1->operand(0);
    EXPECT_EQ(neg1_operand->shape().layout().memory_space(),
              kAlternateMemorySpace);
    auto neg2 = module->GetComputationWithName("false_computation")
                    ->GetInstructionWithName("neg2");
    auto neg2_operand = neg2->operand(0);
    EXPECT_EQ(neg2_operand->shape().layout().memory_space(),
              kAlternateMemorySpace);
  }
}

TEST_P(MemorySpaceAssignmentTest, ConditionalAvoidsUnnecessaryPrefetch) {
  // Checks if we avoid unnecessary allocation in alternate memory if the input
  // won't be used in the computation for a long time.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}, f32[3]{0}) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    neg0 = f32[3]{0} negate(gte0)
    neg1 = f32[3]{0} negate(neg0)
    neg2 = f32[3]{0} negate(neg1)
    neg3 = f32[3]{0} negate(neg2)
    neg4 = f32[3]{0} negate(neg3)
    neg5 = f32[3]{0} negate(neg4)
    neg6 = f32[3]{0} negate(neg5)
    neg7 = f32[3]{0} negate(neg6)
    neg8 = f32[3]{0} negate(neg7)
    neg9 = f32[3]{0} negate(neg8)
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    ROOT add = f32[3]{0} add(neg9, gte1)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple0 = (f32[3]{0}, f32[3]{0}) tuple(copy0, copy1)
    tuple1 = (f32[3]{0}) tuple(copy0)
    ROOT conditional = f32[3]{0} conditional(p1, tuple0, tuple1), true_computation=true_computation, false_computation=false_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    // Check that copy1 doesn't get unnecessarily allocated in alternate mem
    // (due to long negate chain in true_computation) but is prefetched before
    // add.
    auto copy0 =
        module->GetComputationWithName("entry")->GetInstructionWithName(
            "copy0");
    EXPECT_EQ(copy0->shape().layout().memory_space(), kAlternateMemorySpace);
    auto copy1 =
        module->GetComputationWithName("entry")->GetInstructionWithName(
            "copy1");
    EXPECT_EQ(copy1->shape().layout().memory_space(), kDefaultMemorySpace);
    auto add = module->GetComputationWithName("true_computation")
                   ->GetInstructionWithName("add");
    auto add_operand = add->operand(1);
    EXPECT_EQ(add_operand->shape().layout().memory_space(),
              kAlternateMemorySpace);
  }
}

TEST_P(MemorySpaceAssignmentTest, ConditionalMultiUse) {
  // Make sure there is an evict when there is a conditional use followed by
  // another use.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}, f32[3]{0}) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    add0 = f32[3]{0} add(gte0, gte1)
    neg0 = f32[3]{0} negate(add0)
    neg1 = f32[3]{0} negate(neg0)
    neg2 = f32[3]{0} negate(neg1)
    neg3 = f32[3]{0} negate(neg2)
    neg4 = f32[3]{0} negate(neg3)
    neg5 = f32[3]{0} negate(neg4)
    neg6 = f32[3]{0} negate(neg5)
    neg7 = f32[3]{0} negate(neg6)
    neg8 = f32[3]{0} negate(neg7)
    ROOT neg9 = f32[3]{0} negate(neg8)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple0 = (f32[3]{0}, f32[3]{0}) tuple(copy0, copy1)
    tuple1 = (f32[3]{0}) tuple(copy0)
    conditional = f32[3]{0} conditional(p1, tuple0, tuple1), true_computation=true_computation, false_computation=false_computation
    ROOT add1 = f32[3]{0} add(copy1, conditional)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    // Make sure the copy1->add edge is in alternate memory. Before conditional,
    // this should be evicted to default memory and neg uses the input from
    // default memory.
    auto copy1 =
        module->GetComputationWithName("entry")->GetInstructionWithName(
            "copy1");
    EXPECT_EQ(copy1->shape().layout().memory_space(), kAlternateMemorySpace);
    auto add0 = module->GetComputationWithName("true_computation")
                    ->GetInstructionWithName("add0");
    auto add0_operand = add0->operand(1);
    EXPECT_EQ(add0_operand->shape().layout().memory_space(),
              kAlternateMemorySpace);
    auto add1 =
        module->GetComputationWithName("entry")->GetInstructionWithName("add1");
    auto add1_operand = add1->operand(0);
    EXPECT_EQ(add1_operand->shape().layout().memory_space(),
              kDefaultMemorySpace);
    EXPECT_EQ(add1_operand->opcode(), HloOpcode::kCopyDone);
  }
}

TEST_P(MemorySpaceAssignmentTest, ConditionalMultiUseInWhile) {
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg1 = f32[3]{0} negate(gte)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    cond_tuple = (f32[3]{0}) tuple(gte0)
    conditional = f32[3]{0} conditional(gte2, cond_tuple, cond_tuple), true_computation=true_computation, false_computation=false_computation
    add = f32[3]{0} add(conditional, gte1)
    neg0 = f32[3]{0} negate(add)
    neg1 = f32[3]{0} negate(neg0)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, neg1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy0, copy1, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    ROOT gte = f32[3]{0} get-tuple-element(while), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    // Make sure copy1/while{0}/cond_tuple{0} gets alternate memory allocation.
    // This will force an eviction and a prefetch for while body root.
    auto copy0 =
        module->GetComputationWithName("entry")->GetInstructionWithName(
            "copy0");
    EXPECT_EQ(copy0->shape().layout().memory_space(), kAlternateMemorySpace);
    auto conditional = module->GetComputationWithName("while_body")
                           ->GetInstructionWithName("conditional");
    auto conditional_operand = conditional->operand(1);
    EXPECT_EQ(ShapeUtil::GetSubshape(conditional_operand->shape(), {0})
                  .layout()
                  .memory_space(),
              kAlternateMemorySpace);
    auto while_root =
        module->GetComputationWithName("while_body")->root_instruction();
    auto while_root_operand = while_root->operand(0);
    EXPECT_THAT(
        while_root_operand,
        op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                      op::AsyncCopy(kDefaultMemorySpace, kAlternateMemorySpace,
                                    op::GetTupleElement(op::Parameter(0)))));
  }
}

TEST_P(MemorySpaceAssignmentTest, NestedConditional) {
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg1 = f32[3]{0} negate(gte)
  }

  false_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  true_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    slice = f32[1]{0} slice(gte), slice={[0:1]}
    bitcast = f32[] bitcast(slice)
    constant = f32[] constant(0.0)
    compare = pred[] compare(bitcast, constant), direction=GT
    ROOT conditional = f32[3]{0} conditional(compare, p0, p0), true_computation=true_computation2, false_computation=false_computation2
  }

  false_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg3 = f32[3]{0} negate(gte)
  }


  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation1, false_computation=false_computation1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    // Make sure alternate memory allocation gets propagated into both levels of
    // conditional.
    auto copy =
        module->GetComputationWithName("entry")->GetInstructionWithName("copy");
    EXPECT_EQ(copy->shape().layout().memory_space(), kAlternateMemorySpace);
    auto neg1_operand = module->GetComputationWithName("true_computation2")
                            ->GetInstructionWithName("neg1")
                            ->operand(0);
    auto neg2_operand = module->GetComputationWithName("false_computation2")
                            ->GetInstructionWithName("neg2")
                            ->operand(0);
    auto neg3_operand = module->GetComputationWithName("false_computation1")
                            ->GetInstructionWithName("neg3")
                            ->operand(0);
    EXPECT_EQ(neg1_operand->shape().layout().memory_space(),
              kAlternateMemorySpace);
    EXPECT_EQ(neg2_operand->shape().layout().memory_space(),
              kAlternateMemorySpace);
    EXPECT_EQ(neg3_operand->shape().layout().memory_space(),
              kAlternateMemorySpace);
  }
}

TEST_P(MemorySpaceAssignmentTest, NestedConditionalBufferReuseVerificationBug) {
  // Tests a spurious verification failure when there are nested conditionals
  // and the innermost conditional computation reuses the buffer. Here, both the
  // parameter of true_computation2 and neg2 will get the same buffer. Make sure
  // that verification doesn't claim a failure in this case.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    neg1 = f32[3]{0} negate(gte)
    neg2 = f32[3]{0} negate(neg1)
    ROOT neg3 = f32[3]{0} negate(neg2)
  }

  false_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg4 = f32[3]{0} negate(gte)
  }

  true_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    slice = f32[1]{0} slice(gte), slice={[0:1]}
    bitcast = f32[] bitcast(slice)
    constant = f32[] constant(0.0)
    compare = pred[] compare(bitcast, constant), direction=GT
    tuple = (f32[3]{0}) tuple(gte)
    ROOT conditional = f32[3]{0} conditional(compare, tuple, tuple), true_computation=true_computation2, false_computation=false_computation2
  }

  false_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg5 = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation1, false_computation=false_computation1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest,
       ConditionalComputationBufferOverlapBeforeParam) {
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  false_computation {
    c = f32[3]{0} constant({0.0, 1.0, 2.0})
    neg0 = f32[3]{0} negate(c)
    neg1 = f32[3]{0} negate(neg0)
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT add = f32[3]{0} add(gte, neg1)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(module.get());

  auto get_offset = [&](absl::string_view hlo_name) {
    for (const auto& chunk : preset_assignments->chunks()) {
      if (chunk.first.instruction->name() == hlo_name) {
        return chunk.second.offset;
      }
    }
    return static_cast<int64_t>(-1);
  };

  int64_t copy_offset = get_offset("copy");
  int64_t neg0_offset = get_offset("neg0");
  EXPECT_NE(copy_offset, -1);
  EXPECT_NE(neg0_offset, -1);
  EXPECT_NE(copy_offset, neg0_offset);
}

TEST_P(MemorySpaceAssignmentTest,
       RequestIdentifierShouldNotBeAllocatedInAlternateMem) {
  // Ensure that request identifier returned by Send/Recv HLOs are not allocated
  // in the alternate memory.
  absl::string_view hlo_string = R"(
  HloModule SendRecv, is_scheduled=true

  ENTRY %AddDependency (p: f32[3]) -> f32[3] {
    %p = f32[3]{0} parameter(0)
    %after-all = token[] after-all()
    %recv.4 = (f32[3]{0}, u32[], token[]) recv(token[] %after-all), channel_id=7
    %recv-done.4 = (f32[3]{0}, token[]) recv-done((f32[3]{0}, u32[], token[]) %recv.4), channel_id=7
    %token.1 = token[] get-tuple-element((f32[3]{0}, token[]) %recv-done.4), index=1
    %data = f32[3]{0} get-tuple-element((f32[3]{0}, token[]) %recv-done.4), index=0
    %send = (f32[3]{0}, u32[], token[]) send(f32[3]{0} %data, token[] %token.1), channel_id=2
    %send-done = token[] send-done((f32[3]{0}, u32[], token[]) %send), channel_id=2
    ROOT %add = f32[3]{0} add(f32[3]{0} %p, f32[3]{0} %data)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kSend ||
        instruction->opcode() == HloOpcode::kRecv) {
      const Shape& request_identifier_shape =
          ShapeUtil::GetSubshape(instruction->shape(), {1});
      EXPECT_NE(request_identifier_shape.layout().memory_space(),
                kAlternateMemorySpace);
    }
  }
}

TEST_P(MemorySpaceAssignmentTest, SendDoneShouldHaveSendOperand) {
  // Ensure that SendDone has only a Send operand.
  absl::string_view hlo_string = R"(
  HloModule SendRecv, is_scheduled=true

  ENTRY %AddDependency (p: f32[3]) -> f32[3] {
    %p0 = f32[3]{0} parameter(0)
    %p1 = f32[3]{0} parameter(1)
    %neg0 = f32[3]{0} negate(f32[3]{0} %p1)
    %neg1 = f32[3]{0} negate(f32[3]{0} %neg0)
    %neg2 = f32[3]{0} negate(f32[3]{0} %neg1)
    %neg3 = f32[3]{0} negate(f32[3]{0} %neg2)
    %neg4 = f32[3]{0} negate(f32[3]{0} %neg3)
    %neg5 = f32[3]{0} negate(f32[3]{0} %neg4)
    %neg6 = f32[3]{0} negate(f32[3]{0} %neg5)
    %after-all = token[] after-all()
    %send = (f32[3]{0}, u32[], token[]) send(f32[3]{0} %p0, token[] %after-all), channel_id=2
    %send-done = token[] send-done((f32[3]{0}, u32[], token[]) %send), channel_id=2
    ROOT %add = f32[3]{0} add(f32[3]{0} %p0, f32[3]{0} %neg6)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, SendAndSendDoneShouldGetSameAllocation) {
  // Ensure that Send and SendDone have the same allocation.
  absl::string_view hlo_string = R"(
  HloModule SendRecv, is_scheduled=true

  ENTRY %AddDependency (p: f32[3]) -> f32[3] {
    %p0 = f32[3]{0} parameter(0)
    %p1 = f32[3]{0} parameter(1)
    %after-all = token[] after-all()
    %send = (f32[3]{0}, u32[], token[]) send(f32[3]{0} %p0, token[] %after-all), channel_id=2
    %neg0 = f32[3]{0} negate(f32[3]{0} %p1)
    %neg1 = f32[3]{0} negate(f32[3]{0} %neg0)
    %neg2 = f32[3]{0} negate(f32[3]{0} %neg1)
    %neg3 = f32[3]{0} negate(f32[3]{0} %neg2)
    %neg4 = f32[3]{0} negate(f32[3]{0} %neg3)
    %neg5 = f32[3]{0} negate(f32[3]{0} %neg4)
    %neg6 = f32[3]{0} negate(f32[3]{0} %neg5)
    %send-done = token[] send-done((f32[3]{0}, u32[], token[]) %send), channel_id=2
    ROOT %add = f32[3]{0} add(f32[3]{0} %p0, f32[3]{0} %neg6)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/4);
}

TEST_P(MemorySpaceAssignmentTest, LastUseOpt) {
  // Test that checks the last use optimization. It uses two buffers that should
  // be placed in alternate memory.
  //
  //      +-------+
  //     /         \
  // add1--->sub1   +-------->mul2
  //              mul1===>add2
  //
  // Without the last use optimization, the mul1 buffer will be assigned first
  // (because it is larger) to offset 0. Then, add1 will be scheduled for the
  // add1 to sub1 segment. Because offset 0 is available, it will get that
  // offset. But because offset 0 is not available in the sub1 to mul2 offset,
  // it will end up in unnecessary copies. With the last use optimization, these
  // copies can be optimized away.
  HloComputation::Builder builder(TestName());
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(F32, {2, 4});
  PaddingConfig padding_config = MakeEdgePaddingConfig({{0, 0}, {0, 1}});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape1, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape2, "p1"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, p0, p0));
  HloInstruction* sub1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kSubtract, p0, add1));
  HloInstruction* mul1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape2, HloOpcode::kMultiply, p1, p1));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape2, HloOpcode::kAdd, mul1, p1));
  HloInstruction* mul2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kMultiply, add1, sub1));
  HloInstruction* padding_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(F32)));
  HloInstruction* padded_mul2 = builder.AddInstruction(
      HloInstruction::CreatePad(shape2, mul2, padding_value, padding_config));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape2, HloOpcode::kAdd, add2, padded_mul2));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, add1, sub1, mul1, add2, mul2,
                                      padding_value, padded_mul2, add3});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_THAT(
      mul2,
      op::Multiply(
          op::Add(op::Parameter(0), op::Parameter(0)),
          op::Subtract(op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                     op::Parameter(0)),
                       op::Add(op::Parameter(0), op::Parameter(0)))));
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule1) {
  // Test to ensure CopyStart/CopyDone is placed only in the entry computation.
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  // Free cond_param[] (16 bytes), Alloc PRED[] (1 byte)
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_data_increment =
      body_builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}})));
  HloInstruction* body_data_mul =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, body_data, body_data));
  HloInstruction* body_data_add =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data, body_data_increment));
  HloInstruction* body_data_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data_add, body_data_mul));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data_next, body_iter_next}));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_iter"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_data"));
  HloInstruction* p2 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape, "p2"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({data, iter}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloInstruction* while_data = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, while_op, 0));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, while_data, p2));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(body_computation,
                        {body_param, body_iter, body_data, body_iter_increment,
                         body_iter_next, body_data_increment, body_data_mul,
                         body_data_add, body_data_next, body_out});
  schedule.set_sequence(entry_computation,
                        {iter, data, p2, tuple, while_op, while_data, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), -1, 50);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule2) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(xla::F32, {3, 3});

  auto call_builder = HloComputation::Builder("Call");
  HloInstruction* call_param = call_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "call_param"));
  HloInstruction* call_param2 = call_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape2, "call_param2"));
  HloInstruction* slice = call_builder.AddInstruction(
      HloInstruction::CreateSlice(shape, call_param2, {0, 0}, {2, 3}, {1, 1}));
  HloInstruction* mul =
      call_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, call_param, slice));
  HloInstruction* negate0 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, mul));
  HloInstruction* negate1 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* add0 =
      call_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, call_param, negate7));
  HloComputation* call_computation =
      module->AddEmbeddedComputation(call_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape2, "p1"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p0));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add1, p0));
  HloInstruction* negate8 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape2, HloOpcode::kNegate, p1));
  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {add1, negate8}, call_computation));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, add1));
  HloInstruction* add4 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, call, add3));
  HloInstruction* add5 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add2, add4));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      call_computation,
      {call_param, call_param2, slice, mul, negate0, negate1, negate2, negate3,
       negate4, negate5, negate6, negate7, add0});
  schedule.set_sequence(entry_computation,
                        {p0, p1, add1, add2, negate8, call, add3, add4, add5});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), -1, 5);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule3) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(xla::F32, {3, 3});

  auto call_builder = HloComputation::Builder("Call");
  HloInstruction* call_param = call_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "call_param"));
  // Use shape2 here which is larger (scheduled earlier) to occupy alternate
  // memory at the beginning. This should cause a situation where the prefetch
  // of add1 later in the function body gets the wrong offset which cannot be
  // communicated to the outside the function.
  HloInstruction* iota =
      call_builder.AddInstruction(HloInstruction::CreateIota(shape2, 0));
  HloInstruction* slice = call_builder.AddInstruction(
      HloInstruction::CreateSlice(shape, iota, {0, 0}, {2, 3}, {1, 1}));
  HloInstruction* mul =
      call_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, call_param, slice));
  HloInstruction* negate0 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, mul));
  HloInstruction* negate1 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* add0 =
      call_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, call_param, negate7));
  HloComputation* call_computation =
      module->AddEmbeddedComputation(call_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p0));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add1, p0));
  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {add1}, call_computation));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, call, add1));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      call_computation,
      {call_param, iota, slice, mul, negate0, negate1, negate2, negate3,
       negate4, negate5, negate6, negate7, add0});
  schedule.set_sequence(entry_computation, {p0, add1, add2, call, add3});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), -1, 5);
}

// TODO(berkin): This might be an incorrect input graph, investigate.
TEST_P(MemorySpaceAssignmentTest, DISABLED_NonEntryComputationSchedule4) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(xla::F32, {3, 3});

  auto true_builder = HloComputation::Builder("True");
  HloInstruction* true_param = true_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "true_param"));
  HloInstruction* iota =
      true_builder.AddInstruction(HloInstruction::CreateIota(shape2, 0));
  HloInstruction* slice = true_builder.AddInstruction(
      HloInstruction::CreateSlice(shape, iota, {0, 0}, {2, 3}, {1, 1}));
  HloInstruction* mul =
      true_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, true_param, slice));
  HloInstruction* negate0 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, mul));
  HloInstruction* negate1 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* add0 =
      true_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, true_param, negate7));
  HloComputation* true_computation =
      module->AddEmbeddedComputation(true_builder.Build());

  auto false_builder = HloComputation::Builder("False");
  HloInstruction* false_param = false_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "false_param"));
  HloComputation* false_computation =
      module->AddEmbeddedComputation(false_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p0));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add1, p0));
  HloInstruction* pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloInstruction* conditional =
      builder.AddInstruction(HloInstruction::CreateConditional(
          shape, pred, add1, true_computation, add2, false_computation));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, conditional, add1));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      true_computation,
      {true_param, iota, slice, mul, negate0, negate1, negate2, negate3,
       negate4, negate5, negate6, negate7, add0});
  schedule.set_sequence(false_computation, {false_param});
  schedule.set_sequence(entry_computation,
                        {p0, add1, add2, pred, conditional, add3});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), -1, 5);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule5) {
  // This test reproduces the failure in b/143288178.  Given a graph like the
  // following:
  //
  // ... = foo(a)
  // tuple = tuple((..., a)
  // ... = while(tuple) {
  //   p = param(0)
  //   a1 = get-tuple-element(p), index=n-1
  //   ...
  //   ROOT tuple((..., a1))
  // }
  //
  // If a copy to alternate memory is inserted before foo, and if the size of
  // the while body is less than max prefetch interval so that the copy-done is
  // kept in the alternate memory, then we end up referring to the copy-done in
  // the root instruction of the while loop body. I.e.,
  //
  // cs = copy-start(a)
  // ...
  // cd = copy-done(cs)
  // ... = foo(cd)
  // tuple = tuple((..., cd)
  // ... = while(tuple) {
  //   p = param(0)
  //   a1 = get-tuple-element(p), index=n-1
  //   ...
  //   ROOT tuple((..., cd))  <-- Error: cd belongs to outside computation.
  // }
  //
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({shape, scalar_shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_data2 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 2));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data, body_iter_next, body_data2}));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_data"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_iter"));
  HloInstruction* data2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, scalar_shape, "param_data2"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, data));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kSubtract, iter, data2));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({negate7, iter, data2}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloInstruction* while_data = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, while_op, 1));
  HloInstruction* root =
      builder.AddInstruction(HloInstruction::CreateTuple({while_data, sub}));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(body_computation,
                        {body_param, body_iter, body_data, body_iter_increment,
                         body_iter_next, body_data2, body_out});
  schedule.set_sequence(
      entry_computation,
      {iter, data, data2, negate0, negate1, negate2, negate3, negate4, negate5,
       negate6, negate7, sub, tuple, while_op, while_data, root});
  TF_CHECK_OK(module->set_schedule(schedule));

  // Set a large max prefetch interval so that the buffer can be kept in
  // alternate memory.
  AssignMemorySpace(module.get(), -1, 20);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule6) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, scalar_shape, shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_negate0 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_data));
  HloInstruction* body_negate1 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate0));
  HloInstruction* body_negate2 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate1));
  HloInstruction* body_negate3 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate2));
  HloInstruction* body_negate4 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate3));
  HloInstruction* body_negate5 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate4));
  HloInstruction* body_negate6 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate5));
  HloInstruction* body_negate7 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate6));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data, body_iter_next, body_negate7}));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_data"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_iter"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, data));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({data, iter, negate7}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloInstruction* while_data = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, while_op, 0));
  HloInstruction* while_data2 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, while_op, 2));
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, while_data, while_data2));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(
      body_computation,
      {body_param, body_iter, body_data, body_negate0, body_negate1,
       body_negate2, body_negate3, body_negate4, body_negate5, body_negate6,
       body_negate7, body_iter_increment, body_iter_next, body_out});
  schedule.set_sequence(
      entry_computation,
      {iter, data, negate0, negate1, negate2, negate3, negate4, negate5,
       negate6, negate7, tuple, while_op, while_data, while_data2, root});
  TF_CHECK_OK(module->set_schedule(schedule));

  // Pick a large max prefetch interval to ensure all the while inputs are
  // allocated in the alternate memory.
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/25);

  // Index {0} of the while loop argument is not written inside the while loop,
  // so it can be trivially placed in the alternate memory space.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {0})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{1, 0}, /*dim_level_types=*/{}, /*dim_unique=*/{},
          /*dim_ordered=*/{}, /*tiles=*/{},
          /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*pointer_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          kAlternateMemorySpace);
  // Index {1} is a scalar, so it is always placed in the default memory.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {1})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{}, /*dim_level_types=*/{}, /*dim_unique=*/{},
          /*dim_ordered=*/{}, /*tiles=*/{},
          /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*pointer_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          kDefaultMemorySpace);
  // Index {2} of the while loop is placed in the default memory.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {2})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{1, 0}, /*dim_level_types=*/{}, /*dim_unique=*/{},
          /*dim_ordered=*/{}, /*tiles=*/{},
          /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*pointer_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          kDefaultMemorySpace);

  // Expect the layout for the while loop and its aliased buffers.
  EXPECT_THAT(while_op, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(while_op->operand(0), op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(cond_param, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(body_param, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(body_out, op::ShapeWithLayout(tuple_shape));
}

TEST_P(MemorySpaceAssignmentTest, DanglingCopy) {
  // This situation was encountered in vss, where there is a mismatch in the
  // memory space in preset assignments and the output graph.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});

  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 0));
  HloInstruction* p1a = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 1));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, p1a));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* p1b = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1b));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p, p0, negate0, negate1, negate2, negate3, negate4, negate5,
                    negate6, p1a, copy, p1b, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, MultiOutputFusion) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion_builder("fusion");
  HloInstruction* fusion_param0 = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion_param1 = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1"));
  fusion_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion_param0, fusion_param1}));
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      tuple_shape, HloInstruction::FusionKind::kCustom, {p0, p0},
      fusion_computation));
  HloInstruction* element0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion, 0));
  HloInstruction* element1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, element0, element1));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, fusion, element0, element1, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, TupleInput) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion_builder("fusion");
  HloInstruction* fusion_param = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion_element0 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 0));
  HloInstruction* fusion_element1 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 1));
  fusion_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion_element0, fusion_element1));
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p1));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({negate0, negate1}));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kCustom, {tuple}, fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, tuple, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, TupleToTuple1) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion0_builder("fusion0");
  HloInstruction* fusion0_param0 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0_param1 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1"));
  fusion0_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion0_param0, fusion0_param1}));
  HloComputation* fusion0_computation =
      module->AddEmbeddedComputation(fusion0_builder.Build());

  HloComputation::Builder fusion1_builder("fusion1");
  HloInstruction* fusion1_param = fusion1_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion1_element0 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 0));
  HloInstruction* fusion1_element1 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 1));
  fusion1_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion1_element0, fusion1_element1));
  HloComputation* fusion1_computation =
      module->AddEmbeddedComputation(fusion1_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0 = builder.AddInstruction(HloInstruction::CreateFusion(
      tuple_shape, HloInstruction::FusionKind::kCustom, {p0, p0},
      fusion0_computation));
  HloInstruction* element0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion0, 0));
  HloInstruction* element1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion0, 1));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, element0, element1));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add0, negate6));
  HloInstruction* fusion1 = builder.AddInstruction(
      HloInstruction::CreateFusion(shape, HloInstruction::FusionKind::kCustom,
                                   {fusion0}, fusion1_computation));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, add1, fusion1));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation,
      {p0, fusion0, element0, element1, negate0, negate1, negate2, negate3,
       negate4, negate5, negate6, add0, add1, fusion1, mul});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), -1, 5);
  EXPECT_THAT(fusion1,
              op::Fusion(op::Tuple(
                  op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                op::GetTupleElement(op::Fusion(), 0)),
                  op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                op::GetTupleElement(op::Fusion(), 1)))));
}

TEST_P(MemorySpaceAssignmentTest, TupleToTuple2) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  Shape nested_tuple_shape = ShapeUtil::MakeTupleShape({shape, tuple_shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion0_builder("fusion0");
  HloInstruction* fusion0_param0 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0_param1 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* fusion0_tuple = fusion0_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion0_param0, fusion0_param1}));
  fusion0_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion0_param0, fusion0_tuple}));
  HloComputation* fusion0_computation =
      module->AddEmbeddedComputation(fusion0_builder.Build());

  HloComputation::Builder fusion1_builder("fusion1");
  HloInstruction* fusion1_param = fusion1_builder.AddInstruction(
      HloInstruction::CreateParameter(0, nested_tuple_shape, "p"));
  HloInstruction* fusion1_element0 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 0));
  HloInstruction* fusion1_element1 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple_shape, fusion1_param, 1));
  HloInstruction* fusion1_element2 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_element1, 1));
  fusion1_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion1_element0, fusion1_element2));
  HloComputation* fusion1_computation =
      module->AddEmbeddedComputation(fusion1_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0 = builder.AddInstruction(HloInstruction::CreateFusion(
      nested_tuple_shape, HloInstruction::FusionKind::kCustom, {p0, p0},
      fusion0_computation));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* fusion1 = builder.AddInstruction(
      HloInstruction::CreateFusion(shape, HloInstruction::FusionKind::kCustom,
                                   {fusion0}, fusion1_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p0, fusion0, negate0, negate1, negate2, negate3, negate4,
                    negate5, negate6, fusion1});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), -1, 5);

  EXPECT_THAT(
      fusion1,
      op::Fusion(op::Tuple(
          op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                        op::GetTupleElement(op::Fusion(), 0)),
          op::Tuple(
              op::AsyncCopy(
                  kAlternateMemorySpace, kDefaultMemorySpace,
                  op::GetTupleElement(op::GetTupleElement(op::Fusion(), 1), 0)),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::GetTupleElement(
                                op::GetTupleElement(op::Fusion(), 1), 1))))));
}

TEST_P(MemorySpaceAssignmentTest, TupleToTuple3) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion0_builder("fusion0");
  HloInstruction* fusion0_param0 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0_param1 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1"));
  fusion0_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion0_param0, fusion0_param1}));
  HloComputation* fusion0_computation =
      module->AddEmbeddedComputation(fusion0_builder.Build());

  HloComputation::Builder fusion1_builder("fusion1");
  HloInstruction* fusion1_param = fusion1_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion1_element0 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 0));
  HloInstruction* fusion1_element1 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 1));
  fusion1_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion1_element0, fusion1_element1));
  HloComputation* fusion1_computation =
      module->AddEmbeddedComputation(fusion1_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0 = builder.AddInstruction(HloInstruction::CreateFusion(
      tuple_shape, HloInstruction::FusionKind::kCustom, {p0, p0},
      fusion0_computation));
  HloInstruction* fusion1 = builder.AddInstruction(
      HloInstruction::CreateFusion(shape, HloInstruction::FusionKind::kCustom,
                                   {fusion0}, fusion1_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, fusion0, fusion1});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
  EXPECT_THAT(fusion1, op::Fusion(op::Fusion()));
}

TEST_P(MemorySpaceAssignmentTest, InputOutputAlias) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, add));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({p0, add}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p, p0, negate0, negate1, negate2, negate3, negate4, negate5,
                    negate6, p1, add, negate7, tuple});
  TF_CHECK_OK(module->set_schedule(schedule));

  // Make input {0} alias with output {0} and input {1} alias with output {1}.
  TF_CHECK_OK(module->input_output_alias_config().SetUpAlias({0}, 0, {0}));
  TF_CHECK_OK(module->input_output_alias_config().SetUpAlias({1}, 0, {1}));

  AssignMemorySpace(module.get());

  // Make sure the input is in the default memory space.
  EXPECT_EQ(p->shape().tuple_shapes(0).layout().memory_space(),
            kDefaultMemorySpace);
  EXPECT_EQ(p->shape().tuple_shapes(1).layout().memory_space(),
            kDefaultMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, CostAnalysis) {
  // This is mostly a smoke test since it's difficult and brittle to work out
  // the cost of the HLO instructions.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpaceUsingCostAnalysis(module.get());
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, MemoryBoundednessBufferIntervalCompare) {
  // This test is carefully crafted to force only negates to be allocated to the
  // alternate memory. The graph consists of interleaving negate and tanh
  // operations:
  //
  //        +------+      +-------+      +-----
  //       /        \    /         \    /
  //  negate  tanh  negate  tanh   negate  tanh
  //             \          /  \           /
  //              +--------+    +---------+
  //
  // The alternate memory is sized to fit only two f32[4,3] tensors at a time.
  // Also, transcendentals are made to be lower bandwidth than FLOPs. So, the
  // MemoryBoundednessBufferIntervalCompare should prioritize the negates, which
  // are more memory bound.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* tanh0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p1));
  HloInstruction* tanh1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* tanh2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh1));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* tanh3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh2));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* tanh4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh3));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({tanh4, negate4}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, tanh0, negate0, tanh1, negate1, tanh2, negate2,
                         tanh3, negate3, tanh4, negate4, tuple});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpaceUsingCostAnalysis(module.get());
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {4, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kDefaultMemorySpace);
  // Expect only negates to be in alternate memory space. Not all might fit but
  // make sure at least one does.
  std::vector<HloInstruction*> negate_instructions = {negate0, negate1, negate2,
                                                      negate3, negate4};
  int64_t num_negates_in_alternate_mem = absl::c_count_if(
      negate_instructions, [&](const HloInstruction* instruction) {
        return instruction->shape().layout().memory_space() ==
               kAlternateMemorySpace;
      });
  EXPECT_GE(num_negates_in_alternate_mem, 1);
  EXPECT_THAT(tanh0, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(tanh1, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(tanh2, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(tanh3, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(tanh4, op::ShapeWithLayout(shape_in_default_mem));
}

TEST_P(MemorySpaceAssignmentTest, SimpleWhileTupleTest) {
  Shape s32 = ShapeUtil::MakeShape(xla::S32, {});
  Shape f32v1 = ShapeUtil::MakeShape(F32, {1});
  Shape t_s32_f32v1 = ShapeUtil::MakeTupleShape({s32, f32v1});
  auto module = CreateNewVerifiedModule("SimpleWhile");
  HloSchedule schedule(module.get());

  // A simple compare-to-limit (x < 4) computation for a While.
  //
  // condition:
  //   const4[s32] -----------------------------------\
  //                                                   \
  //   param[(s32,f32[4])] --- get-tuple-element[0] --- less-than
  //
  HloComputation* cond_computation;
  {
    auto builder = HloComputation::Builder("WhileCond");
    auto const4 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(4)));
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v1, "x"));
    auto index = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(const4->shape(), param, 0));
    auto compare = builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), index,
                                      const4, ComparisonDirection::kLt));
    cond_computation = module->AddEmbeddedComputation(builder.Build());
    schedule.set_sequence(cond_computation, {const4, param, index, compare});
  }

  // Builds a simple body computation for a While.
  //
  // body:
  //   constv[f32[1]] --------------------------------------\
  //                                                         \
  //                           /--- get-tuple-elementv[1] --- addv ---\
  //   param[(s32,f32[1])] ---|                                    tuple
  //                           \--- get-tuple-elementc[0] --- addc ---/
  //                                                         /
  //   const1[s32] -----------------------------------------/
  //
  HloComputation* body_computation;
  {
    auto builder = HloComputation::Builder("WhileBody");
    auto const1 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(1)));
    auto constv = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1.1f})));
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v1, "x"));
    auto indexc = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(const1->shape(), param, 0));
    auto addc = builder.AddInstruction(HloInstruction::CreateBinary(
        indexc->shape(), HloOpcode::kAdd, indexc, const1));
    auto indexv = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(constv->shape(), param, 1));
    auto addv = builder.AddInstruction(HloInstruction::CreateBinary(
        constv->shape(), HloOpcode::kAdd, indexv, constv));
    auto tuple =
        builder.AddInstruction(HloInstruction::CreateTuple({addc, addv}));
    body_computation = module->AddEmbeddedComputation(builder.Build());
    schedule.set_sequence(body_computation, {const1, constv, param, indexc,
                                             addc, indexv, addv, tuple});
  }

  // This tests a simple while loop where the parameters are aliased with the
  // output buffers.
  auto builder = HloComputation::Builder("SimpleWhile");
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, t_s32_f32v1, "param"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(s32, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32v1, param, 1));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
  auto while0 = builder.AddInstruction(HloInstruction::CreateWhile(
      t_s32_f32v1, cond_computation, body_computation, tuple));

  HloComputation* computation = module->AddEntryComputation(builder.Build());
  schedule.set_sequence(computation, {param, gte0, gte1, tuple, while0});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/50);

  // Ensure all parameters and while are placed in default memory.
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {4, 6},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kDefaultMemorySpace);
  Shape s32_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      xla::S32, {},
      /*minor_to_major=*/{}, /*tiles=*/{}, kDefaultMemorySpace);
  Shape f32v1_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {1},
      /*minor_to_major=*/{0}, /*tiles=*/{}, kDefaultMemorySpace);
  Shape t_s32_f32v1_in_default_mem =
      ShapeUtil::MakeTupleShape({s32_in_default_mem, f32v1_in_default_mem});
  EXPECT_THAT(param, op::ShapeWithLayout(t_s32_f32v1_in_default_mem));
  EXPECT_THAT(while0, op::ShapeWithLayout(t_s32_f32v1_in_default_mem));
}

TEST_P(MemorySpaceAssignmentTest, EvictionsShouldntBeDelayed) {
  // This test reproduces an eviction scheduling bug where evictions to default
  // memory can happen later than intended, causing memory corruption. This test
  // is a variant of MemoryBoundednessBufferIntervalCompare but uses f32[4,3]
  // tensors instead, so at most two tensors should fit in the alternate memory
  // space at a given time. We have a number of redundant operations
  // (tanh_redundant ops) that do not have users. The bug was due to
  // SimplifyGraph removing dead instructions, and removing them from the
  // schedule. However, the CopyStart/CopyDone insertion relies on the schedule
  // indexes, so they could be inserted too late.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* tanh0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, tanh0));
  HloInstruction* tanh1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, negate0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* tanh2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh1));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* tanh3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh2));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({tanh3, negate3, tanh0}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation,
      {p0, tanh0, tanh_redundant0, tanh_redundant1, tanh_redundant2,
       tanh_redundant3, tanh_redundant4, tanh_redundant5, tanh_redundant6,
       negate0, tanh1, negate1, tanh2, negate2, tanh3, negate3, tuple});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpaceUsingCostAnalysis(module.get());

  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_live_range,
                          HloLiveRange::Run(module->schedule(), *alias_analysis,
                                            module->entry_computation()));

  std::vector<int> num_live_buffers_in_alternate_mem(
      hlo_live_range->flattened_instruction_sequence().size() + 1, 0);

  // Go through each value and for those that are allocated in the alternate
  // memory space, increment (inclusive) num_live_buffers_in_alternate_mem for
  // every time step that they are live.
  for (const HloValue* value : alias_analysis->dataflow_analysis().values()) {
    const Shape& shape = value->shape();
    if (!shape.has_layout() ||
        shape.layout().memory_space() == kDefaultMemorySpace) {
      continue;
    }

    HloLiveRange::TimeBound time_bound =
        hlo_live_range->buffer_live_ranges().at(value);
    for (int i = time_bound.start; i <= time_bound.end; ++i) {
      ++num_live_buffers_in_alternate_mem[i];
    }
  }

  // The test memory can at most hold two f32[4,3] buffers at a time. If there
  // is more than that, it means we have memory corruption.
  for (int i = 0; i < num_live_buffers_in_alternate_mem.size(); ++i) {
    EXPECT_LE(num_live_buffers_in_alternate_mem[i], 2);
  }
}

TEST_P(MemorySpaceAssignmentTest,
       InputOutputsInAlternateMemShouldntBeAssigned) {
  // When input/outputs are marked to be in the alternate memory (e.g.
  // go/tpu-fast-mem-inference), do not allocate those and assume they will live
  // in the alternate memory for the entire computation. The BufferAssignment
  // pass, which is run after this, will allocate those buffers.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kAlternateMemorySpace);
  // p0 is in the default memory space.
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  // p1 is in the alternate memory space.
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape_in_alternate_mem, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape_in_alternate_mem, HloOpcode::kAdd, negate6, p1));
  // Index {0} of the root instruction is in the alternate memory space, index
  // {1} is in the default memory space.
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add, negate5}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, negate0, negate1, negate2, negate3, negate4,
                         negate5, negate6, add, tuple});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.is_allowed_in_alternate_mem_fn = [](const HloValue& value) {
    return true;
  };
  std::unique_ptr<PresetAssignments> preset_assignments = AssignMemorySpace(
      module.get(),
      /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/2, options);

  // Ensure that p1 is in the alternate memory and add, which has p1 as an
  // operand, has a direct dependency to p1 (no CopyStart/CopyDone).
  EXPECT_THAT(p1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(add, op::Add(op::Negate(), op::Parameter(1)));
  // Make sure add is still in the alternate memory space.
  EXPECT_THAT(add, op::ShapeWithLayout(shape_in_alternate_mem));

  // Check the preset assignments and ensure the inputs/outputs in the alternate
  // memory space aren't in the preset assignments. Inputs/outputs in the
  // alternate memory space are left to BufferAssignment to be allocated.
  for (const auto& position_and_chunk : preset_assignments->chunks()) {
    const HloPosition& position = position_and_chunk.first;
    EXPECT_NE(position.instruction, p1);
    EXPECT_NE(position.instruction, add);
  }
}

TEST_P(MemorySpaceAssignmentTest, PendingChunkMemoryCorruptionBug) {
  // Tests a memory corruption bug where the allocated chunk overlaps with a
  // pending chunk. To test this, we provide a new buffer interval compare where
  // we prioritize the allocation of sine, cosine, and tanh to create the
  // situation:
  //
  //    Max memory
  //  -------------------------------------------
  //      +------------+
  //      |     b      |
  //      +------------+
  //  +-------+
  //  |       |
  //  |       |
  //  |   a   |
  //  |       |                 +------------+
  //  |       |                 |     n      |
  //  +-------+                 +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  //
  //
  // Then allocating for buffer d, we have these two prefetch buffers
  // overlapping:
  //
  //    Max memory
  //  -------------------------------------------
  //      +------------+ +----------+
  //      |     b      | | prefetch |
  //      +------------+ | for o    |
  //  +-------+     +---------+     |
  //  |       |     |    |    |     |
  //  |       |     |    |    |     |
  //  |   a   |     |    +----|-----+
  //  |       |     | prefetch| +------------+
  //  |       |     | for m   | |     n      |
  //  +-------+     +---------+ +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  //
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY %Entry {
    %param0 = f32[8,3] parameter(0)
    %param1 = f32[2,4] parameter(1)
    %a = f32[8,3] sine(%param0)
    %b = f32[2,4] cosine(%param1)
    %d = f32[8,3] tanh(%a)
    %c = f32[8,3] negate(%a)
    %e = f32[2,4] negate(%b)
    %f = f32[2,4] negate(%e)
    %g = f32[2,4] negate(%f)
    %h = f32[2,4] negate(%g)
    %i = f32[2,4] negate(%h)
    %j = f32[2,4] negate(%i)
    %k = f32[2,4] negate(%j)
    %l = f32[2,4] negate(%k)
    %m = f32[8,3] negate(%d)
    %n = f32[2,4] sine(%l)
    %o = f32[8,3] negate(%d)
    %p = f32[2,4] negate(%n)
    %q = f32[8,3] negate(%m)
    ROOT %tuple = (f32[2,4], f32[8,3], f32[8,3]) tuple(%p, %q, %o)
  }
  )";

  MemorySpaceAssignment::BufferIntervalCompare buffer_interval_compare =
      [](const MemorySpaceAssignment::BufferInterval& a,
         const MemorySpaceAssignment::BufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kSin:
              return 0;
            case HloOpcode::kCos:
              return 1;
            case HloOpcode::kTanh:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    buffer_interval_compare, &prefetch_interval_picker);
}

TEST_P(MemorySpaceAssignmentTest, WhileAliasedArgumentRequiredAssignmentBug) {
  // Tests an overly pessimistic assertion when the same HloValue is passed
  // multiple times to a while HLO. We already handle this case that the two
  // arguments must alias and get the same allocation in AllocateSegment so the
  // assertion isn't necessary.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  while_condition {
    param1 = (f32[2,4], f32[2,4], f32[2,4]) parameter(0)
    ROOT cond = pred[] constant(true)
  }

  while_body {
    param2 = (f32[2,4], f32[2,4], f32[2,4]) parameter(0)
    gte2 = f32[2,4] get-tuple-element(param2), index=0
    gte3 = f32[2,4] get-tuple-element(param2), index=1
    gte4 = f32[2,4] get-tuple-element(param2), index=2
    add = f32[2,4] add(gte2, gte3)
    ROOT tuple2 = (f32[2,4], f32[2,4], f32[2,4]) tuple(add, gte3, gte4)
  }

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    a = f32[2,4] negate(param0)
    b = f32[2,4] negate(param0)
    tuple = (f32[2,4], f32[2,4], f32[2,4]) tuple(a, b, b)
    while = (f32[2,4], f32[2,4], f32[2,4]) while(tuple), condition=while_condition, body=while_body
    gte1 = f32[2,4] get-tuple-element(while), index=0
    gte2 = f32[2,4] get-tuple-element(while), index=1
    ROOT root = f32[2,4] add(gte1, gte2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, DisallowedUseBug) {
  // When we have a disallowed use (in this case tanh), we aren't allowed to
  // allocate this use in alternate memory. However, if we have another use
  // after this on the same buffer (o), this use may refer to "a" instead of the
  // evicted value, which is illegal because "a" will be allocated in the
  // alternate memory space.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[8,3] cosine(param0)
    b = f32[2,4] negate(param1)
    d = f32[8,3] negate(a)
    c = f32[2,4] negate(b)
    e = f32[2,4] negate(c)
    f = f32[8,3] tanh(a)
    g = f32[2,4] negate(e)
    h = f32[2,4] negate(g)
    i = f32[2,4] negate(h)
    j = f32[2,4] negate(i)
    k = f32[2,4] negate(j)
    l = f32[2,4] negate(k)
    m = f32[2,4] negate(l)
    n = f32[2,4] sine(m)
    o = f32[8,3] negate(a)
    p = f32[2,4] negate(n)
    q = f32[8,3] add(o, f)
    r = f32[8,3] add(q, d)
    ROOT tuple = (f32[2,4], f32[8,3]) tuple(p, r)
  }
  )";

  MemorySpaceAssignment::BufferIntervalCompare buffer_interval_compare =
      [](const MemorySpaceAssignment::BufferInterval& a,
         const MemorySpaceAssignment::BufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kSin:
              return 0;
            case HloOpcode::kCos:
              return 1;
            case HloOpcode::kTanh:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.is_use_allowed_in_alternate_mem_fn = [](const HloUse& use) {
    return use.instruction->opcode() != HloOpcode::kTanh;
  };
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    buffer_interval_compare, &prefetch_interval_picker,
                    options);
}

TEST_P(MemorySpaceAssignmentTest, DisallowedUseBugInWhile) {
  // Test for situations where we disallow a use (tanh in this case) in the
  // alternate memory space and there is a subsequent use that also requires the
  // buffer to be in the default memory space. In this case, the allocation in
  // the default memory space might not be the very last one, so we need to
  // search the allocation sequence and find the one in the default memory
  // space.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=3
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = f32[3]{0} get-tuple-element(p0), index=2
    gte3 = pred[] get-tuple-element(p0), index=3
    add = f32[3]{0} add(gte0, gte0)
    negate0 = f32[3]{0} negate(add)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    negate15 = f32[3]{0} negate(gte2)
    tanh = f32[3]{0} tanh(gte2)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(negate14, tanh, gte2, gte3)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(copy0, copy0, copy1, p1)
    while = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    ROOT gte = f32[3]{0} get-tuple-element(while), index=2
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.is_use_allowed_in_alternate_mem_fn = [](const HloUse& use) {
    return use.instruction->opcode() != HloOpcode::kTanh;
  };
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/2,
                    options);
}

TEST_P(MemorySpaceAssignmentTest, AvoidRedundantEvictionInWhile) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    add = f32[3]{0} add(negate14, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[3]{0} get-tuple-element(while), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    // Expect that while{1} is allocated to alternate memory space. Also expect
    // that this buffer is prefetched at the end of the while loop body but is
    // never evicted (since it has a copy in the default memory space).
    const HloInstruction* while_instr = FindInstruction(module.get(), "while");
    EXPECT_EQ(while_instr->shape().tuple_shapes(1).layout().memory_space(),
              kAlternateMemorySpace);
    const HloInstruction* gte1 = FindInstruction(module.get(), "gte1");
    EXPECT_EQ(gte1->user_count(), 1);
    EXPECT_EQ(gte1->users()[0]->opcode(), HloOpcode::kTanh);
    const HloInstruction* while_root =
        while_instr->while_body()->root_instruction();
    EXPECT_THAT(while_root->operand(1),
                op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                              op::GetTupleElement(op::Parameter(0))));
  }
}

TEST_P(MemorySpaceAssignmentTest,
       RedundantEvictionEliminationShouldntAddRedundantParam) {
  // Check that if there wasn't an eviction in the while loop, we don't add the
  // buffer in default memory as an additional parameter to the while loop.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    add = f32[3]{0} add(negate1, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[3]{0} get-tuple-element(while), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect that while tuple shape contains 3 elements like the original.
  const HloInstruction* while_instr = FindInstruction(module.get(), "while");
  EXPECT_EQ(while_instr->shape().tuple_shapes_size(), 3);
}

TEST_P(MemorySpaceAssignmentTest, AvoidRedundantEvictionInNestedWhile) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    add = f32[3]{0} add(negate14, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  while_cond1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT while2 = (f32[3]{0}, f32[3]{0}, pred[]) while(p0), condition=while_cond2, body=while_body2
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while1 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond1, body=while_body1
    gte = f32[3]{0} get-tuple-element(while1), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    // Expect that while1{1} and while2{1} are allocated to alternate memory
    // space. Also expect that this buffer is prefetched at the end of the while
    // loop body but is never evicted (since it has a copy in the default memory
    // space).
    const HloInstruction* while1_instr =
        FindInstruction(module.get(), "while1");
    EXPECT_EQ(while1_instr->shape().tuple_shapes(1).layout().memory_space(),
              kAlternateMemorySpace);
    const HloInstruction* while2_instr =
        FindInstruction(module.get(), "while2");
    EXPECT_EQ(while2_instr->shape().tuple_shapes(1).layout().memory_space(),
              kAlternateMemorySpace);
    const HloInstruction* gte1 = FindInstruction(module.get(), "gte1");
    EXPECT_EQ(gte1->user_count(), 1);
    EXPECT_EQ(gte1->users()[0]->opcode(), HloOpcode::kTanh);
    const HloInstruction* while_root =
        while2_instr->while_body()->root_instruction();
    EXPECT_THAT(while_root->operand(1),
                op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                              op::GetTupleElement(op::Parameter(0))));
  }
}

TEST_P(MemorySpaceAssignmentTest, RedundantEvictionEliminationBug) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    add0 = f32[3]{0} add(negate14, tanh)
    add1 = f32[3]{0} add(add0, gte1)
    negate = f32[3]{0} negate(add1)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add1, negate, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[3]{0} get-tuple-element(while), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect that redundant eviction elimination doesn't kick in because
  // while{1} is updated within the body.
  const HloInstruction* while_instr = FindInstruction(module.get(), "while");
  EXPECT_EQ(while_instr->shape().tuple_shapes_size(), 3);
  if (GetParam()) {
    EXPECT_EQ(while_instr->shape().tuple_shapes(1).layout().memory_space(),
              kAlternateMemorySpace);
    const HloInstruction* gte1 = FindInstruction(module.get(), "gte1");
    EXPECT_EQ(gte1->user_count(), 2);
    EXPECT_NE(
        absl::c_find_if(gte1->users(), HloPredicateIsOp<HloOpcode::kCopyStart>),
        gte1->users().end());
  }
}

TEST_P(MemorySpaceAssignmentTest, RedundantEvictionEliminationInChainedWhile) {
  // Check against a bug where a while HLO feeding to another while HLO can
  // cause a crash if we performa redundant eviction elimination to the first
  // while but not the other (while operand/parameter shapes would mismatch).
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    add = f32[3]{0} add(negate14, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  while_cond2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    add = f32[3]{0} add(negate0, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while1 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond1, body=while_body1
    while2 = (f32[3]{0}, f32[3]{0}, pred[]) while(while1), condition=while_cond2, body=while_body2
    gte = f32[3]{0} get-tuple-element(while2), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    // Expect that while1 has one more value than while2 in its shape.
    EXPECT_EQ(
        FindInstruction(module.get(), "while1")->shape().tuple_shapes_size(),
        FindInstruction(module.get(), "while2")->shape().tuple_shapes_size() +
            1);
  }
}

TEST_P(MemorySpaceAssignmentTest, AvoidRedundantEvictionAfterWhile) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    add = f32[3]{0} add(gte0, gte1)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, add, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    negate0 = f32[3]{0} negate(p0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, negate14, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte0 = f32[3]{0} get-tuple-element(while), index=0
    gte1 = f32[3]{0} get-tuple-element(while), index=1
    negate20 = f32[3]{0} negate(gte1)
    negate21 = f32[3]{0} negate(negate20)
    negate22 = f32[3]{0} negate(negate21)
    negate23 = f32[3]{0} negate(negate22)
    negate24 = f32[3]{0} negate(negate23)
    negate25 = f32[3]{0} negate(negate24)
    negate26 = f32[3]{0} negate(negate25)
    negate27 = f32[3]{0} negate(negate26)
    negate28 = f32[3]{0} negate(negate27)
    negate29 = f32[3]{0} negate(negate28)
    negate30 = f32[3]{0} negate(negate29)
    negate31 = f32[3]{0} negate(negate30)
    negate32 = f32[3]{0} negate(negate31)
    negate33 = f32[3]{0} negate(negate32)
    negate34 = f32[3]{0} negate(negate33)
    ROOT add = f32[3]{0} add(negate34, gte0)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    EXPECT_THAT(
        module->entry_computation()->root_instruction()->operand(1),
        op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, op::Copy()));
  }
}

TEST_P(MemorySpaceAssignmentTest, AvoidRedundantEvictionAfterWhile2) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    add = f32[3]{0} add(gte0, gte1)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, add, gte2)
  }

  while_cond2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    add = f32[3]{0} add(gte0, gte1)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, add, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple1 = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while1 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple1), condition=while_cond1, body=while_body1
    gte0 = f32[3]{0} get-tuple-element(while1), index=0
    gte1 = f32[3]{0} get-tuple-element(while1), index=1
    negate0 = f32[3]{0} negate(gte1)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    tuple2 = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, negate14, p1)
    while2 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple2), condition=while_cond2, body=while_body2
    gte2 = f32[3]{0} get-tuple-element(while2), index=0
    gte3 = f32[3]{0} get-tuple-element(while2), index=1
    negate20 = f32[3]{0} negate(gte3)
    negate21 = f32[3]{0} negate(negate20)
    negate22 = f32[3]{0} negate(negate21)
    negate23 = f32[3]{0} negate(negate22)
    negate24 = f32[3]{0} negate(negate23)
    negate25 = f32[3]{0} negate(negate24)
    negate26 = f32[3]{0} negate(negate25)
    negate27 = f32[3]{0} negate(negate26)
    negate28 = f32[3]{0} negate(negate27)
    negate29 = f32[3]{0} negate(negate28)
    negate30 = f32[3]{0} negate(negate29)
    negate31 = f32[3]{0} negate(negate30)
    negate32 = f32[3]{0} negate(negate31)
    negate33 = f32[3]{0} negate(negate32)
    negate34 = f32[3]{0} negate(negate33)
    ROOT add = f32[3]{0} add(negate34, gte2)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    EXPECT_THAT(
        module->entry_computation()->root_instruction()->operand(1),
        op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                      op::AsyncCopy(kDefaultMemorySpace, kAlternateMemorySpace,
                                    op::GetTupleElement(op::While()))));
  }
}

TEST_P(MemorySpaceAssignmentTest,
       AfterWhileRedundantEarlierEvictionModifiedBuffer) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    add = f32[3]{0} add(gte0, gte1)
    negate = f32[3]{0} negate(gte0)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(negate, add, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    negate0 = f32[3]{0} negate(p0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, negate14, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte0 = f32[3]{0} get-tuple-element(while), index=0
    gte1 = f32[3]{0} get-tuple-element(while), index=1
    negate20 = f32[3]{0} negate(gte1)
    negate21 = f32[3]{0} negate(negate20)
    negate22 = f32[3]{0} negate(negate21)
    negate23 = f32[3]{0} negate(negate22)
    negate24 = f32[3]{0} negate(negate23)
    negate25 = f32[3]{0} negate(negate24)
    negate26 = f32[3]{0} negate(negate25)
    negate27 = f32[3]{0} negate(negate26)
    negate28 = f32[3]{0} negate(negate27)
    negate29 = f32[3]{0} negate(negate28)
    negate30 = f32[3]{0} negate(negate29)
    negate31 = f32[3]{0} negate(negate30)
    negate32 = f32[3]{0} negate(negate31)
    negate33 = f32[3]{0} negate(negate32)
    negate34 = f32[3]{0} negate(negate33)
    ROOT add = f32[3]{0} add(negate34, gte0)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  if (GetParam()) {
    EXPECT_THAT(
        module->entry_computation()->root_instruction()->operand(1),
        op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                      op::AsyncCopy(kDefaultMemorySpace, kAlternateMemorySpace,
                                    op::GetTupleElement(op::While()))));
  }
}

TEST_P(MemorySpaceAssignmentTest, BitcastRoot) {
  // Tests against a bug where the root of entry computation is a bitcast
  // instruction and it ends up getting an allocation in the alternate memory.
  absl::string_view hlo_string = R"(
HloModule primitive_computation_gather.4, is_scheduled=true

%while_body {
  %param.1 = (s32[], f32[3,3,3]) parameter(0)
  %get-tuple-element.32 = s32[] get-tuple-element(%param.1), index=0
  %copy.6 = s32[] copy(s32[] %get-tuple-element.32)
  %constant.8 = s32[] constant(1)
  %add = s32[] add(s32[] %copy.6, s32[] %constant.8)
  %get-tuple-element.35 = f32[3,3,3] get-tuple-element(%param.1), index=1
  negate = f32[3,3,3] negate(get-tuple-element.35)
  ROOT %tuple.10 = (s32[], f32[3,3,3]) tuple(s32[] %add, f32[3,3,3] negate)
}

%while_cond {
  %param.0 = (s32[], f32[3,3,3]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element(%param.0), index=0
  %constant.3 = s32[] constant(3)
  ROOT %compare = pred[] compare(s32[] %get-tuple-element, s32[] %constant.3), direction=LT
}

ENTRY %primitive_computation_gather.4 (parameter.1: f32[3,10,5], parameter.2: s32[3,1]) -> f32[3,3,3] {
  %constant.1 = s32[] constant(0)
  %copy.11 = s32[] copy(s32[] %constant.1)
  %constant = f32[] constant(0)
  %broadcast = f32[3,3,3] broadcast(f32[] %constant), dimensions={}
  %tuple.8 = (s32[], f32[3,3,3]) tuple(s32[] %copy.11, f32[3,3,3] %broadcast)
  %while = (s32[], f32[3,3,3]) while(%tuple.8), condition=%while_cond, body=%while_body
  %get-tuple-element.7 = f32[3,3,3] get-tuple-element(%while), index=1
  ROOT %bitcast.1 = f32[3,3,3] bitcast(f32[3,3,3] %get-tuple-element.7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(!root->shape().has_layout() ||
              root->shape().layout().memory_space() == kDefaultMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, AsyncOpShortLiveRange) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate1 = bf16[4]{0} negate(param)
  negate2 = bf16[4]{0} negate(negate1)
  negate3 = bf16[4]{0} negate(negate2)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate3)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect both the source and destination buffers to get alternate memory
  // allocations.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, AsyncOpShortLiveRangeInputBufferConsumer) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate1 = bf16[4]{0} negate(negate0)
  negate2 = bf16[4]{0} negate(negate1)
  negate3 = bf16[4]{0} negate(negate2)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate3)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect only the destination buffer to get alternate memory allocation
  // because negate0 is also used by negate1.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, AsyncOpLongLiveRange) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate1 = bf16[4]{0} negate(param)
  negate2 = bf16[4]{0} negate(negate1)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  negate5 = bf16[4]{0} negate(negate4)
  negate6 = bf16[4]{0} negate(negate5)
  negate7 = bf16[4]{0} negate(negate6)
  negate8 = bf16[4]{0} negate(negate7)
  negate9 = bf16[4]{0} negate(negate8)
  negate10 = bf16[4]{0} negate(negate9)
  negate11 = bf16[4]{0} negate(negate10)
  negate12 = bf16[4]{0} negate(negate11)
  negate13 = bf16[4]{0} negate(negate12)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate13)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect none of the buffers to get alternate memory allocations because of
  // the long live range.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, AsyncOpLongLiveRangeInputBufferConsumer) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate1 = bf16[4]{0} negate(negate0)
  negate2 = bf16[4]{0} negate(negate1)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  negate5 = bf16[4]{0} negate(negate4)
  negate6 = bf16[4]{0} negate(negate5)
  negate7 = bf16[4]{0} negate(negate6)
  negate8 = bf16[4]{0} negate(negate7)
  negate9 = bf16[4]{0} negate(negate8)
  negate10 = bf16[4]{0} negate(negate9)
  negate11 = bf16[4]{0} negate(negate10)
  negate12 = bf16[4]{0} negate(negate11)
  negate13 = bf16[4]{0} negate(negate12)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate13)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect none of the buffers to get alternate memory allocations because of
  // the long live range and because negate0 is also used by negate1.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, InPlaceAsyncCollectivePermute) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  negate1 = bf16[4]{0} negate(param)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0 = (s32[]) tuple(const0)
  tuple1 = (s32[]) tuple(const1)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0, negate1, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate4)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect both the source and destination buffers to get alternate memory
  // allocations.
  if (GetParam()) {
    HloInstruction* collective_permute_start =
        module->entry_computation()->GetInstructionWithName(
            "collective-permute-start");
    EXPECT_TRUE(collective_permute_start->shape()
                    .tuple_shapes(0)
                    .layout()
                    .memory_space() == kAlternateMemorySpace);
    EXPECT_TRUE(collective_permute_start->shape()
                    .tuple_shapes(1)
                    .layout()
                    .memory_space() == kAlternateMemorySpace);
  }
}

TEST_P(MemorySpaceAssignmentTest, InPlaceAsyncCollectivePermuteSameBuffer) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0 = (s32[]) tuple(const0)
  tuple1 = (s32[]) tuple(const1)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0, negate0, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate4)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect both the source and destination buffers to get alternate memory
  // allocations.
  if (GetParam()) {
    HloInstruction* collective_permute_start =
        module->entry_computation()->GetInstructionWithName(
            "collective-permute-start");
    EXPECT_TRUE(collective_permute_start->shape()
                    .tuple_shapes(0)
                    .layout()
                    .memory_space() == kAlternateMemorySpace);
    EXPECT_TRUE(collective_permute_start->shape()
                    .tuple_shapes(1)
                    .layout()
                    .memory_space() == kAlternateMemorySpace);
  }
}

TEST_P(MemorySpaceAssignmentTest,
       InPlaceAsyncCollectivePermuteSameBufferChained) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0 = (s32[]) tuple(const0)
  tuple1 = (s32[]) tuple(const1)
  collective-permute-start.1 = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0, negate0, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done.1 = bf16[4]{0} collective-permute-done(collective-permute-start.1)
  collective-permute-start.2 = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(collective-permute-done.1, collective-permute-done.1, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate5 = bf16[4]{0} negate(negate4)
  negate6 = bf16[4]{0} negate(negate5)
  negate7 = bf16[4]{0} negate(negate6)
  collective-permute-done.2 = bf16[4]{0} collective-permute-done(collective-permute-start.2)
  ROOT add = add(collective-permute-done.2, negate7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect both the source and destination buffers to get alternate memory
  // allocations.
  if (GetParam()) {
    HloInstruction* collective_permute_start_1 =
        module->entry_computation()->GetInstructionWithName(
            "collective-permute-start.1");
    EXPECT_TRUE(collective_permute_start_1->shape()
                    .tuple_shapes(0)
                    .layout()
                    .memory_space() == kAlternateMemorySpace);
    EXPECT_TRUE(collective_permute_start_1->shape()
                    .tuple_shapes(1)
                    .layout()
                    .memory_space() == kAlternateMemorySpace);
    HloInstruction* collective_permute_start_2 =
        module->entry_computation()->GetInstructionWithName(
            "collective-permute-start.2");
    EXPECT_TRUE(collective_permute_start_2->shape()
                    .tuple_shapes(0)
                    .layout()
                    .memory_space() == kAlternateMemorySpace);
    EXPECT_TRUE(collective_permute_start_2->shape()
                    .tuple_shapes(1)
                    .layout()
                    .memory_space() == kAlternateMemorySpace);
  }
}

TEST_P(MemorySpaceAssignmentTest,
       TupleInPlaceAsyncCollectivePermuteSameBufferChained) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  param2 = bf16[48]{0} parameter(1)
  negate0.1 = bf16[48]{0} negate(param2)
  negate0.2 = bf16[48]{0} negate(param2)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0.0 = (s32[]) tuple(const0)
  tuple0 = ((s32[]), (s32[])) tuple(tuple0.0, tuple0.0)
  tuple1.0 = (s32[]) tuple(const1)
  tuple1 = ((s32[]), (s32[])) tuple(tuple1.0, tuple1.0)
  tuple2 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.1, negate0.2)
  collective-permute-start.1 = ((bf16[48]{0}, bf16[48]{0}), (bf16[48]{0}, bf16[48]{0}), u32[], u32[]) collective-permute-start(tuple2, tuple2, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done.1 = (bf16[48]{0}, bf16[48]{0}) collective-permute-done(collective-permute-start.1)
  collective-permute-start.2 = ((bf16[48]{0}, bf16[48]{0}), (bf16[48]{0}, bf16[48]{0}), u32[], u32[]) collective-permute-start(collective-permute-done.1, collective-permute-done.1, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate5 = bf16[4]{0} negate(negate4)
  negate6 = bf16[4]{0} negate(negate5)
  negate7 = bf16[4]{0} negate(negate6)
  collective-permute-done.2 = (bf16[48]{0}, bf16[48]{0}) collective-permute-done(collective-permute-start.2)
  gte = bf16[48]{0} get-tuple-element(collective-permute-done.2), index=0
  ROOT root = (bf16[48]{0}, bf16[4]{0}) tuple(gte, negate7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* cp_done1 =
      FindInstruction(module.get(), "collective-permute-done.1");
  EXPECT_EQ(cp_done1->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
  const HloInstruction* cp_done2 =
      FindInstruction(module.get(), "collective-permute-done.2");
  EXPECT_EQ(cp_done2->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
}

TEST_P(MemorySpaceAssignmentTest,
       TupleInPlaceAsyncCollectivePermuteSameBuffer) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  param2 = bf16[48]{0} parameter(1)
  negate0.1 = bf16[48]{0} negate(param2)
  negate0.2 = bf16[48]{0} negate(param2)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0.0 = (s32[]) tuple(const0)
  tuple0 = ((s32[]), (s32[])) tuple(tuple0.0, tuple0.0)
  tuple1.0 = (s32[]) tuple(const1)
  tuple1 = ((s32[]), (s32[])) tuple(tuple1.0, tuple1.0)
  tuple2 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.1, negate0.1)
  tuple3 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.2, negate0.2)
  collective-permute-start.1 = ((bf16[48]{0}, bf16[48]{0}), (bf16[48]{0}, bf16[48]{0}), u32[], u32[]) collective-permute-start(tuple2, tuple3, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done.1 = (bf16[48]{0}, bf16[48]{0}) collective-permute-done(collective-permute-start.1)
  gte = bf16[48]{0} get-tuple-element(collective-permute-done.1), index=0
  ROOT root = (bf16[48]{0}, bf16[4]{0}) tuple(gte, negate4)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* cp_done1 =
      FindInstruction(module.get(), "collective-permute-done.1");
  EXPECT_EQ(cp_done1->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
}

TEST_P(MemorySpaceAssignmentTest,
       TupleInPlaceAsyncCollectivePermuteSameBufferRoot) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  param2 = bf16[48]{0} parameter(1)
  negate0.1 = bf16[48]{0} negate(param2)
  negate0.2 = bf16[48]{0} negate(param2)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0.0 = (s32[]) tuple(const0)
  tuple0 = ((s32[]), (s32[])) tuple(tuple0.0, tuple0.0)
  tuple1.0 = (s32[]) tuple(const1)
  tuple1 = ((s32[]), (s32[])) tuple(tuple1.0, tuple1.0)
  tuple2 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.1, negate0.1)
  tuple3 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.2, negate0.2)
  collective-permute-start.1 = ((bf16[48]{0}, bf16[48]{0}), (bf16[48]{0}, bf16[48]{0}), u32[], u32[]) collective-permute-start(tuple2, tuple3, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  ROOT collective-permute-done.1 = (bf16[48]{0}, bf16[48]{0}) collective-permute-done(collective-permute-start.1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* cp_done1 =
      FindInstruction(module.get(), "collective-permute-done.1");
  EXPECT_EQ(cp_done1->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
  ShapeUtil::ForEachSubshape(
      cp_done1->shape(),
      [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        if (subshape.IsArray() && subshape.has_layout()) {
          EXPECT_EQ(subshape.layout().memory_space(), kDefaultMemorySpace);
        }
      });
}

TEST_P(MemorySpaceAssignmentTest, TupleInPlaceAsyncCollectivePermuteRoot) {
  absl::string_view hlo_string = R"(
 HloModule inplace_collective_permute, is_scheduled=true

 ENTRY %inplace_collective_permute {
   %param.0 = u32[8,1,1] parameter(0)
   %constant.1000 = u32[] constant(1000)
   %broadcast.1 = u32[8,1,1] broadcast(u32[] %constant.1000), dimensions={}
   %broadcast.2 = u32[8,1,1] broadcast(u32[] %constant.1000), dimensions={}
   %tuple.input = (u32[8,1,1], u32[8,1,1]) tuple(u32[8,1,1] %param.0, u32[8,1,1] %param.0)
   %tuple.output = (u32[8,1,1], u32[8,1,1]) tuple(u32[8,1,1] %broadcast.1, u32[8,1,1] %broadcast.2)
   %constant.0 = s32[] constant(0)
   %constant.1 = s32[] constant(1)
   %constant.2 = s32[] constant(2)
   %indices.0.0.0 = (s32[], s32[], s32[]) tuple(s32[] %constant.0, s32[] %constant.0, s32[] %constant.0)
   %indices.1.0.0 = (s32[], s32[], s32[]) tuple(s32[] %constant.1, s32[] %constant.0, s32[] %constant.0)
   %indices.2.0.0 = (s32[], s32[], s32[]) tuple(s32[] %constant.2, s32[] %constant.0, s32[] %constant.0)
   %indices.000.100 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple((s32[], s32[], s32[]) %indices.0.0.0, (s32[], s32[], s32[]) %indices.1.0.0)
   %indices.000.200 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple((s32[], s32[], s32[]) %indices.0.0.0, (s32[], s32[], s32[]) %indices.2.0.0)
   %indices.000.0 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple((s32[], s32[], s32[]) %indices.0.0.0, (s32[], s32[], s32[]) %indices.0.0.0)
   %input.indices = (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) tuple(((s32[], s32[], s32[]), (s32[], s32[], s32[])) %indices.000.100, ((s32[], s32[], s32[]), (s32[], s32[], s32[])) %indices.000.0)
   %output.indices = (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) tuple(((s32[], s32[], s32[]), (s32[], s32[], s32[])) %indices.000.100, ((s32[], s32[], s32[]), (s32[], s32[], s32[])) %indices.000.200)
   %collective-permute-start = ((u32[8,1,1], u32[8,1,1]), (u32[8,1,1], u32[8,1,1]), u32[], u32[]) collective-permute-start((u32[8,1,1], u32[8,1,1]) %tuple.input, (u32[8,1,1], u32[8,1,1]) %tuple.output, (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) %input.indices, (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) %output.indices), channel_id=42, source_target_pairs={{0,1},{1,0},{1,0},{0,1}}, slice_sizes={{4},{4},{4},{4}}
   ROOT %collective-permute-done = (u32[8,1,1], u32[8,1,1]) collective-permute-done(((u32[8,1,1], u32[8,1,1]), (u32[8,1,1], u32[8,1,1]), u32[], u32[]) %collective-permute-start)
 }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* cp_done =
      FindInstruction(module.get(), "collective-permute-done");
  EXPECT_EQ(cp_done->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
  ShapeUtil::ForEachSubshape(
      cp_done->shape(),
      [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        if (subshape.IsArray() && subshape.has_layout()) {
          EXPECT_EQ(subshape.layout().memory_space(), kDefaultMemorySpace);
        }
      });
}

TEST_P(MemorySpaceAssignmentTest, ReservedScopedMemory) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[2,4] parameter(0)
  a = f32[2,4] negate(param0)
  b = f32[2,4] negate(a)
  c = f32[2,4] negate(b)
  d = f32[2,4] negate(c)
  e = f32[2,4] negate(d)
  ROOT f = f32[2,4] add(e, b)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  // Make instruction c reserve 64 bytes in the alternate memory. This should
  // prevent both b and c to put their outputs in the alternate memory.
  options.reserved_scoped_memory_fn = [&](const HloInstruction* instruction) {
    if (instruction->name() == "c") {
      return 100;
    }
    return 0;
  };
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/2,
                    options);
  auto get_memory_space = [&](absl::string_view instruction_name) {
    return module->entry_computation()
        ->GetInstructionWithName(instruction_name)
        ->shape()
        .layout()
        .memory_space();
  };
  EXPECT_TRUE(get_memory_space("a") == kAlternateMemorySpace);
  EXPECT_TRUE(get_memory_space("b") == kDefaultMemorySpace);
  EXPECT_TRUE(get_memory_space("c") == kDefaultMemorySpace);
  EXPECT_TRUE(get_memory_space("d") == kAlternateMemorySpace);
  EXPECT_TRUE(get_memory_space("e") == kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, ConstantAllocationFar) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[2,4] parameter(0)
  const = f32[2,4] constant({...})
  a = f32[2,4] negate(param0)
  b = f32[2,4] negate(a)
  c = f32[2,4] negate(b)
  d = f32[2,4] negate(c)
  e = f32[2,4] negate(d)
  ROOT negate = f32[2,4] add(const, e)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  EXPECT_TRUE(module->entry_computation()
                  ->GetInstructionWithName("const")
                  ->shape()
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(module->entry_computation()
                  ->GetInstructionWithName("negate")
                  ->operand(0)
                  ->shape()
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, ConstantAllocationNear) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[2,4] parameter(0)
  a = f32[2,4] negate(param0)
  b = f32[2,4] negate(a)
  c = f32[2,4] negate(b)
  d = f32[2,4] negate(c)
  e = f32[2,4] negate(d)
  const = f32[2,4] constant({...})
  ROOT negate = f32[2,4] add(const, e)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  EXPECT_TRUE(module->entry_computation()
                  ->GetInstructionWithName("const")
                  ->shape()
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(module->entry_computation()
                  ->GetInstructionWithName("negate")
                  ->operand(0)
                  ->shape()
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

// A mock MemorySpaceAssignmentRepacker class that accepst a map of
// (start_time,offset) -> new_offset values. Using this map, the repacker
// repacks the allocations to the new_offset.
class FakeMemorySpaceAssignmentRepacker : public MemorySpaceAssignmentRepacker {
 public:
  explicit FakeMemorySpaceAssignmentRepacker(
      absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t>& repack_map,
      std::function<void(absl::Span<AllocationBlock*>)> check_fun = nullptr,
      bool always_return_modified = false)
      : MemorySpaceAssignmentRepacker(/*max_size=*/128, /*alignment=*/8),
        repack_map_(repack_map),
        check_fun_(check_fun),
        always_return_modified_(always_return_modified) {}

  StatusOr<bool> Repack(absl::Span<AllocationBlock*> allocations) override {
    bool modified = false;
    for (AllocationBlock* block : allocations) {
      absl::flat_hash_set<int64_t> colocations;
      std::string colocations_str;
      for (const AllocationBlock* colocation : block->colocations) {
        absl::StrAppend(&colocations_str, colocation->id, ", ");
        colocations.insert(colocation->id);
      }
      VLOG(1) << "Alloc id: " << block->id << " time: [" << block->start_time
              << ", " << block->end_time << "] size: " << block->size
              << " init offset: " << block->initial_offset << " colocations: {"
              << colocations_str << "}";
      auto it = repack_map_.find({block->start_time, block->initial_offset});
      if (it != repack_map_.end()) {
        modified = true;
        block->offset = it->second;
      } else {
        block->offset = block->initial_offset;
      }
      for (AllocationBlock* colocation : block->colocations) {
        if (it != repack_map_.end()) {
          colocation->offset = it->second;
        } else {
          colocation->offset = colocation->initial_offset;
        }
      }
    }
    if (check_fun_) {
      check_fun_(allocations);
    }

    return always_return_modified_ || modified;
  }

 private:
  // A map from (start_time, offset) to new_offset.
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map_;
  std::function<void(absl::Span<AllocationBlock*>)> check_fun_;
  bool always_return_modified_;
};

TEST_P(MemorySpaceAssignmentTest, Repack) {
  // We initially perform the following allocations at these offsets.
  //
  //    Max memory
  //  -------------------------------------------
  //
  //
  //
  //
  //      +------------+
  //      |     b      |
  //      +------------+
  //  +-------+                 +------------+
  //  |   a   |                 |     n      |
  //  +-------+                 +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  //
  // Next up, we try to allocate the prefetch for m. However due to
  // fragmentation, this won't be possible:
  //
  //    Max memory
  //  -------------------------------------------
  //
  //
  //
  //                +---------+
  //      +------------+      |
  //      |     b   |  |      |
  //      +------------+      |
  //  +-------+     |         | +------------+
  //  |   a   |     |    d    | |     n      |
  //  +-------+     +---------+ +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  //
  // We then call repack to repack the existing allocations which allows us to
  // allocate the prefetch for m:
  //
  //    Max memory
  //  -------------------------------------------
  //                +---------+
  //                |         |
  //                |         |
  //                |         |
  //  +-------+     |         |
  //  |   a   |     |    d    |
  //  +-------+     +---------+
  //      +------------+        +------------+
  //      |      b     |        |     n      |
  //      +------------+        +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[2,4] sine(param1)
    b = f32[2,4] cosine(param1)
    c = f32[8,3] negate(param0)
    j = f32[2,4] negate(a)
    d = f32[8,3] tanh(param0)
    k = f32[2,4] negate(j)
    l = f32[2,4] add(b, k)
    m = f32[8,3] negate(d)
    n = f32[2,4] sine(l)
    o = f32[8,3] negate(m)
    p = f32[2,4] negate(n)
    q = f32[8,3] negate(m)
    ROOT tuple = (f32[2,4], f32[8,3], f32[8,3]) tuple(p, q, o)
  }
  )";

  MemorySpaceAssignment::BufferIntervalCompare buffer_interval_compare =
      [](const MemorySpaceAssignment::BufferInterval& a,
         const MemorySpaceAssignment::BufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kSin:
              return 0;
            case HloOpcode::kCos:
              return 1;
            case HloOpcode::kTanh:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  // Move "a" from offset 0 to 32.
  repack_map[{2, 0}] = 32;
  // Move "b" from offset 32 to 0.
  repack_map[{3, 32}] = 0;
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map);
  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.max_repacks = 1;
  options.repacker = &repacker;
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    buffer_interval_compare, &prefetch_interval_picker,
                    options);

  // If repacking succeeds, we should find the buffer for d in alternate memory.
  const HloInstruction* d =
      module->entry_computation()->GetInstructionWithName("d");
  EXPECT_EQ(d->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, RepackExportsAliasedOffsets) {
  // This test is that we are correctly exporting aliased offsets for repacking.
  // In this example, the buffer produced at HLO "a" will be allocated first,
  // and will consist of four allocations:
  //    1) a produced in the alternate memory (and then evicted to the default
  //    memory). 2) a prefetched to the alternate memory to be used by q and
  //    while HLOs. 3) a used within the while loop body. 4) the output of while
  //    HLO, used by u.
  //
  // Since a will be allocated first (the test is crafted to prioritize sine
  // HLO), all four allocations should get the same (zero) offsets. However,
  // while allocations 2, 3, and 4 need to be colocated with each other,
  // allocation 1 doesn't need to be colocated with the other three.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  while_condition {
    param1 = (f32[2,4], f32[2,4]) parameter(0)
    ROOT cond = pred[] constant(true)
  }

  while_body {
    param2 = (f32[2,4], f32[2,4]) parameter(0)
    gte2 = f32[2,4] get-tuple-element(param2), index=0
    gte3 = f32[2,4] get-tuple-element(param2), index=1
    add = f32[2,4] add(gte2, gte3)
    ROOT tuple2 = (f32[2,4], f32[2,4]) tuple(add, gte3)
  }

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    a = f32[2,4] sine(param0)
    b = f32[2,4] negate(a)
    c = f32[2,4] negate(b)
    d = f32[2,4] negate(c)
    e = f32[2,4] negate(d)
    f = f32[2,4] negate(e)
    g = f32[2,4] negate(f)
    h = f32[2,4] negate(g)
    i = f32[2,4] negate(h)
    j = f32[2,4] negate(i)
    k = f32[2,4] negate(j)
    l = f32[2,4] negate(k)
    m = f32[2,4] negate(l)
    n = f32[2,4] negate(m)
    o = f32[2,4] negate(n)
    p = f32[2,4] negate(o)
    q = f32[2,4] add(p, a)
    tuple = (f32[2,4], f32[2,4]) tuple(q, a)
    while = (f32[2,4], f32[2,4]) while(tuple), condition=while_condition, body=while_body
    gte0 = f32[2,4] get-tuple-element(while), index=0
    gte1 = f32[2,4] get-tuple-element(while), index=1
    r = f32[2,4] negate(gte0)
    s = f32[2,4] negate(r)
    t = f32[2,4] negate(s)
    constant = f32[] constant(0)
    broadcast = f32[8,4] broadcast(constant), dimensions={}
    cos = f32[8,4] cosine(broadcast)
    u = f32[2,4] add(t, gte1)
    v = f32[2,4] add(u, param0)
    w = f32[8,4] negate(cos)
    ROOT tuple3 = (f32[2,4], f32[8,4]) tuple(v, w)
  }
  )";

  MemorySpaceAssignment::BufferIntervalCompare buffer_interval_compare =
      [](const MemorySpaceAssignment::BufferInterval& a,
         const MemorySpaceAssignment::BufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kSin:
              return 0;
            case HloOpcode::kCos:
              return 1;
            case HloOpcode::kTanh:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;

  // Expect that of the four separate allocations for the "a" buffer, the first
  // and the next three are in separate colocations.
  auto check_fun =
      [](absl::Span<MemorySpaceAssignmentRepacker::AllocationBlock*>
             allocations) {
        EXPECT_TRUE(allocations.at(0)->colocations.size() == 1 ||
                    allocations.at(0)->colocations.size() == 3);
        EXPECT_EQ(allocations.at(1)->colocations.size(), 3);
        EXPECT_EQ(allocations.at(2)->colocations.size(), 3);
        EXPECT_TRUE(allocations.at(3)->colocations.size() == 1 ||
                    allocations.at(3)->colocations.size() == 3);
      };
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, check_fun);
  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.max_repacks = 1;
  options.repacker = &repacker;
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    buffer_interval_compare, &prefetch_interval_picker,
                    options);
}

TEST_P(MemorySpaceAssignmentTest,
       RepackExportsAliasedOffsetsForReservedScopedMemory) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[2,4] parameter(0)
  a = f32[2,4] negate(param0)
  b = f32[2,4] negate(a)
  c = f32[2,4] negate(b)
  d = f32[2,4] negate(c)
  e = f32[2,4] negate(d)
  ROOT f = f32[2,4] add(e, b)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.max_repacks = 1;
  // Make two instructions reserve scoped memory.
  options.reserved_scoped_memory_fn = [&](const HloInstruction* instruction) {
    if (instruction->name() == "c" || instruction->name() == "d") {
      return 100;
    }
    return 0;
  };

  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  bool repacker_ran = false;

  // Expect that the first two value to repack has a colocations size of 2,
  // corresponding to the scoped allocations.
  auto check_fun =
      [&](absl::Span<MemorySpaceAssignmentRepacker::AllocationBlock*>
              allocations) {
        EXPECT_EQ(allocations.at(0)->colocations.size(), 2);
        EXPECT_EQ(allocations.at(1)->colocations.size(), 2);
        repacker_ran = true;
      };
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, check_fun);
  options.repacker = &repacker;
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/2,
                    options);
  EXPECT_TRUE(repacker_ran);
}

TEST_P(MemorySpaceAssignmentTest,
       RepackShouldntEraseRequiredAssignmentForConditionalOutput) {
  // This is a test case for b/171040271. Repacks erase the required assignments
  // (since some required assignments are inserted conditionally based on
  // allocation decisions), including the fact that conditional outputs are
  // always required to get assignments in the default memory. After repacking,
  // this required assignment was never added back, causing conditionals to get
  // alternate-memory allocations.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]) parameter(0)
    gte = f32[3] get-tuple-element(p0), index=0
    neg1 = f32[3] negate(gte)
    ROOT tuple1 = (f32[3]) tuple(neg1)
  }

  false_computation {
    p0 = (f32[3]) parameter(0)
    gte = f32[3] get-tuple-element(p0), index=0
    neg2 = f32[3] negate(gte)
    ROOT tuple2 = (f32[3]) tuple(neg2)
  }

  ENTRY entry {
    p0 = f32[3] parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3] copy(p0)
    tuple = (f32[3]) tuple(copy)
    conditional = (f32[3]) conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
    ROOT gte = f32[3] get-tuple-element(conditional), index=0
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, nullptr,
                                        /*always_return_modified=*/true);
  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.max_repacks = 10;
  options.repacker = &repacker;
  options.repack_after_every_allocation = true;
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*buffer_interval_compare=*/{}, &prefetch_interval_picker,
                    options);
}

TEST_P(MemorySpaceAssignmentTest, Determinism) {
  // Run memory space assignment a few times to make sure every time it compiles
  // to the same thing.
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get());
  std::string module_str = module->ToString();

  for (int i = 0; i < 10; ++i) {
    std::unique_ptr<HloModule> other_module = CreateEvictAndPrefetchModule();
    AssignMemorySpace(other_module.get());
    EXPECT_EQ(module_str, other_module->ToString());
  }
}

TEST_P(MemorySpaceAssignmentTest, InPlaceOp) {
  // Tests that in-place ops like DynamicUpdateSlice get the same allocation as
  // its input.
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation {
  param0 = f32[2,3] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast = f32[2,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[2,3] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
}

ENTRY main {
  param = f32[2,3] parameter(0)
  negate = f32[2,3] negate(param)
  fusion = f32[2,3] fusion(negate), kind=kLoop, calls=fused_computation
  ROOT add = f32[2,3] add(fusion, fusion)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(module.get());
  HloInstruction* negate_instruction =
      module->entry_computation()->GetInstructionWithName("negate");
  int64_t negate_offset =
      GetAlternateMemoryOffset(*preset_assignments, negate_instruction);
  HloInstruction* fusion_instruction =
      module->entry_computation()->GetInstructionWithName("fusion");
  int64_t fusion_offset =
      GetAlternateMemoryOffset(*preset_assignments, fusion_instruction);
  // We expect negate and fusion to get the same offsets.
  EXPECT_EQ(negate_offset, fusion_offset);
  const bool allocate_across_sequential_calls = GetParam();
  if (allocate_across_sequential_calls) {
    EXPECT_NE(negate_offset, -1);
  }
}

TEST_P(MemorySpaceAssignmentTest, ConditionalInPlaceOp) {
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation {
  param0 = f32[2,3] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast = f32[2,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[2,3] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
}

true_computation {
  p0 = (f32[2,3]) parameter(0)
  gte = f32[2,3] get-tuple-element(p0), index=0
  ROOT neg1 = f32[2,3] negate(gte)
}

false_computation {
  p0 = (f32[2,3]) parameter(0)
  gte = f32[2,3] get-tuple-element(p0), index=0
  neg2 = f32[2,3] negate(gte)
  ROOT fusion = f32[2,3] fusion(neg2), kind=kLoop, calls=fused_computation
}

ENTRY entry {
  p0 = f32[2,3] parameter(0)
  p1 = pred[] parameter(1)
  copy = f32[2,3] copy(p0)
  tuple = (f32[2,3]) tuple(copy)
  ROOT conditional = f32[2,3] conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_P(MemorySpaceAssignmentTest, AsyncCallDisableAlternateMem) {
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

called_comp {
  p0 = f32[2,3] parameter(0)
  negate10 = f32[2,3] negate(p0)
  negate11 = f32[2,3] negate(negate10)
  negate12 = f32[2,3] negate(negate11)
  negate13 = f32[2,3] negate(negate12)
  negate14 = f32[2,3] negate(negate13)
  ROOT negate15 = f32[2,3] negate(negate14)
}, execution_thread="foobar"

async_comp {
  p0 = f32[2,3] parameter(0)
  ROOT call = f32[2,3] call(p0), to_apply=called_comp
}, execution_thread="foobar"

ENTRY entry {
  p0 = f32[2,3] parameter(0)
  negate0 = f32[2,3] negate(p0)
  negate1 = f32[2,3] negate(negate0)
  negate2 = f32[2,3] negate(negate1)
  negate3 = f32[2,3] negate(negate2)
  negate4 = f32[2,3] negate(negate3)
  async-start = ((f32[2,3]), f32[2,3], f32[2]) async-start(negate1), async_group_id=0, async_execution_thread="foobar", calls=async_comp
  async-done = f32[2,3] async-done(async-start), async_group_id=0, async_execution_thread="foobar", calls=async_comp
  add0 = f32[2,3] add(negate0, async-done)
  negate5 = f32[2,3] negate(add0)
  negate6 = f32[2,3] negate(negate5)
  negate7 = f32[2,3] negate(negate6)
  negate8 = f32[2,3] negate(negate7)
  negate9 = f32[2,3] negate(negate8)
  ROOT add1 = f32[2,3] add(negate9, async-done)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.is_use_allowed_in_alternate_mem_fn = [](const HloUse& use) {
    return use.instruction->opcode() != HloOpcode::kAsyncStart &&
           use.instruction->opcode() != HloOpcode::kAsyncDone &&
           use.instruction->parent()->IsMainThread();
  };
  options.is_position_allowed_in_alternate_mem_fn = [](const HloPosition& pos) {
    return pos.instruction->opcode() != HloOpcode::kAsyncStart &&
           pos.instruction->opcode() != HloOpcode::kAsyncDone &&
           pos.instruction->parent()->IsMainThread();
  };
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/2,
                    options);
  auto has_alternate_memory_allocation =
      [&](const HloInstruction* instruction) {
        bool result = false;
        auto shape_has_alternate_memory_allocation =
            [&](const Shape& subshape, const ShapeIndex& /*index*/) {
              if (subshape.IsArray() &&
                  subshape.layout().memory_space() == kAlternateMemorySpace) {
                result = true;
              }
            };
        ShapeUtil::ForEachSubshape(instruction->shape(),
                                   shape_has_alternate_memory_allocation);
        for (const HloInstruction* operand : instruction->operands()) {
          ShapeUtil::ForEachSubshape(operand->shape(),
                                     shape_has_alternate_memory_allocation);
        }
        return result;
      };

  // Check that the async ops themselves and the instructions inside async
  // computations do not have any alternate memory allocations.
  const HloInstruction* async_start =
      FindInstruction(module.get(), "async-start");
  const HloInstruction* async_done =
      FindInstruction(module.get(), "async-done");
  EXPECT_FALSE(has_alternate_memory_allocation(async_start));
  EXPECT_FALSE(has_alternate_memory_allocation(async_done));
  for (const HloInstruction* instruction :
       async_start->async_wrapped_instruction()
           ->called_computations()[0]
           ->instructions()) {
    EXPECT_FALSE(has_alternate_memory_allocation(instruction));
  }
  // Check that we still allow the tensor used/produced by the async computation
  // to be placed in the alternate memory before/after the async computation.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Add(op::Negate(),
                      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                    op::AsyncDone())));
  EXPECT_THAT(async_start,
              op::AsyncStart(op::AsyncCopy(
                  kDefaultMemorySpace, kAlternateMemorySpace, op::Negate())));
}

INSTANTIATE_TEST_SUITE_P(MemorySpaceAssignmentInstantiation,
                         MemorySpaceAssignmentTest,
                         ::testing::Values(false, true));

using AsynchronousCopyOrderingTest = ::testing::Test;

TEST_F(AsynchronousCopyOrderingTest, Simple) {
  // Given asynchronous copies like the following, ensure the pipelining order
  // is maintained (earlier start time must have earlier end time).
  // 3,11       +-------+         OK
  // 1,8      +------+            OK
  // 5,14         +--------+      OK
  // 7,14           +------+      OK
  // 2,16      +-------------+    Violate
  // 9,12             +--+        Violate
  // 6,17          +----------+   Violate
  // 5,13         +-------+       OK (same start as 5,14)
  // 5,14         +--------+      OK (same as 5,14)
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyOrdering ordering;
  EXPECT_FALSE(ordering.ViolatesOrdering(3, 11));
  ordering.AddCopy({3, 11, 1, alternate_mem_space, 0});
  EXPECT_FALSE(ordering.ViolatesOrdering(1, 8));
  ordering.AddCopy({1, 8, 1, alternate_mem_space, 1});
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 14));
  ordering.AddCopy({5, 14, 1, alternate_mem_space, 2});
  EXPECT_FALSE(ordering.ViolatesOrdering(7, 14));
  ordering.AddCopy({7, 14, 1, alternate_mem_space, 3});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 16));
  EXPECT_TRUE(ordering.ViolatesOrdering(9, 12));
  EXPECT_TRUE(ordering.ViolatesOrdering(6, 17));
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 13));
  ordering.AddCopy({5, 13, 1, alternate_mem_space, 4});
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 14));
  ordering.AddCopy({5, 14, 1, alternate_mem_space, 5});
}

TEST_F(AsynchronousCopyOrderingTest, SameInterval) {
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyOrdering ordering;
  EXPECT_FALSE(ordering.ViolatesOrdering(1, 5));
  EXPECT_FALSE(ordering.ViolatesOrdering(2, 4));
  ordering.AddCopy({1, 5, 1, alternate_mem_space, 0});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.AddCopy({1, 5, 1, alternate_mem_space, 1});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.AddCopy({1, 5, 1, alternate_mem_space, 2});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.RemoveCopy({1, 5, 1, alternate_mem_space, 1});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.RemoveCopy({1, 5, 1, alternate_mem_space, 2});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.RemoveCopy({1, 5, 1, alternate_mem_space, 0});
  EXPECT_FALSE(ordering.ViolatesOrdering(2, 4));
}

using AsynchronousCopyResourceTest = ::testing::Test;

TEST_F(AsynchronousCopyResourceTest, Simple) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 3 1 6 7 1 7 2 2 4
  // -1,3,5    +-----+                OK
  // resource:  0 0 1 6 7 1 7 2 2 4
  //  1,4,4        +---+              OK
  // resource:  0 0 0 3 7 1 7 2 2 4
  //  5,9,10               +-----+
  // resource:  0 0 0 3 7 1 0 0 1 4
  //  4,9,3              +-------+    Violate
  //  4,8,2              +-----+      OK; The 5,9 copy shifts resource to right.
  // resource:  0 0 0 3 7 0 0 0 0 4
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 3.0, 1.0, 6.0, 7.0, 1.0, 7.0, 2.0, 2.0, 4.0});
  EXPECT_TRUE(resource.HasEnoughResource(-1, 3, 5.0));
  resource.AddCopy({-1, 3, 5.0, alternate_mem_space, 0});
  EXPECT_TRUE(resource.HasEnoughResource(1, 4, 4.0));
  resource.AddCopy({1, 4, 4.0, alternate_mem_space, 1});
  EXPECT_TRUE(resource.HasEnoughResource(5, 9, 10.0));
  resource.AddCopy({5, 9, 10.0, alternate_mem_space, 2});
  EXPECT_FALSE(resource.HasEnoughResource(4, 9, 3.0));
  EXPECT_TRUE(resource.HasEnoughResource(4, 8, 2.0));
  resource.AddCopy({4, 8, 2.0, alternate_mem_space, 3});
}

TEST_F(AsynchronousCopyResourceTest, Propagate) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 2 2 2 2 2 2 2 2 2
  // 6,10,2                  +-----+   OK
  // resource:  2 2 2 2 2 2 2 0 2 2
  // 5,9,2                 +-----+     OK
  // resource:  2 2 2 2 2 2 0 0 2 2
  // 4,8,2               +-----+       OK
  // resource:  2 2 2 2 2 0 0 0 2 2
  // 3,7,2             +-----+         OK
  // resource:  2 2 2 2 0 0 0 0 2 2
  // 2,6,2           +-----+           OK
  // resource:  2 2 2 0 0 0 0 0 2 2
  // 1,5,2         +-----+             OK
  // resource:  2 2 0 0 0 0 0 0 2 2
  // 0,4,3       +-----+               OK
  // resource:  2 0 0 0 0 0 0 0 1 2
  // 0,4,3       +-----+               OK
  // resource:  2 0 0 0 0 0 0 0 0 0
  // 0,4,1       +-----+               Violate
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
  EXPECT_TRUE(resource.HasEnoughResource(6, 10, 2.0));
  resource.AddCopy({6, 10, 2.0, alternate_mem_space, 0});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(5, 9, 2.0));
  resource.AddCopy({5, 9, 2.0, alternate_mem_space, 1});
  EXPECT_TRUE(resource.HasEnoughResource(4, 8, 2.0));
  resource.AddCopy({4, 8, 2.0, alternate_mem_space, 2});
  EXPECT_TRUE(resource.HasEnoughResource(3, 7, 2.0));
  resource.AddCopy({3, 7, 2.0, alternate_mem_space, 3});
  EXPECT_TRUE(resource.HasEnoughResource(2, 6, 2.0));
  resource.AddCopy({2, 6, 2.0, alternate_mem_space, 4});
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 2.0));
  resource.AddCopy({1, 5, 2.0, alternate_mem_space, 5});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 3.0));
  resource.AddCopy({0, 4, 3.0, alternate_mem_space, 6});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 3.0));
  resource.AddCopy({0, 4, 3.0, alternate_mem_space, 7});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  EXPECT_FALSE(resource.HasEnoughResource(0, 4, 1.0));
}

TEST_F(AsynchronousCopyResourceTest, CantPropagate) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 2 2 2 2 2 2 2 2 2
  // 5,10,2                +-------+   OK
  // resource:  2 2 2 2 2 2 0 2 2 2
  // 4,7,2               +---+         OK
  // resource:  2 2 2 2 2 0 0 2 2 2
  // 4,8,4               +-----+       OK
  // resource:  2 2 2 2 2 0 0 0 0 2
  // 3,6,4             +---+           Violate
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
  EXPECT_TRUE(resource.HasEnoughResource(5, 10, 2.0));
  resource.AddCopy({5, 10, 2.0, alternate_mem_space, 0});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(4, 7, 2.0));
  resource.AddCopy({4, 7, 2.0, alternate_mem_space, 1});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(4, 8, 4.0));
  resource.AddCopy({4, 8, 4.0, alternate_mem_space, 2});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0}));
  EXPECT_FALSE(resource.HasEnoughResource(3, 6, 4.0));
}

TEST_F(AsynchronousCopyResourceTest, Nested) {
  // time:      0 1 2 3 4
  // resource:  2 2 2 2 2
  // 1,3,2         +-+       OK
  // resource:  2 2 0 2 2
  // 0,4,4       +-----+     Violate
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource({2.0, 2.0, 2.0, 2.0, 2.0});
  EXPECT_TRUE(resource.HasEnoughResource(1, 3, 2.0));
  resource.AddCopy({1, 3, 2.0, alternate_mem_space, 0});
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 2.0, 2.0}));
  EXPECT_FALSE(resource.HasEnoughResource(0, 4, 4.0));
}

TEST_F(AsynchronousCopyResourceTest, Remove) {
  // time:      0 1 2 3 4
  // resource:  2 2 2 2 2
  // add:2,5,2       +---+   OK
  // resource:  2 2 2 0 2
  // add:-1,2,3+---+         OK
  // resource:  0 1 2 0 2
  // add:0,4,4   +-----+     OK
  // resource:  0 0 0 0 1
  // rem:0,4,4   +-----+
  // resource:  0 1 2 0 2
  // rem:2,5,2       +---+
  // resource:  0 1 2 2 2
  // rem:-1,2,3+---+
  // resource:  2 2 2 2 2
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource({2.0, 2.0, 2.0, 2.0, 2.0});
  AsynchronousCopy copy1{2, 5, 2.0, alternate_mem_space, 0};
  AsynchronousCopy copy2{-1, 2, 3.0, alternate_mem_space, 1};
  AsynchronousCopy copy3{0, 4, 4.0, alternate_mem_space, 2};
  EXPECT_TRUE(resource.HasEnoughResource(2, 5, 2.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 0.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(-1, 2, 3.0));
  resource.AddCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 1.0, 2.0, 0.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 4.0));
  resource.AddCopy(copy3);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 0.0, 0.0, 0.0, 1.0}));
  resource.RemoveCopy(copy3);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 1.0, 2.0, 0.0, 2.0}));
  resource.RemoveCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 1.0, 2.0, 2.0, 2.0}));
  resource.RemoveCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0}));
}

TEST_F(AsynchronousCopyResourceTest, NestedRemove) {
  // time:      0 1 2 3 4
  // resource:  2 2 2 2 2
  // add:1,3,2     +-+       OK
  // resource:  2 2 0 2 2
  // add:0,4,4   +-----+     Violate
  // rem:1,3,2     +-+
  // resource:  2 2 2 2 2
  // add:0,4,4   +-----+     OK
  // resource:  2 0 0 2 2
  // add:1,3,2     +-+       Violate
  // rem:0,4,4   +-----+
  // resource:  2 2 2 2 2
  // add:1,3,2     +-+       OK
  // resource:  2 2 0 2 2
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource({2.0, 2.0, 2.0, 2.0, 2.0});
  AsynchronousCopy copy1{1, 3, 2.0, alternate_mem_space, 0};
  AsynchronousCopy copy2{0, 4, 4.0, alternate_mem_space, 1};
  EXPECT_TRUE(resource.HasEnoughResource(1, 3, 2.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 2.0, 2.0}));
  EXPECT_FALSE(resource.HasEnoughResource(0, 4, 4.0));
  resource.RemoveCopy(copy1);
  auto current_resources = resource.GetCurrentResources();
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 4.0));
  resource.AddCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 0.0, 0.0, 2.0, 2.0}));
  EXPECT_FALSE(resource.HasEnoughResource(1, 3, 2.0));
  resource.RemoveCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 3, 2.0));
}

TEST_F(AsynchronousCopyResourceTest, PropagateRemove) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 2 2 2 2 2 2 2 2 2
  // add:6,10,2              +-----+   OK
  // resource:  2 2 2 2 2 2 2 0 2 2
  // add:5,9,2             +-----+     OK
  // resource:  2 2 2 2 2 2 0 0 2 2
  // add:4,8,2           +-----+       OK
  // resource:  2 2 2 2 2 0 0 0 2 2
  // add:3,7,2         +-----+         OK
  // resource:  2 2 2 2 0 0 0 0 2 2
  // add:2,6,2       +-----+           OK
  // resource:  2 2 2 0 0 0 0 0 2 2
  // add:1,5,2     +-----+             OK
  // resource:  2 2 0 0 0 0 0 0 2 2
  // add:0,4,3   +-----+               OK
  // resource:  2 0 0 0 0 0 0 0 1 2
  // add:0,5,3   +-------+             OK
  // resource:  2 0 0 0 0 0 0 0 0 0
  // rem:0,5,3   +-------+
  // resource:  2 0 0 0 0 0 0 0 1 2
  // rem:0,4,3   +-----+
  // resource:  2 2 0 0 0 0 0 0 2 2
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
  EXPECT_TRUE(resource.HasEnoughResource(6, 10, 2.0));
  resource.AddCopy({6, 10, 2.0, alternate_mem_space, 0});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(5, 9, 2.0));
  resource.AddCopy({5, 9, 2.0, alternate_mem_space, 1});
  EXPECT_TRUE(resource.HasEnoughResource(4, 8, 2.0));
  resource.AddCopy({4, 8, 2.0, alternate_mem_space, 2});
  EXPECT_TRUE(resource.HasEnoughResource(3, 7, 2.0));
  resource.AddCopy({3, 7, 2.0, alternate_mem_space, 3});
  EXPECT_TRUE(resource.HasEnoughResource(2, 6, 2.0));
  resource.AddCopy({2, 6, 2.0, alternate_mem_space, 4});
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 2.0));
  resource.AddCopy({1, 5, 2.0, alternate_mem_space, 5});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0}));
  AsynchronousCopy copy1{0, 4, 3.0, alternate_mem_space, 6};
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 3.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 5, 3.0));
  AsynchronousCopy copy2{0, 5, 3.0, alternate_mem_space, 7};
  resource.AddCopy(copy2);
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  resource.RemoveCopy(copy2);
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0}));
  resource.RemoveCopy(copy1);
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0}));
}

TEST_F(AsynchronousCopyResourceTest, StartAtZeroAndRemove) {
  // time:      0 1 2 3 4
  // resource:  0 0 1 1 2
  // add:0,4,2   +-----+     OK
  // resource:  0 0 0 0 2
  // rem:0,4,2   +-----+
  // resource:  0 0 1 1 2
  // add:0,4,2   +-----+     OK
  // resource:  0 0 0 0 2
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource({0.0, 0.0, 1.0, 1.0, 2.0});
  AsynchronousCopy copy1{0, 4, 2.0, alternate_mem_space, 0};
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 2.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 0.0, 0.0, 0.0, 2.0}));
  resource.RemoveCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 0.0, 1.0, 1.0, 2.0}));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 0.0, 0.0, 0.0, 2.0}));
}

TEST_F(AsynchronousCopyResourceTest, OutOfOrderRemovalSameStartTime) {
  // time:      0 1 2 3 4
  // resource:  2 2 2 2 2
  // add:1,3,1     +-+       OK
  // resource:  2 2 1 2 2
  // add:1,4,2     +---+     OK
  // resource:  2 2 0 1 2
  // rem:1,3,1     +-+
  // resource:  2 2 0 2 2
  // add:1,5,1     +-----+   OK
  // resource:  2 2 0 1 2
  // add:1,5,1     +-----+   OK
  // resource:  2 2 0 0 2
  // add:1,5,1     +-----+   OK
  // resource:  2 2 0 0 1
  // add:1,5,1     +-----+   OK
  // resource:  2 2 0 0 0
  // add:1,5,1     +-----+   Violate
  // rem:1,4,2     +---+
  // resource:  2 2 0 0 2
  // rem:1,5,1     +-----+
  // resource:  2 2 0 1 2
  // rem:1,5,1     +-----+
  // resource:  2 2 0 2 2
  // rem:1,5,1     +-----+
  // resource:  2 2 1 2 2
  // rem:1,5,1     +-----+
  // resource:  2 2 2 2 2
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyResource resource({2.0, 2.0, 2.0, 2.0, 2.0});
  AsynchronousCopy copy1{1, 3, 1.0, alternate_mem_space, 0};
  AsynchronousCopy copy2{1, 4, 2.0, alternate_mem_space, 1};
  EXPECT_TRUE(resource.HasEnoughResource(1, 3, 1.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 1.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 4, 2.0));
  resource.AddCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 1.0, 2.0}));
  resource.RemoveCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 2.0, 2.0}));

  AsynchronousCopy copy3{1, 5, 1.0, alternate_mem_space, 2};
  AsynchronousCopy copy4{1, 5, 1.0, alternate_mem_space, 3};
  AsynchronousCopy copy5{1, 5, 1.0, alternate_mem_space, 4};
  AsynchronousCopy copy6{1, 5, 1.0, alternate_mem_space, 5};
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 1.0));
  resource.AddCopy(copy3);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 1.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 1.0));
  resource.AddCopy(copy4);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 0.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 1.0));
  resource.AddCopy(copy5);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 0.0, 1.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 1.0));
  resource.AddCopy(copy6);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 0.0, 0.0}));
  EXPECT_FALSE(resource.HasEnoughResource(1, 5, 1.0));

  resource.RemoveCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 0.0, 2.0}));
  resource.RemoveCopy(copy3);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 1.0, 2.0}));
  resource.RemoveCopy(copy4);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 2.0, 2.0}));
  resource.RemoveCopy(copy5);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 1.0, 2.0, 2.0}));
  resource.RemoveCopy(copy6);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0}));
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));
  }

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0),
                      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                    op::Parameter(1))));
}

TEST_P(MemorySpaceAssignmentTest, MultiCrossProgramPrefetchTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kFirstOutput = 4;
  constexpr int kSecondOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto first_weight_shape = ShapeUtil::MakeShape(F32, {kFeature, kFirstOutput});
  auto second_weight_shape =
      ShapeUtil::MakeShape(F32, {kFirstOutput, kSecondOutput});
  auto intermediate_shape = ShapeUtil::MakeShape(F32, {kBatch, kFirstOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kSecondOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* first_weight = builder.AddInstruction(
      HloInstruction::CreateParameter(1, first_weight_shape, "first_weight"));
  HloInstruction* second_weight = builder.AddInstruction(
      HloInstruction::CreateParameter(2, second_weight_shape, "second_weight"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto first_dot = builder.AddInstruction(
      HloInstruction::CreateDot(intermediate_shape, lhs, first_weight,
                                dot_dnums, DefaultPrecisionConfig(2)));

  auto second_dot = builder.AddInstruction(
      HloInstruction::CreateDot(result_shape, first_dot, second_weight,
                                dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {lhs, first_weight, second_weight, first_dot, second_dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_cross_program_prefetches = -1;
  options.max_size_in_bytes = 256;
  options.alignment_in_bytes = 8;
  options.verify = true;
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/2,
                    options);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 2);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));
  }
  if (cross_program_prefetches.size() > 1) {
    EXPECT_EQ(cross_program_prefetches[1].parameter, 2);
    EXPECT_EQ(cross_program_prefetches[1].index, ShapeIndex({}));
  }

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Dot(op::Dot(op::Parameter(0),
                      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                    op::Parameter(1))),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::Parameter(2))));
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));
  }
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchBitcastTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kOutput, kFeature});
  auto bitcast_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  auto bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(bitcast_shape, rhs));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, bitcast, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {lhs, rhs, bitcast, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));
  }
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchBitcastTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kOutput, kFeature});
  auto bitcast_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));

  auto bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(bitcast_shape, rhs));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, bitcast, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, lhs, rhs, bitcast, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));
  }
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchNestedTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  auto tuple_tuple_shape = ShapeUtil::MakeTupleShape({tuple_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_tuple_shape, "p0"));

  auto gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple_shape, param, 0));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, gte, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, gte, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, gte, lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchUnusedParamTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, rhs_shape, "p0"));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchTooBigTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 8;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchTooBigTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 8;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchFusionTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 2;
  constexpr int kFeature = 2;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder fusion_builder("fusion");
  {
    HloInstruction* lhs = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
    HloInstruction* rhs = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(1, rhs_shape, "rhs"));
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    auto dot = fusion_builder.AddInstruction(HloInstruction::CreateDot(
        result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));
    (void)dot;
  }
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  auto activations = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.0, 1.0}, {2.0, 3.0}})));
  auto weights = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.0, 1.0}, {2.0, 3.0}})));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      result_shape, HloInstruction::FusionKind::kCustom, {activations, weights},
      fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {activations, weights, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchFusionTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 2;
  constexpr int kFeature = 2;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder fusion_builder("fusion");
  {
    HloInstruction* param = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "p0"));
    auto lhs = fusion_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
    auto rhs = fusion_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    auto dot = fusion_builder.AddInstruction(HloInstruction::CreateDot(
        result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));
    (void)dot;
  }
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  auto activations = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.0, 1.0}, {2.0, 3.0}})));
  auto weights = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.0, 1.0}, {2.0, 3.0}})));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({activations, weights}));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      result_shape, HloInstruction::FusionKind::kCustom, {tuple},
      fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {activations, weights, tuple, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchPinnedTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {kFeature, kOutput},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kAlternateMemorySpace);
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.is_allowed_in_alternate_mem_fn = [](const HloValue& value) {
    return true;
  };
  std::unique_ptr<PresetAssignments> preset_assignments = AssignMemorySpace(
      module.get(),
      /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/2, options);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchPinnedTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {kFeature, kOutput},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, kAlternateMemorySpace);
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options;
  options.max_size_in_bytes = 128;
  options.alignment_in_bytes = 8;
  options.verify = true;
  options.is_allowed_in_alternate_mem_fn = [](const HloValue& value) {
    return true;
  };
  std::unique_ptr<PresetAssignments> preset_assignments = AssignMemorySpace(
      module.get(),
      /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/2, options);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramRootDupMayAlias) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true, input_output_alias={ {}: (0, {}, may-alias) }
    ENTRY CrossProgramPrefetch {
      c0 = s32[1,2] constant({{77, 77}})
      c1 = s32[] constant(0)
      p0 = s32[2,2] parameter(0)
      ROOT dup = s32[2,2] dynamic-update-slice(s32[2,2] p0, s32[1,2] c0, s32[] c1, s32[] c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
  EXPECT_THAT(FindInstruction(module.get(), "dup")->operand(0),
              op::Parameter(0));
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramRootDup) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true
    ENTRY CrossProgramPrefetch {
      c0 = s32[1,2] constant({{77, 77}})
      c1 = s32[] constant(0)
      p0 = s32[2,2] parameter(0)
      ROOT dup = s32[2,2] dynamic-update-slice(s32[2,2] p0, s32[1,2] c0, s32[] c1, s32[] c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
  EXPECT_THAT(FindInstruction(module.get(), "dup")->operand(0),
              op::Parameter(0));
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramRootDupDot) {
  // Cross program prefetch since the parameter and the root don't alias.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true
    ENTRY CrossProgramPrefetch {
      c0 = s32[1,2] constant({{77, 77}})
      c1 = s32[] constant(0)
      p0 = s32[2,2] parameter(0)
      p1 = s32[2,2] parameter(1)
      dup = s32[2,2] dynamic-update-slice(s32[2,2] p0, s32[1,2] c0, s32[] c1, s32[] c1)
      ROOT dot = s32[2,2] dot(p1, dup), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_THAT(FindInstruction(module.get(), "dup")->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::Parameter(0)));
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramRootDotMayAlias) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true, input_output_alias={ {}: (0, {}, may-alias) }
    ENTRY CrossProgramPrefetch {
      p0 = s32[2,2] parameter(0)
      p1 = s32[2,2] parameter(1)
      ROOT dot = s32[2,2] dot(p1, p0), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_THAT(FindInstruction(module.get(), "dot")->operand(1),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::Parameter(0)));
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramRootParameter) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true
    ENTRY CrossProgramPrefetch {
      p0 = s32[2,2] parameter(0)
      ROOT bitcast = u32[2,2] bitcast(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchNoReuse) {
  // This test is for checking if the cross-program-prefetched buffer is freed
  // after its last use and there is an end-of-program prefetch.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = f32[8,8]{1,0} parameter(0)
    p1 = f32[8,2]{1,0} parameter(1)
    dot = f32[8,2]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT negate.9 = f32[8,2]{1,0} negate(negate.8)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  LOG(ERROR) << "module: " << module->ToString();
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(1), {});
  // Expect that there are two prefetches that use this value, one is the
  // cross-program prefetch, the other is the end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            1);
  // Also verify that the copy-done for the end-of-program prefetch is the last
  // instruction in schedule.
  const HloInstruction* last_instruction =
      module->schedule()
          .sequence(module->entry_computation())
          .instructions()[module->entry_computation()->instruction_count() - 1];
  EXPECT_THAT(last_instruction, op::CopyDone());
  EXPECT_NE(last_instruction, module->entry_computation()->root_instruction());
  // Cross program prefetch would use offset 0 because that's the first
  // assignment. Since we are freeing the cross-program prefetch buffer, we
  // would also expect to see some of the intermediate computations (one of the
  // negate ops) to also get 0 offset allocations.
  bool has_zero_offset_allocations = false;
  for (auto pos_and_chunk : preset_assignments->chunks()) {
    if (pos_and_chunk.first.instruction->opcode() == HloOpcode::kNegate &&
        pos_and_chunk.second.offset == 0) {
      has_zero_offset_allocations = true;
    }
  }
  EXPECT_TRUE(has_zero_offset_allocations);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchTupleNoReuse) {
  // This test is for checking if the cross-program-prefetched buffer is freed
  // after its last use and there is an end-of-program prefetch.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = (f32[8,8]{1,0}, f32[8,2]{1,0}) parameter(0)
    get-tuple-element = f32[8,8]{1,0} get-tuple-element(p0), index=0
    get-tuple-element.1 = f32[8,2]{1,0} get-tuple-element(p0), index=1
    dot = f32[8,2]{1,0} dot(get-tuple-element, get-tuple-element.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT negate.9 = f32[8,2]{1,0} negate(negate.8)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), /*max_outstanding_async_copies=*/-1,
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(0), {1});
  // Expect that there are two prefetches that use this value, one is the
  // cross-program prefetch, the other is the end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            1);
  // Also verify that the copy-done for the end-of-program prefetch is the last
  // instruction in schedule.
  const HloInstruction* last_instruction =
      module->schedule()
          .sequence(module->entry_computation())
          .instructions()[module->entry_computation()->instruction_count() - 1];
  EXPECT_THAT(last_instruction, op::CopyDone());
  EXPECT_NE(last_instruction, module->entry_computation()->root_instruction());
  // Cross program prefetch would use offset 0 because that's the first
  // assignment. Since we are freeing the cross-program prefetch buffer, we
  // would also expect to see some of the intermediate computations (one of the
  // negate ops) to also get 0 offset allocations.
  bool has_zero_offset_allocations = false;
  for (auto pos_and_chunk : preset_assignments->chunks()) {
    if (pos_and_chunk.first.instruction->opcode() == HloOpcode::kNegate &&
        pos_and_chunk.second.offset == 0) {
      has_zero_offset_allocations = true;
    }
  }
  EXPECT_TRUE(has_zero_offset_allocations);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchReuse) {
  // This tests the scenario that the cross-program-prefetched buffer is used
  // again close to the end of the computation. In this case, it is better not
  // to free the buffer.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = f32[8,8]{1,0} parameter(0)
    p1 = f32[8,2]{1,0} parameter(1)
    dot = f32[8,2]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT dot.2 = f32[2,2]{1,0} dot(negate.8, p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(1), {});
  // Expect that there is one prefetch that use this value, the cross-program
  // prefetch. There shouldn't be an end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchTupleReuse) {
  // This tests the scenario that the cross-program-prefetched buffer is used
  // again close to the end of the computation. In this case, it is better not
  // to free the buffer.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = (f32[8,8]{1,0}, f32[8,2]{1,0}) parameter(0)
    get-tuple-element = f32[8,8]{1,0} get-tuple-element(p0), index=0
    get-tuple-element.1 = f32[8,2]{1,0} get-tuple-element(p0), index=1
    dot = f32[8,2]{1,0} dot(get-tuple-element, get-tuple-element.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT dot.2 = f32[2,2]{1,0} dot(negate.8, get-tuple-element.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
    EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(0), {1});
  // Expect that there is one prefetch that use this value, the cross-program
  // prefetch. There shouldn't be an end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            0);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchBufferUnused) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

%fused_computation {
  %param_0.2 = f32[32]{0} parameter(0)
  %param_1.4 = s32[100]{0} parameter(1)
  %custom-call.1 = s32[100]{0} custom-call(s32[100]{0} %param_1.4), custom_call_target="AssumeGatherIndicesInBound", operand_layout_constraints={s32[100]{0}}
  %slice.1 = s32[32]{0} slice(s32[100]{0} %custom-call.1), slice={[0:32]}
  %reshape.7 = s32[32]{0} reshape(s32[32]{0} %slice.1)
  %transpose.5 = s32[32]{0} transpose(s32[32]{0} %reshape.7), dimensions={0}
  %gather.1 = f32[32]{0} gather(f32[32]{0} %param_0.2, s32[32]{0} %transpose.5), offset_dims={}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1}
  %transpose.4 = f32[32]{0} transpose(f32[32]{0} %gather.1), dimensions={0}
  ROOT %reshape.6 = f32[32]{0} reshape(f32[32]{0} %transpose.4)
}

%i.reduce_sub_computation {
  %rhs = s32[] parameter(1)
  %lhs = s32[] parameter(0)
  ROOT %add = s32[] add(s32[] %lhs, s32[] %rhs)
}

%fused_computation.1 {
  %constant.4 = s32[] constant(0)
  %broadcast.4 = s32[100]{0} broadcast(s32[] %constant.4), dimensions={}
  %param_0.4 = s32[32]{0} parameter(0)
  %pad.1 = s32[100]{0} pad(s32[32]{0} %param_0.4, s32[] %constant.4), padding=0_68
  %constant.3 = s32[] constant(76031)
  %broadcast.3 = s32[100]{0} broadcast(s32[] %constant.3), dimensions={}
  ROOT %clamp.1 = s32[100]{0} clamp(s32[100]{0} %broadcast.4, s32[100]{0} %pad.1, s32[100]{0} %broadcast.3)
}

ENTRY %main {
  %constant = s32[] constant(0)
  %i = s32[32,1]{0,1} parameter(1)
  %o = f32[32]{0} parameter(0)
  %reduce = s32[32]{0} reduce(s32[32,1]{0,1} %i, s32[] %constant), dimensions={1}, to_apply=%i.reduce_sub_computation
  %fusion.1 = s32[100]{0} fusion(s32[32]{0} %reduce), kind=kLoop, calls=%fused_computation.1
  ROOT %fusion = f32[32]{0} fusion(f32[32]{0} %o, s32[100]{0} %fusion.1), kind=kCustom, calls=%fused_computation
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Fusion(op::AsyncCopy(kAlternateMemorySpace,
                                       kDefaultMemorySpace, op::Parameter(0)),
                         op::Fusion()));
}

using CostAnalysisPrefetchIntervalPickerTest = HloTestBase;

TEST_F(CostAnalysisPrefetchIntervalPickerTest, PrefetchIntervalOrder) {
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    a = f32[2,4] negate(param0)
    b = f32[2,4] negate(a)
    c = f32[2,4] negate(b)
    d = f32[2,4] negate(c)
    e = f32[2,4] negate(d)
    f = f32[2,4] negate(e)
    g = f32[2,4] negate(f)
    h = f32[2,4] negate(g)
    i = f32[2,4] negate(h)
    j = f32[2,4] negate(i)
    k = f32[2,4] negate(j)
    l = f32[2,4] negate(k)
    m = f32[2,4] negate(l)
    n = f32[2,4] negate(m)
    o = f32[2,4] negate(n)
    p = f32[2,4] negate(o)
    q = f32[2,4] negate(p)
    r = f32[2,4] negate(q)
    s = f32[2,4] negate(r)
    t = f32[2,4] negate(s)
    u = f32[2,4] negate(t)
    ROOT v = f32[2,4] add(u, param0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis(ShapeSize);
  Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module, options));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/4.0,
      /*mem_size_bytes=*/32);

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  interval_picker.Begin(use, /*start_time=*/0, /*end_time=*/22, std::nullopt);

  // Expect that the first interval is (15, 22), which has elapsed time of 6.0,
  // twice of the async copy elased (3.0). Then we expect that intervals will be
  // visited in alternating increasing and decreasing orders until hitting the
  // min and max async copy overlap ratios, which are the intervals (18, 22)
  // and (9, 22) respectively.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 15);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 16);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 14);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 17);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 13);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 18);  // Min async overlap ratio reached.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 12);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 11);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 10);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 9);  // Max async overlap ratio reached.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_TRUE(interval_picker.Done());

  // Expect that if the time between start_time and end_time is too short, there
  // won't be any available intervals.
  interval_picker.Begin(use, /*start_time=*/19, /*end_time=*/22, std::nullopt);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_TRUE(interval_picker.Done());
}

TEST_F(CostAnalysisPrefetchIntervalPickerTest, PrefetchIntervalOrderWhile) {
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  while_condition {
    param1 = (f32[2,4]) parameter(0)    // 19
    ROOT cond = pred[] constant(true)   // 20
  }

  while_body {
    param2 = (f32[2,4]) parameter(0)    // 21
    gte2 = f32[2,4] get-tuple-element(param2), index=0  // 22
    add = f32[2,4] add(gte2, gte2)      // 23
    ROOT tuple2 = (f32[2,4]) tuple(add) // 24
  }

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)  // 0
    a = f32[2,4] negate(param0)     // 1
    b = f32[2,4] negate(a)          // 2
    c = f32[2,4] negate(b)          // 3
    d = f32[2,4] negate(c)          // 4
    e = f32[2,4] negate(d)          // 5
    f = f32[2,4] negate(e)          // 6
    g = f32[2,4] negate(f)          // 7
    h = f32[2,4] negate(g)          // 8
    i = f32[2,4] negate(h)          // 9
    j = f32[2,4] negate(i)          // 10
    k = f32[2,4] negate(j)          // 11
    l = f32[2,4] negate(k)          // 12
    m = f32[2,4] negate(l)          // 13
    n = f32[2,4] negate(m)          // 14
    o = f32[2,4] negate(n)          // 15
    p = f32[2,4] negate(o)          // 16
    q = f32[2,4] negate(p)          // 17
    tuple = (f32[2,4]) tuple(q)     // 18
    while = (f32[2,4]) while(tuple), condition=while_condition, body=while_body  // 25
    gte1 = f32[2,4] get-tuple-element(while), index=0  // 26
    r = f32[2,4] negate(gte1)       // 27
    s = f32[2,4] negate(r)          // 28
    t = f32[2,4] negate(s)          // 29
    u = f32[2,4] negate(t)          // 30
    ROOT v = f32[2,4] add(u, param0)  // 31
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis(ShapeSize);
  Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module, options));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/12.0,
      /*mem_size_bytes=*/32);

  EXPECT_EQ(cost_analysis->options()
                .xla_tpu_memory_space_assignment_while_execution_count,
            5);
  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  interval_picker.Begin(use, /*start_time=*/0, /*end_time=*/31, std::nullopt);

  // Because there are while loop computations between [19, 24], we ensure that
  // the interval picker avoids this interval.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 25);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 26);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 18);
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 27);  // Min async overlap ratio reached.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_EQ(interval_picker.Next(), 17);  // Max async overlap ratio reached.
  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_TRUE(interval_picker.Done());
}

TEST_F(CostAnalysisPrefetchIntervalPickerTest, NestedWhile) {
  // This test is to check against a bug where we didn't assign
  // while_nest_level_ for while instructions, and defaulting to 0. This could
  // cause the prefetch interval logic to think a nested while instruction is
  // the same level as the outermost computation.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  while_condition.2 {
    param1 = (f32[2,4]) parameter(0)    // 11
    ROOT cond = pred[] constant(true)   // 12
  }

  while_body.2 {
    param2 = (f32[2,4]) parameter(0)    // 13
    gte2 = f32[2,4] get-tuple-element(param2), index=0  // 14
    add = f32[2,4] add(gte2, gte2)      // 15
    ROOT tuple2 = (f32[2,4]) tuple(add) // 16
  }

  while_condition.1 {
    param3 = (f32[2,4]) parameter(0)    // 5
    ROOT cond = pred[] constant(true)   // 6
  }

  while_body.1 {
    param4 = (f32[2,4]) parameter(0)    // 7
    gte1 = f32[2,4] get-tuple-element(param4), index=0  // 8
    add1 = f32[2,4] add(gte1, gte1)     // 9
    tuple1 = (f32[2,4]) tuple(add1)     // 10
    while = (f32[2,4]) while(tuple1), condition=while_condition.2, body=while_body.2  // 17
    gte2 = f32[2,4] get-tuple-element(while), index=0  // 18
    add2 = f32[2,4] add(gte2, gte2)     // 19
    ROOT tuple2 = (f32[2,4]) tuple(add2)  // 20
  }

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)  // 0
    a = f32[2,4] negate(param0)     // 1
    b = f32[2,4] negate(a)          // 2
    c = f32[2,4] negate(b)          // 3
    tuple = (f32[2,4]) tuple(c)     // 4
    while = (f32[2,4]) while(tuple), condition=while_condition.1, body=while_body.1  // 21
    gte1 = f32[2,4] get-tuple-element(while), index=0  // 22
    ROOT root = f32[2,4] add(gte1, param0)  // 23
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis(ShapeSize);
  Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module, options));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/12.0,
      /*mem_size_bytes=*/32);

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  const Shape& shape = root->operand(1)->shape();

  // We expect the root's latest prefetch start time to be before the while loop
  // (logical time 4).
  EXPECT_EQ(interval_picker.LatestPrefetchStartTime(shape, /*start_time=*/0,
                                                    /*end_time=*/23, &use),
            4);
}

TEST_F(CostAnalysisPrefetchIntervalPickerTest, ConsecutiveConditionals) {
  // This is a test for b/170668492, where prefetching for consecutive
  // conditionals can cause the prefetch to start in the conditional's
  // computation.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  true_computation.0 {
    p0 = (f32[3]{0}) parameter(0)                   // 5
    gte = f32[3]{0} get-tuple-element(p0), index=0  // 6
    ROOT neg1 = f32[3]{0} negate(gte)               // 7
  }

  false_computation.0 {
    p0 = (f32[3]{0}) parameter(0)                   // 8
    gte = f32[3]{0} get-tuple-element(p0), index=0  // 9
    ROOT neg2 = f32[3]{0} negate(gte)               // 10
  }

  true_computation.1 {
    p0 = (f32[3]{0}) parameter(0)                   // 12
    gte = f32[3]{0} get-tuple-element(p0), index=0  // 13
    ROOT neg1 = f32[3]{0} negate(gte)               // 14
  }

  false_computation.1 {
    p0 = (f32[3]{0}) parameter(0)                   // 15
    gte = f32[3]{0} get-tuple-element(p0), index=0  // 16
    ROOT neg2 = f32[3]{0} negate(gte)               // 17
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)       // 0
    p1 = f32[3]{0} parameter(1)       // 1
    p2 = pred[] parameter(2)          // 2
    tuple0 = (f32[3]{0}) tuple(p0)    // 3
    tuple1 = (f32[3]{0}) tuple(p1)    // 4
    conditional0 = f32[3]{0} conditional(p2, tuple0, tuple0), true_computation=true_computation.0, false_computation=false_computation.0  // 11
    conditional1 = f32[3]{0} conditional(p2, tuple1, tuple1), true_computation=true_computation.1, false_computation=false_computation.1  // 18
    ROOT tuple2 = (f32[3]{0}, f32[3]{0}) tuple(conditional0, conditional1)  // 19
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis(ShapeSize);
  Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module, options));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/12.0,
      /*mem_size_bytes=*/32);

  LOG(INFO) << module->ToString();

  HloInstruction* conditional1 =
      module->entry_computation()->GetInstructionWithName("conditional1");
  const HloUse use{conditional1, /*operand_number=*/1, /*operand_index=*/{0}};
  const Shape& shape =
      module->entry_computation()->parameter_instruction(0)->shape();

  // Expect that the prefetch to start before conditional0's called
  // computations.
  EXPECT_LT(interval_picker.LatestPrefetchStartTime(shape, /*start_time=*/0,
                                                    /*end_time=*/11, &use),
            5);
}

TEST_F(CostAnalysisPrefetchIntervalPickerTest, EarliestLatestWindowTooSmall) {
  // This tests the scenario where there is an op that takes a long time (tanh
  // in this example) and as a result the earliest and latest times both fall
  // inside this long-running op. In this case, we should still return a valid
  // prefetch interval just before the long-running op.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    negate = f32[2,4] negate(param0)
    tanh = f32[2,4] tanh(param0)
    ROOT add = f32[2,4] add(tanh, negate)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis(ShapeSize);
  Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module, options));
  cost_analysis->SetOverrideForGetInstructionElapsed(
      [](const HloInstruction& hlo) {
        if (hlo.opcode() == HloOpcode::kTanh) {
          return 20.0;
        }
        return 1.0;
      });
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_overlap_to_async_copy_ratio=*/1.0,
      /*preferred_overlap_to_async_copy_ratio=*/2.0,
      /*max_overlap_to_mem_size_async_copy_ratio=*/12.0,
      /*mem_size_bytes=*/32);

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  interval_picker.Begin(use, /*start_time=*/1, /*end_time=*/3, std::nullopt);

  LOG(INFO) << interval_picker.ToDebugString();
  EXPECT_FALSE(interval_picker.Done());
  EXPECT_EQ(interval_picker.Next(), 1);
  EXPECT_TRUE(interval_picker.Done());
}

class MemorySpaceAssignmentCostAnalysisTest : public HloTestBase {
 protected:
  Status Initialize(const HloModule* module,
                    float pipeline_overhead_window_size_mib = 0.0) {
    HloCostAnalysis::Options options;
    options_.alternate_mem_bandwidth_bytes_per_second = 128;
    options_.async_copy_bandwidth_bytes_per_second = 32;
    options_.pipeline_overhead_window_size_mib =
        pipeline_overhead_window_size_mib;
    options.shape_size = ShapeSize;
    options.set_flops_per_second(8);
    options.set_bytes_per_second(32);
    options.set_transcendentals_per_second(16);
    hlo_cost_analysis_ = std::make_unique<HloCostAnalysis>(options);
    TF_RETURN_IF_ERROR(
        module->entry_computation()->Accept(hlo_cost_analysis_.get()));
    TF_ASSIGN_OR_RETURN(cost_analysis_,
                        MemorySpaceAssignmentCostAnalysis::Create(
                            *hlo_cost_analysis_, options_, *module));
    return OkStatus();
  }

  Options options_;
  std::unique_ptr<HloCostAnalysis> hlo_cost_analysis_;
  std::unique_ptr<MemorySpaceAssignmentCostAnalysis> cost_analysis_;
};

TEST_F(MemorySpaceAssignmentCostAnalysisTest, NoPipelineOverhead) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    param1 = f32[2,4] parameter(1)
    ROOT add = f32[2,4] add(param0, param1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(Initialize(module.get()));

  const HloInstruction* add = module->entry_computation()->root_instruction();
  const float expected_compute_elapsed =
      /*num_flops=*/8 / /*flops_per_second=*/8.0;
  LOG(INFO) << "Expected compute elapsed = " << expected_compute_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToCompute(*add),
            expected_compute_elapsed);
  float expected_memory_elapsed =
      /*bytes_accessed=*/(3 * 4 * 8) / /*bytes_per_second=*/32.0;
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(*add),
            expected_memory_elapsed);

  // This HLO is memory-bound.
  EXPECT_EQ(cost_analysis_->GetInstructionElapsed(*add),
            expected_memory_elapsed);
  EXPECT_EQ(
      cost_analysis_->GetInstructionElapsedInAlternateMemory(*add, {}, {}),
      expected_memory_elapsed);

  // Put operand 0 in alternate memory. Still memory bound.
  expected_memory_elapsed =
      (/*bytes_accessed=*/(2 * 4 * 8) / /*bytes_per_second=*/32.0) +
      (/*bytes_accessed=*/(4 * 8) / /*bytes_per_second=*/128.0);
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(*add, {{0, {}}}),
            expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}}, {}),
            expected_memory_elapsed);

  // Put operand 0 and output in alternate memory. Still memory bound.
  expected_memory_elapsed =
      (/*bytes_accessed=*/(4 * 8) / /*bytes_per_second=*/32.0) +
      (/*bytes_accessed=*/(2 * 4 * 8) / /*bytes_per_second=*/128.0);
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(
      cost_analysis_->GetInstructionElapsedDueToMemory(*add, {{0, {}}}, {{}}),
      expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}}, {{}}),
            expected_memory_elapsed);

  // Put everything in alternate memory. We're now compute bound.
  expected_memory_elapsed =
      /*bytes_accessed=*/(3 * 4 * 8) / /*bytes_per_second=*/128.0;
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_compute_elapsed);
}

TEST_F(MemorySpaceAssignmentCostAnalysisTest, PipelineOverhead) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    param1 = f32[2,4] parameter(1)
    ROOT add = f32[2,4] add(param0, param1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Set the window size 64B.
  TF_ASSERT_OK(
      Initialize(module.get(),
                 /*pipeline_overhead_window_size_mib=*/(64.0 / 1024 / 1024)));

  const HloInstruction* add = module->entry_computation()->root_instruction();
  const float expected_compute_elapsed =
      /*num_flops=*/8 / /*flops_per_second=*/8.0;
  LOG(INFO) << "Expected compute elapsed = " << expected_compute_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToCompute(*add),
            expected_compute_elapsed);
  float expected_memory_elapsed =
      /*bytes_accessed=*/(3 * 4 * 8) / /*bytes_per_second=*/32.0;
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(*add),
            expected_memory_elapsed);

  float expected_overhead = expected_compute_elapsed * 2 / 3;
  LOG(INFO) << "Expected overhead = " << expected_overhead;
  EXPECT_EQ(cost_analysis_->GetDefaultMemoryAccessOverhead(*add),
            expected_overhead);
  // This HLO is memory-bound.
  EXPECT_EQ(cost_analysis_->GetInstructionElapsed(*add),
            expected_memory_elapsed + expected_overhead);
  EXPECT_EQ(
      cost_analysis_->GetInstructionElapsedInAlternateMemory(*add, {}, {}),
      expected_memory_elapsed + expected_overhead);

  // Put operand 0 in alternate memory. Still memory bound.
  expected_memory_elapsed =
      (/*bytes_accessed=*/(2 * 4 * 8) / /*bytes_per_second=*/32.0) +
      (/*bytes_accessed=*/(4 * 8) / /*bytes_per_second=*/128.0);
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  EXPECT_EQ(cost_analysis_->GetDefaultMemoryAccessOverhead(*add, {{0, {}}}),
            expected_overhead);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(*add, {{0, {}}}),
            expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}}, {}),
            expected_memory_elapsed + expected_overhead);

  // Put operand 0 and output in alternate memory. Still memory bound.
  expected_memory_elapsed =
      (/*bytes_accessed=*/(4 * 8) / /*bytes_per_second=*/32.0) +
      (/*bytes_accessed=*/(2 * 4 * 8) / /*bytes_per_second=*/128.0);
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  expected_overhead = expected_compute_elapsed / 3;
  LOG(INFO) << "Expected overhead = " << expected_overhead;
  EXPECT_EQ(
      cost_analysis_->GetDefaultMemoryAccessOverhead(*add, {{0, {}}}, {{}}),
      expected_overhead);
  EXPECT_EQ(
      cost_analysis_->GetInstructionElapsedDueToMemory(*add, {{0, {}}}, {{}}),
      expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}}, {{}}),
            expected_memory_elapsed + expected_overhead);

  // Put everything in alternate memory. We're now compute bound.
  expected_memory_elapsed =
      /*bytes_accessed=*/(3 * 4 * 8) / /*bytes_per_second=*/128.0;
  LOG(INFO) << "Expected memory elapsed = " << expected_memory_elapsed;
  expected_overhead = 0;
  LOG(INFO) << "Expected overhead = " << expected_overhead;
  EXPECT_EQ(cost_analysis_->GetDefaultMemoryAccessOverhead(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_overhead);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedDueToMemory(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_memory_elapsed);
  EXPECT_EQ(cost_analysis_->GetInstructionElapsedInAlternateMemory(
                *add, {{0, {}}, {1, {}}}, {{}}),
            expected_compute_elapsed);
}

class MemoryBoundLoopOptimizerTest : public HloTestBase {
 public:
  MemoryBoundLoopOptimizerTest() = default;

 protected:
  const int64_t kAlternateMemorySpace = 1;
  const int64_t kDefaultMemorySpace = 0;

  Status Initialize(const HloModule* module,
                    uint64_t alternate_memory_size = 256) {
    HloCostAnalysis::Options options;
    MemoryBoundLoopOptimizerOptions optimizer_options;
    optimizer_options.set_enabled(true);
    optimizer_options.set_desired_copy_ratio(0.7);
    optimizer_options.set_allow_unsatisfied_fully_pipelined_prefetch(false);
    options_.memory_bound_loop_optimizer_options = optimizer_options;
    options_.alternate_mem_bandwidth_bytes_per_second = 128;
    options_.async_copy_bandwidth_bytes_per_second = 32;
    options_.pipeline_overhead_window_size_mib = 1;
    options.shape_size = ShapeSize;
    options.set_flops_per_second(16);
    options.set_bytes_per_second(32);
    options.set_transcendentals_per_second(16);
    hlo_cost_analysis_ = std::make_unique<HloCostAnalysis>(options);
    TF_RETURN_IF_ERROR(
        module->entry_computation()->Accept(hlo_cost_analysis_.get()));
    TF_ASSIGN_OR_RETURN(cost_analysis_,
                        MemorySpaceAssignmentCostAnalysis::Create(
                            *hlo_cost_analysis_, options_, *module));
    TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module));
    TF_ASSIGN_OR_RETURN(live_range_,
                        HloLiveRange::Run(module->schedule(), *alias_analysis_,
                                          module->entry_computation()));
    return OkStatus();
  }

  StatusOr<MemoryBoundLoopOptimizer*> CreateOptimizer(
      int loop_start, int loop_end, const HloModule* module,
      uint64_t alternate_memory_size = 256) {
    TF_RETURN_IF_ERROR(Initialize(module, alternate_memory_size));
    MemoryBoundLoopOptimizerOptions optimizer_options;
    optimizer_options.set_enabled(true);
    optimizer_options.set_desired_copy_ratio(0.7);
    optimizer_options.set_allow_unsatisfied_fully_pipelined_prefetch(false);
    TF_ASSIGN_OR_RETURN(
        optimizer_,
        MemoryBoundLoopOptimizer::Create(
            loop_start, loop_end, alternate_memory_size, optimizer_options,
            *live_range_, *alias_analysis_, *cost_analysis_, SizeFunction));
    return optimizer_.get();
  }

  StatusOr<std::unique_ptr<HloModule>> ParseAndCreateOptimizer(
      absl::string_view hlo_loop_str, uint64_t alternate_memory_size,
      int& loop_start_idx, MemoryBoundLoopOptimizer** optimizer) {
    int loop_end_idx;
    TF_ASSIGN_OR_RETURN(
        std::string module_str,
        ParseAndCreateModuleString(hlo_loop_str, loop_start_idx, loop_end_idx));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        ParseAndReturnVerifiedModule(module_str));
    TF_ASSIGN_OR_RETURN(
        *optimizer, CreateOptimizer(loop_start_idx, loop_end_idx, module.get(),
                                    alternate_memory_size));
    return std::move(module);
  }

  // Parse a loop string description like the following:
  //  $op0 = f32[1,4] add(f32[1,4] $param0, f32[1,4] $prev_op4)
  //  $op1 = f32[8,4] add(f32[8,4] $param1, f32[8,4] $prev_op3)
  //  $op2 = f32[1,4] add(f32[1,4] $param2, f32[1,4] $op0)
  //  $op3 = f32[8,4] add(f32[8,4] $param3, f32[8,4] $op1)
  //  $op4 = f32[1,4] add(f32[1,4] $param4, f32[1,4] $op2)
  StatusOr<std::string> ParseAndCreateModuleString(
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

  StatusOr<std::unique_ptr<PresetAssignments>> RunMsa(
      HloModule* module, uint64_t alternate_memory_size = 256) {
    options_.max_size_in_bytes = alternate_memory_size;
    options_.alignment_in_bytes = 8;
    options_.verify = true;

    options_.alternate_memory_space = kAlternateMemorySpace;

    if (!cost_analysis_) {
      TF_RETURN_IF_ERROR(Initialize(module, alternate_memory_size));
    }
    MemorySpaceAssignmentCostAnalysis::Cache cache;
    options_.buffer_interval_compare =
        MemorySpaceAssignment::GetMemoryBoundednessBufferIntervalCompare(
            *cost_analysis_, &cache);
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

  Status VerifyMsaEquivalence(HloModule* module) {
    // Create a map indexed by instruction number and operand number.
    absl::flat_hash_map<std::pair<int, int>,
                        const MemorySpaceAssignment::Allocation*>
        allocation_map;
    for (const MemoryBoundLoopOptimizer::LoopValue& value :
         optimizer_->loop_values()) {
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
          TF_RET_CHECK(allocation_map.contains({inst_number, operand_number}));
          const MemorySpaceAssignment::Allocation* allocation =
              allocation_map.at({inst_number, operand_number});
          if (!allocation->is_copy_allocation()) {
            // We don't expect a prefetch here.
            EXPECT_NE(operand->opcode(), HloOpcode::kCopyDone);
            int expected_memory_space =
                allocation->memory_space() ==
                        MemorySpaceAssignment::MemorySpace::kDefault
                    ? kDefaultMemorySpace
                    : kAlternateMemorySpace;
            EXPECT_EQ(operand->shape().layout().memory_space(),
                      expected_memory_space);
          } else {
            EXPECT_EQ(allocation->memory_space(),
                      MemorySpaceAssignment::MemorySpace::kAlternate);
            TF_RET_CHECK(operand->opcode() == HloOpcode::kCopyDone);
            const MemorySpaceAssignment::CopyAllocation* copy_allocation =
                static_cast<const MemorySpaceAssignment::CopyAllocation*>(
                    allocation);
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
    return OkStatus();
  }

 private:
  Options options_;
  std::unique_ptr<HloCostAnalysis> hlo_cost_analysis_;
  std::unique_ptr<MemorySpaceAssignmentCostAnalysis> cost_analysis_;
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
      EXPECT_EQ(allocation->memory_space(),
                MemorySpaceAssignment::MemorySpace::kDefault);
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
  std::vector<const MemorySpaceAssignment::CopyAllocation*> prefetches;
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    if (!loop_value.allocations.empty() &&
        loop_value.allocations.back()->is_copy_allocation()) {
      prefetches.push_back(
          static_cast<const MemorySpaceAssignment::CopyAllocation*>(
              loop_value.allocations.back().get()));
    }
  }
  EXPECT_EQ(prefetches.size(), 3);
  bool seen_overlap = false;
  bool seen_nonoverlap = false;
  for (const MemorySpaceAssignment::CopyAllocation* prefetch : prefetches) {
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
  std::vector<const MemorySpaceAssignment::CopyAllocation*> prefetches;
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    if (!loop_value.allocations.empty() &&
        loop_value.allocations.back()->is_copy_allocation()) {
      prefetches.push_back(
          static_cast<const MemorySpaceAssignment::CopyAllocation*>(
              loop_value.allocations.back().get()));
    }
  }
  EXPECT_EQ(prefetches.size(), 2);
  std::optional<int> expected_op14_copy_start_time;
  for (const MemorySpaceAssignment::CopyAllocation* prefetch : prefetches) {
    const HloUse& use = *prefetch->uses().begin();
    if (use.instruction->name() == "op1") {
      EXPECT_EQ(prefetch->copy_done_schedule_before(), 1);
      EXPECT_GT(prefetch->copy_start_schedule_after(), 1);
      expected_op14_copy_start_time = prefetch->copy_start_schedule_after();
    }
  }
  EXPECT_TRUE(expected_op14_copy_start_time.has_value());
  for (const MemorySpaceAssignment::CopyAllocation* prefetch : prefetches) {
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
  std::vector<const MemorySpaceAssignment::CopyAllocation*> prefetches;
  for (const MemoryBoundLoopOptimizer::LoopValue& loop_value :
       optimizer->loop_values()) {
    if (!loop_value.allocations.empty() &&
        loop_value.allocations.back()->is_copy_allocation()) {
      prefetches.push_back(
          static_cast<const MemorySpaceAssignment::CopyAllocation*>(
              loop_value.allocations.back().get()));
    }
  }
  EXPECT_EQ(prefetches.size(), 3);
  bool seen_overlap = false;
  bool seen_nonoverlap = false;
  for (const MemorySpaceAssignment::CopyAllocation* prefetch : prefetches) {
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

  LOG(INFO) << "Running Optimize";
  optimizer->Optimize();

  LOG(INFO) << "Running MSA";
  TF_ASSERT_OK_AND_ASSIGN(auto preset_assignments,
                          RunMsa(module.get(), /*alternate_memory_size=*/1024));

  TF_ASSERT_OK(VerifyMsaEquivalence(module.get()));
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

  LOG(INFO) << "Running MSA";
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

}  // namespace
}  // namespace xla
