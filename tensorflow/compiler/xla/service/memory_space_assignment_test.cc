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

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

constexpr int64 kPointerSize = 8;
constexpr float kAsyncCopyBandwidth = 100;
constexpr float kAlternateMemBandwidth = 1000;
constexpr float kBytesPerSecond = 100;
constexpr float kFlopsPerSecond = 1000;
constexpr float kTranscendentalsPerSecond = 10;

int64 ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

class MemorySpaceAssignmentTest : public HloTestBase,
                                  public ::testing::WithParamInterface<bool> {
 protected:
  // We use the following two memory space values to describe the default (slow
  // and large) and alternate (fast and small) memory spaces.
  const int64 kDefaultMemorySpace = 0;
  const int64 kAlternateMemorySpace = 1;

  std::unique_ptr<PresetAssignments> AssignMemorySpaceUsingCostAnalysis(
      HloModule* module) {
    HloCostAnalysis hlo_cost_analysis(ShapeSize);
    hlo_cost_analysis.set_flops_per_second(kFlopsPerSecond);
    hlo_cost_analysis.set_bytes_per_second(kBytesPerSecond);
    hlo_cost_analysis.set_transcendentals_per_second(kTranscendentalsPerSecond);
    for (HloComputation* computation : module->MakeNonfusionComputations()) {
      TF_CHECK_OK(computation->Accept(&hlo_cost_analysis));
    }
    auto alias_analysis = HloAliasAnalysis::Run(module).ValueOrDie();
    auto cost_analysis = MemorySpaceAssignmentCostAnalysis::Create(
                             hlo_cost_analysis, kAsyncCopyBandwidth,
                             kAlternateMemBandwidth, *module)
                             .ValueOrDie();
    CostAnalysisPrefetchIntervalPicker prefetch_interval_picker(
        CostAnalysisPrefetchIntervalPicker(
            *cost_analysis, /*min_async_copy_to_overlap_ratio=*/0.8,
            /*max_async_copy_to_overlap_ratio=*/10.0,
            /*preferred_async_copy_to_overlap_ratio=*/1.5));
    return AssignMemorySpace(
        module, /*max_outstanding_async_copies=*/-1,
        MemorySpaceAssignment::GetMemoryBoundednessBufferIntervalCompare(
            *cost_analysis, &cache_),
        &prefetch_interval_picker);
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, int64 max_outstanding_async_copies = -1,
      int64 max_prefetch_interval = 10, int64 min_prefetch_interval = 2,
      absl::optional<MemorySpaceAssignment::Options> options = absl::nullopt) {
    InstructionCountPrefetchIntervalPicker prefetch_interval_picker(
        min_prefetch_interval, max_prefetch_interval);
    return AssignMemorySpace(module, max_outstanding_async_copies,
                             /*buffer_interval_compare=*/{},
                             &prefetch_interval_picker, options);
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, int64 max_outstanding_async_copies,
      absl::optional<MemorySpaceAssignment::BufferIntervalCompare>
          buffer_interval_compare,
      PrefetchIntervalPicker* prefetch_interval_picker,
      absl::optional<MemorySpaceAssignment::Options>
          memory_space_assignment_options = absl::nullopt) {
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

    MemorySpaceAssignment::Options options;
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
    options.is_allowed_in_alternate_mem_fn = is_allowed_in_alternate_mem;
    options.max_outstanding_prefetches = max_outstanding_async_copies;
    options.max_outstanding_evictions = max_outstanding_async_copies;
    options.allocate_across_sequential_calls = GetParam();

    auto alias_analysis = HloAliasAnalysis::Run(module).ValueOrDie();
    std::unique_ptr<HloLiveRange> hlo_live_range =
        HloLiveRange::Run(module->schedule(), *alias_analysis,
                          module->entry_computation())
            .ValueOrDie();

    std::unique_ptr<PresetAssignments> preset_assignments =
        MemorySpaceAssignment::Run(module, *hlo_live_range, *alias_analysis,
                                   options)
            .ValueOrDie();
    if (check_parameters_in_default_memory) {
      CheckParametersInDefaultMemory(module);
    }
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

  struct OutstandingAsyncCopies {
    int64 max_copies;
    int64 max_prefetches;
    int64 max_evictions;
  };

  /*static*/ OutstandingAsyncCopies CountMaximumOutstandingAsyncCopies(
      const HloModule& module) {
    OutstandingAsyncCopies copies{0, 0, 0};
    int64 current_copies = 0;
    int64 current_prefetches = 0;
    int64 current_evictions = 0;
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

  int64 GetAlternateMemoryOffset(const PresetAssignments& preset_assignments,
                                 const HloInstruction* instruction,
                                 const ShapeIndex& index = {}) const {
    // Returns the offset of the assignment, -1 if it's not in the alternate
    // memory.
    const HloModule* module = instruction->parent()->parent();
    auto alias_analysis = HloAliasAnalysis::Run(module).ValueOrDie();
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
  Create(const HloCostAnalysis& cost_analysis, const HloModule& module) {
    TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(&module));
    TF_ASSIGN_OR_RETURN(auto hlo_live_range,
                        HloLiveRange::Run(module.schedule(), *alias_analysis,
                                          module.entry_computation()));
    auto call_graph = CallGraph::Build(&module);
    return absl::WrapUnique(new FakeMemorySpaceAssignmentCostAnalysis(
        cost_analysis, /*async_copy_bandwidth_bytes_per_second=*/1,
        /*alternate_mem_bandwidth_bytes_per_second=*/1,
        std::move(alias_analysis), std::move(hlo_live_range),
        std::move(call_graph)));
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
      absl::optional<int64> operand_in_alternate_mem,
      bool output_in_alternate_mem) const override {
    if (get_instruction_elapsed_in_alternate_memory_override_) {
      return get_instruction_elapsed_in_alternate_memory_override_(
          instruction, operand_in_alternate_mem, output_in_alternate_mem);
    }
    if (operand_in_alternate_mem) {
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
      std::function<float(const HloInstruction&, absl::optional<int64>, bool)>
          function) {
    get_instruction_elapsed_in_alternate_memory_override_ = function;
  }
  void SetOverrideForGetAsyncCopyElapsed(
      std::function<float(const Shape&)> function) {
    get_async_copy_elapsed_override_ = function;
  }

 protected:
  FakeMemorySpaceAssignmentCostAnalysis(
      const HloCostAnalysis& cost_analysis,
      float async_copy_bandwidth_bytes_per_second,
      float alternate_mem_bandwidth_bytes_per_second,
      std::unique_ptr<HloAliasAnalysis> alias_analysis,
      std::unique_ptr<HloLiveRange> hlo_live_range,
      std::unique_ptr<CallGraph> call_graph)
      : MemorySpaceAssignmentCostAnalysis(
            cost_analysis, async_copy_bandwidth_bytes_per_second,
            alternate_mem_bandwidth_bytes_per_second, std::move(alias_analysis),
            std::move(hlo_live_range), std::move(call_graph)) {}

 private:
  std::function<float(const HloInstruction&)>
      get_instruction_elapsed_override_ = nullptr;
  std::function<float(const HloInstruction&, absl::optional<int64>, bool)>
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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

TEST_P(MemorySpaceAssignmentTest, TupleSelect) {
  // Make sure tuple-select is not optimized away.
  absl::string_view hlo_string = R"(
  HloModule tuple, is_scheduled=true

  ENTRY %main (a: f32[2], b: f32[2], c: f32[2], d: f32[2], cond: pred[]) -> f32[2] {
    %cond = pred[]{:T(128)E(32)} parameter(4)
    %token0 = token[] after-all()
    %d = f32[2]{0:T(128)} parameter(3)
    %c = f32[2]{0:T(128)} parameter(2)
    %b = f32[2]{0:T(128)} parameter(1)
    %a = f32[2]{0:T(128)} parameter(0)
    %tup0 = (f32[2]{0:T(128)}, f32[2]{0:T(128)}) tuple(f32[2]{0:T(128)} %a, f32[2]{0:T(128)} %b)
    %tup1 = (f32[2]{0:T(128)}, f32[2]{0:T(128)}) tuple(f32[2]{0:T(128)} %c, f32[2]{0:T(128)} %d)
    %s = (f32[2]{0:T(128)}, f32[2]{0:T(128)}) tuple-select(pred[]{:T(128)E(32)} %cond, (f32[2]{0:T(128)}, f32[2]{0:T(128)}) %tup0, (f32[2]{0:T(128)}, f32[2]{0:T(128)}) %tup1)
    %gte = f32[2]{0:T(128)} get-tuple-element((f32[2]{0:T(128)}, f32[2]{0:T(128)}) %s), index=0
    ROOT %negate = f32[2]{0:T(128)} negate(f32[2]{0:T(128)} %gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Negate(op::GetTupleElement(op::TupleSelect())));
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
    ROOT %compare.16 = pred[]{:T(128)E(32)} compare(s32[]{:T(128)} %constant.15, s32[]{:T(128)} %bitcast), direction=GT
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
  // Expect the output to have default memory space.
  EXPECT_EQ(module->entry_computation()
                ->root_instruction()
                ->shape()
                .layout()
                .memory_space(),
            kDefaultMemorySpace);
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

TEST_P(MemorySpaceAssignmentTest, CopyOrdering) {
  // Test to make sure the CopyStarts follow the same CopyDone order. The shapes
  // are picked in increasing order to exploit the fact that heap simulator
  // processes larger tensors first. This checks the ability of the compiler to
  // reschedule:
  //
  //  CS1            CD1
  //   +--------------+
  //    +-----------+
  //   CS2         CD2
  //
  // into:
  //
  //    CS1          CD1
  //     +------------+
  //    +-----------+
  //   CS2         CD2
  HloComputation::Builder builder(TestName());
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 1});
  Shape shape2 = ShapeUtil::MakeShape(F32, {2, 2});
  Shape shape3 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape4 = ShapeUtil::MakeShape(F32, {2, 4});
  PaddingConfig padding_config = MakeEdgePaddingConfig({{0, 0}, {0, 1}});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape3, shape4});
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* p4 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape4, p0, 1));
  HloInstruction* p3 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape3, p0, 0));
  HloInstruction* p2 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape2, "p2"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape1, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, p1));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate5));
  HloInstruction* padding_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(F32)));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, negate6, p1));
  HloInstruction* padded_add1 = builder.AddInstruction(
      HloInstruction::CreatePad(shape2, add1, padding_value, padding_config));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape2, HloOpcode::kAdd, padded_add1, p2));
  HloInstruction* padded_add2 = builder.AddInstruction(
      HloInstruction::CreatePad(shape3, add2, padding_value, padding_config));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape4, HloOpcode::kNegate, p4));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape3, HloOpcode::kAdd, padded_add2, p3));
  HloInstruction* padded_add3 = builder.AddInstruction(
      HloInstruction::CreatePad(shape4, add3, padding_value, padding_config));
  HloInstruction* add4 = builder.AddInstruction(HloInstruction::CreateBinary(
      shape4, HloOpcode::kAdd, padded_add3, negate7));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0,
                                      p4,
                                      p3,
                                      p2,
                                      p1,
                                      negate0,
                                      negate1,
                                      negate2,
                                      negate3,
                                      negate4,
                                      negate5,
                                      negate6,
                                      padding_value,
                                      add1,
                                      padded_add1,
                                      add2,
                                      padded_add2,
                                      negate7,
                                      add3,
                                      padded_add3,
                                      add4});
  TF_CHECK_OK(module->set_schedule(schedule));

  // Use a large max prefetch interval to force CopyStart/CopyDone right after
  // the parameters.
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/50);

  // Iterate over the schedule to make sure CopyStart order and the
  // corresponding CopyDone order match.
  std::list<const HloInstruction*> copy_starts;
  for (HloInstruction* instruction : module->schedule()
                                         .sequence(module->entry_computation())
                                         .instructions()) {
    if (instruction->opcode() == HloOpcode::kCopyStart) {
      copy_starts.push_back(instruction);
    }
    if (instruction->opcode() == HloOpcode::kCopyDone) {
      EXPECT_EQ(copy_starts.front(), instruction->operand(0));
      copy_starts.pop_front();
    }
  }
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
          /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
          kAlternateMemorySpace);
  // Index {1} is a scalar, so it is always placed in the default memory.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {1})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{}, /*tiles=*/{}, /*element_size_in_bits=*/0,
          kDefaultMemorySpace);
  // Index {2} of the while loop is placed in the default memory.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {2})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {4, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  // Expect only negates to be in alternate memory space. Not all might fit but
  // make sure at least one does.
  std::vector<HloInstruction*> negate_instructions = {negate0, negate1, negate2,
                                                      negate3, negate4};
  int64 num_negates_in_alternate_mem = absl::c_count_if(
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
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {4, 6},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  Shape s32_in_default_mem = ShapeUtil::MakeShapeWithLayout(
      xla::S32, {},
      /*minor_to_major=*/{}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  Shape f32v1_in_default_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {1},
      /*minor_to_major=*/{0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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

  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get());

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

TEST_P(MemorySpaceAssignmentTest, MoveCopyDoneEarlier) {
  // This tests the case where an earlier placed smaller buffer may block a
  // larger buffer due to asynchronous copy ordering. The smaller buffer (the
  // operand of sin) will be placed first. The cos, whose operand is 3 times
  // larger than sin's, needs longer time for the asynhronous copy. The cos is
  // placed right after sin, leading to a copy ordering violation:
  //
  // param1------------------>CS----->CD->sin
  // param0------------->CS------------------->CD->cos
  //
  // To fix this, we need to move copy done for cos earlier and ensure both of
  // these buffers get alternate memory allocations:
  //
  // param1------------------>CS----->CD->sin
  // param0-->CS------------------->CD------------>cos
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[2,4] negate(param1)
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
    sin = f32[2,4] sine(param1)
    o = f32[2,4] negate(n)
    cos = f32[8,3] cosine(param0)
    ROOT tuple = (f32[8,3], f32[2,4], f32[2,4]) tuple(cos, sin, o)
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

        auto get_user_priority = [&](const HloValue& value) {
          int priority = INT_MAX;
          for (const auto& use : value.uses()) {
            priority = std::min(priority,
                                get_opcode_priority(use.instruction->opcode()));
          }
          return priority;
        };

        return get_user_priority(*a.buffer) < get_user_priority(*b.buffer);
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloCostAnalysis hlo_cost_analysis(ShapeSize);
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module));
  cost_analysis->SetOverrideForGetAsyncCopyElapsed([](const Shape& shape) {
    // This should return 2 for f32[2,4] and 6 for f32[8,3].
    return ShapeSize(shape) / 16;
  });
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_async_copy_to_overlap_ratio=*/1.0,
      /*max_async_copy_to_overlap_ratio=*/4.0,
      /*preferred_async_copy_to_overlap_ratio=*/1.5);
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    buffer_interval_compare, &interval_picker);

  // Check that both cos and sin could get their operands prefetched.
  const HloInstruction* cos =
      module->entry_computation()->GetInstructionWithName("cos");
  const HloInstruction* sin =
      module->entry_computation()->GetInstructionWithName("sin");
  EXPECT_THAT(sin->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::Parameter(1)));
  EXPECT_THAT(cos->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::Parameter(0)));

  // Sanity check that the cos' operand copy-done is scheduled earlier than
  // sin's operand.
  auto find_schedule_index = [&](const HloInstruction* instruction) {
    const auto& instructions =
        module->schedule().sequence(module->entry_computation()).instructions();
    for (int i = 0; i < instructions.size(); ++i) {
      if (instruction == instructions[i]) {
        return i;
      }
    }
    CHECK(false);
    return -1;
  };
  EXPECT_GT(find_schedule_index(sin->operand(0)),
            find_schedule_index(cos->operand(0)));
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
  MemorySpaceAssignment::Options options;
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
  MemorySpaceAssignment::Options options;
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
  %tuple.8 = (s32[], f32[3,10,5], s32[3,1], f32[3,3,3]) tuple(s32[] %copy.11, f32[3,3,3] %broadcast)
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

// A mock MemorySpaceAssignmentRepacker class that accepst a map of
// (start_time,offset) -> new_offset values. Using this map, the repacker
// repacks the allocations to the new_offset.
class FakeMemorySpaceAssignmentRepacker : public MemorySpaceAssignmentRepacker {
 public:
  explicit FakeMemorySpaceAssignmentRepacker(
      absl::flat_hash_map<std::pair<int64, int64>, int64>& repack_map,
      std::function<void(absl::Span<AllocationBlock*>)> check_fun = nullptr,
      bool always_return_modified = false)
      : MemorySpaceAssignmentRepacker(/*max_size=*/128, /*alignment=*/8),
        repack_map_(repack_map),
        check_fun_(check_fun),
        always_return_modified_(always_return_modified) {}

  StatusOr<bool> Repack(absl::Span<AllocationBlock*> allocations) override {
    bool modified = false;
    for (AllocationBlock* block : allocations) {
      absl::flat_hash_set<int64> colocations;
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
  absl::flat_hash_map<std::pair<int64, int64>, int64> repack_map_;
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
  absl::flat_hash_map<std::pair<int64, int64>, int64> repack_map;
  // Move "a" from offset 0 to 32.
  repack_map[{2, 0}] = 32;
  // Move "b" from offset 32 to 0.
  repack_map[{3, 32}] = 0;
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map);
  MemorySpaceAssignment::Options options;
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
  absl::flat_hash_map<std::pair<int64, int64>, int64> repack_map;

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
  MemorySpaceAssignment::Options options;
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
  absl::flat_hash_map<std::pair<int64, int64>, int64> repack_map;
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, nullptr,
                                        /*always_return_modified=*/true);
  MemorySpaceAssignment::Options options;
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
  // Make sure the root of the entry computation is in the default memory space.
  EXPECT_EQ(module->entry_computation()
                ->root_instruction()
                ->shape()
                .layout()
                .memory_space(),
            kDefaultMemorySpace);
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
  int64 negate_offset =
      GetAlternateMemoryOffset(*preset_assignments, negate_instruction);
  HloInstruction* fusion_instruction =
      module->entry_computation()->GetInstructionWithName("fusion");
  int64 fusion_offset =
      GetAlternateMemoryOffset(*preset_assignments, fusion_instruction);
  // We expect negate and fusion to get the same offsets.
  EXPECT_EQ(negate_offset, fusion_offset);
  const bool allocate_across_sequential_calls = GetParam();
  if (allocate_across_sequential_calls) {
    EXPECT_NE(negate_offset, -1);
  }
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
  ordering.AddCopy({3, 11, alternate_mem_space});
  EXPECT_FALSE(ordering.ViolatesOrdering(1, 8));
  ordering.AddCopy({1, 8, alternate_mem_space});
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 14));
  ordering.AddCopy({5, 14, alternate_mem_space});
  EXPECT_FALSE(ordering.ViolatesOrdering(7, 14));
  ordering.AddCopy({7, 14, alternate_mem_space});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 16));
  EXPECT_TRUE(ordering.ViolatesOrdering(9, 12));
  EXPECT_TRUE(ordering.ViolatesOrdering(6, 17));
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 13));
  ordering.AddCopy({5, 13, alternate_mem_space});
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 14));
  ordering.AddCopy({5, 14, alternate_mem_space});
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchTest) {
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
    EXPECT_EQ(cross_program_prefetches[0].first, 0);
    EXPECT_EQ(cross_program_prefetches[0].second, ShapeIndex({1}));
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
    EXPECT_EQ(cross_program_prefetches[0].first, 0);
    EXPECT_EQ(cross_program_prefetches[0].second, ShapeIndex({1}));
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
  auto rhs_shape = ShapeUtil::MakeShapeWithLayout(
      F32, {kFeature, kOutput},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchReuse) {
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

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  if (!cross_program_prefetches.empty()) {
    EXPECT_EQ(cross_program_prefetches[0].first, 0);
    EXPECT_EQ(cross_program_prefetches[0].second, ShapeIndex({1}));
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
           use.instruction->is_cross_program_prefetch();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.uses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->is_cross_program_prefetch();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.uses(),
                             is_end_of_program_prefetch),
            1);
}

TEST_P(MemorySpaceAssignmentTest, CrossProgramPrefetchNoReuse) {
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
    EXPECT_EQ(cross_program_prefetches[0].first, 0);
    EXPECT_EQ(cross_program_prefetches[0].second, ShapeIndex({1}));
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
           use.instruction->is_cross_program_prefetch();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.uses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->is_cross_program_prefetch();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.uses(),
                             is_end_of_program_prefetch),
            0);
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
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_async_copy_to_overlap_ratio=*/1.0,
      /*max_async_copy_to_overlap_ratio=*/4.0,
      /*preferred_async_copy_to_overlap_ratio=*/2.0);

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  interval_picker.Begin(use, /*start_time=*/0, /*end_time=*/22);

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
  interval_picker.Begin(use, /*start_time=*/19, /*end_time=*/22);
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
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_async_copy_to_overlap_ratio=*/1.0,
      /*max_async_copy_to_overlap_ratio=*/12.0,
      /*preferred_async_copy_to_overlap_ratio=*/2.0);

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloUse use{root, /*operand_number=*/1, /*operand_index=*/{}};
  interval_picker.Begin(use, /*start_time=*/0, /*end_time=*/31);

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
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_async_copy_to_overlap_ratio=*/1.0,
      /*max_async_copy_to_overlap_ratio=*/12.0,
      /*preferred_async_copy_to_overlap_ratio=*/2.0);

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
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeMemorySpaceAssignmentCostAnalysis::Create(
                              hlo_cost_analysis, *module));
  CostAnalysisPrefetchIntervalPicker interval_picker(
      *cost_analysis,
      /*min_async_copy_to_overlap_ratio=*/1.0,
      /*max_async_copy_to_overlap_ratio=*/12.0,
      /*preferred_async_copy_to_overlap_ratio=*/2.0);

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

}  // namespace
}  // namespace xla
