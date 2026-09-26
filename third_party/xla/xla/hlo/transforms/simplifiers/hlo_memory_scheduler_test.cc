/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/memory_scheduler_metrics.pb.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class HloSchedulingTest : public HloHardwareIndependentTestBase {
 protected:
  AliasInfo alias_info_;
};

int64_t PeakMemoryUseOfEntryComputation(
    HloModule* module,
    const LogicalBuffer::SizeFunction* absl_nonnull size_function) {
  CHECK(module->has_entry_computation());
  CHECK(module->has_schedule());

  AliasInfo alias_info;
  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module, &alias_info).value();

  const HloSchedule& schedule = module->schedule();

  HloComputation* computation = module->entry_computation();
  const HloInstructionSequence& sequence = schedule.sequence(computation);
  return HeapSimulator::Run(
             std::make_unique<NoFragmentationStatsHeap<HloValue>>(),
             *computation, sequence, *alias_analysis, &alias_info,
             size_function)
      .value()
      .heap_size;
}

TEST_F(HloSchedulingTest, LastUseScheduledFirst) {
  // Tests scheduling of the following HLO code:
  //
  //   %ab = abs(%param)
  //   %exp = exp(%param)
  //   %add = add(%ab, %exp)
  //   %negate = negate(%exp)
  //   %sub = subtract(%add, %negate)
  //
  // %add should be scheduled before %negate because %add is the last (and only)
  // use of %ab. Scheduling %add first then frees up %ab's buffer.
  const Shape vec = ShapeUtil::MakeShape(xla::F32, {42});
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec, "param"));
  auto ab = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kAbs, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kExp, param));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kAdd, ab, exp));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kNegate, exp));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kSubtract, add, negate));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloMemoryScheduler scheduler(&alias_info_, [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape());
  });
  ASSERT_FALSE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, scheduler.Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());

  // Verify that all instructions are in the sequence.
  const std::vector<HloInstruction*>& sequence =
      module->schedule().sequence(module->entry_computation()).instructions();
  EXPECT_EQ(module->entry_computation()->instruction_count(), sequence.size());

  // The first instruction should be the parameter and the last the root "sub".
  EXPECT_EQ(param, sequence.front());
  EXPECT_EQ(sub, sequence.back());

  SequentialHloOrdering ordering(module->schedule());
  EXPECT_TRUE(ordering.ExecutesBefore(add, negate));

  // Clear the schedule using the descheduling pass.
  HloDescheduler descheduler;
  EXPECT_TRUE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(bool descheduler_changed,
                          descheduler.Run(module.get()));
  EXPECT_TRUE(descheduler_changed);
  EXPECT_FALSE(module->has_schedule());
}

TEST_F(HloSchedulingTest, ListSchedulerHandlesAliasing) {
  const char* module_str = R"(
HloModule test_aliasing_module

ENTRY root {
  param = s32[1000] parameter(0)
  p0 = s32[1000] copy(param)
  p1 = s32[1000] copy(param)
  t = (s32[1000], s32[1000]) tuple(p0, p1)
  a = s32[1000] get-tuple-element(t), index=0
  b = s32[1000] get-tuple-element(t), index=1
  c = s32[1000] add(a, b)
  d = s32[1000] add(c, b)
  e = s32[1000] add(c, c)
  f = s32[1000] add(e, e)
  ROOT result = (s32[1000], s32[1000], s32[1000]) tuple(d, e, f)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  BufferValue::SizeFunction size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  int64_t peak_memory;
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), ListMemoryScheduler(&alias_info_, &size_fn),
                     /*execution_threads=*/{}, &peak_memory));
  TF_ASSERT_OK(module->set_schedule(schedule));
  // Verify that all instructions are in the sequence.
  const std::vector<HloInstruction*>& sequence =
      schedule.sequence(module->entry_computation()).instructions();
  EXPECT_EQ(module->entry_computation()->instruction_count(), sequence.size());

  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name;
  for (const HloInstruction* instruction : sequence) {
    instructions_by_name[instruction->name()] = instruction;
  }

  // The first instruction should be the parameter and the last the root.
  EXPECT_EQ(instructions_by_name.at("param"), sequence.front());
  EXPECT_EQ(instructions_by_name.at("result"), sequence.back());

  // Instructions "d" and "e" will both be schedulable at the same time, but
  // instruction "d" allows us to free the buffer of "p1", so the list scheduler
  // should prefer it.
  SequentialHloOrdering ordering(schedule);
  EXPECT_TRUE(ordering.ExecutesBefore(instructions_by_name.at("d"),
                                      instructions_by_name.at("e")));
  EXPECT_EQ(PeakMemoryUseOfEntryComputation(module.get(), &size_fn),
            peak_memory);
}

// Knobs of a generated tuple heavy while body. Every element the loop carries
// has its own size, so the priorities of its consumers differ.
struct TupleHeavyWhileBodySpec {
  std::string name;
  int tuple_width;
  // Results are tupled again and read back through get-tuple-element, so
  // consumers use a buffer through an alias of its defining instruction.
  bool read_through_aliases;
  // An add whose two operands hold the same buffer.
  bool overlapping_operands;
  // A multiply of a value with itself.
  bool duplicated_operands;
  // Elements that also combine with constants. Two instructions share one
  // constant: one on the main chain and an early userless one that reads the
  // element through a tuple, so its priority goes stale until the constant's
  // other use refreshes it. A second constant feeds a userless negate alone.
  int constants;
  // Slices without users, which compete with the root for the last slots.
  int dead_instructions;
  uint32_t seed;
};

void PrintTo(const TupleHeavyWhileBodySpec& spec, std::ostream* os) {
  *os << spec.name;
}

// Builds the HLO text of a while loop whose body follows the spec. The seed
// picks the element sizes, the pairs of elements that mix, the sizes of the
// dead slices and the elements the root passes through unchanged.
std::string TupleHeavyWhileBodyHlo(const TupleHeavyWhileBodySpec& spec) {
  CHECK_GE(spec.tuple_width, 2);
  CHECK_LE(spec.constants, spec.tuple_width);
  std::mt19937 rng(spec.seed);
  const int width = spec.tuple_width;
  std::vector<int> sizes(width);
  for (int i = 0; i < width; ++i) {
    sizes[i] = 8 * (i + 1);
    std::swap(sizes[i], sizes[rng() % (i + 1)]);
  }
  auto array = [&](int i) { return absl::StrFormat("f32[%d]", sizes[i]); };
  std::vector<std::string> element_shapes;
  for (int i = 0; i < width; ++i) {
    element_shapes.push_back(array(i));
  }
  const std::string carried_shape =
      absl::StrCat("(", absl::StrJoin(element_shapes, ", "), ", s32[])");

  std::string hlo = absl::StrFormat(
      "HloModule tuple_heavy_while_%s\n\nbody {\n  p = %s parameter(0)\n",
      spec.name, carried_shape);
  auto line = [&](absl::string_view name, absl::string_view shape,
                  absl::string_view op) {
    absl::StrAppend(&hlo, "  ", name, " = ", shape, " ", op, "\n");
  };
  for (int i = 0; i < width; ++i) {
    line(absl::StrCat("g", i), array(i),
         absl::StrFormat("get-tuple-element(p), index=%d", i));
  }
  line("counter", "s32[]",
       absl::StrFormat("get-tuple-element(p), index=%d", width));
  line("one", "s32[]", "constant(1)");
  line("next", "s32[]", "add(counter, one)");
  for (int i = 0; i < width; ++i) {
    line(absl::StrCat("a", i), array(i), absl::StrFormat("negate(g%d)", i));
  }
  std::vector<std::string> reads(width);
  for (int i = 0; i < width; ++i) {
    reads[i] = absl::StrCat("a", i);
  }
  if (spec.read_through_aliases) {
    std::vector<std::string> operands;
    for (int i = 0; i < width; ++i) {
      operands.push_back(reads[i]);
    }
    line("t", absl::StrCat("(", absl::StrJoin(element_shapes, ", "), ")"),
         absl::StrFormat("tuple(%s)", absl::StrJoin(operands, ", ")));
    for (int i = 0; i < width; ++i) {
      reads[i] = absl::StrCat("ta", i);
      line(reads[i], array(i),
           absl::StrFormat("get-tuple-element(t), index=%d", i));
    }
  }
  for (int i = 0; i < width; ++i) {
    line(absl::StrCat("b", i), array(i),
         spec.overlapping_operands
             ? absl::StrFormat("add(%s, a%d)", reads[i], i)
             : absl::StrFormat("exponential(%s)", reads[i]));
    line(absl::StrCat("c", i), array(i),
         spec.duplicated_operands ? absl::StrFormat("multiply(b%d, b%d)", i, i)
                                  : absl::StrFormat("negate(b%d)", i));
    if (i < spec.constants) {
      std::vector<int> literal(sizes[i]);
      absl::c_iota(literal, 0);
      line(absl::StrCat("k", i), array(i),
           absl::StrFormat("constant({%s})", absl::StrJoin(literal, ", ")));
      line(absl::StrCat("ts", i), absl::StrCat("(", array(i), ")"),
           absl::StrFormat("tuple(a%d)", i));
      line(absl::StrCat("gs", i), array(i),
           absl::StrFormat("get-tuple-element(ts%d), index=0", i));
      line(absl::StrCat("r", i), array(i),
           absl::StrFormat("subtract(gs%d, k%d)", i, i));
      line(absl::StrCat("kk", i), array(i),
           absl::StrFormat("constant({%s})", absl::StrJoin(literal, ", ")));
      line(absl::StrCat("kd", i), array(i), absl::StrFormat("negate(kk%d)", i));
      line(absl::StrCat("d", i), array(i),
           absl::StrFormat("add(c%d, k%d)", i, i));
    } else {
      line(absl::StrCat("d", i), array(i), absl::StrFormat("negate(c%d)", i));
    }
  }
  for (int i = 0; i < width; ++i) {
    const int partner = (i + 1 + rng() % (width - 1)) % width;
    line(absl::StrCat("cat", i),
         absl::StrFormat("f32[%d]", sizes[i] + sizes[partner]),
         absl::StrFormat("concatenate(d%d, d%d), dimensions={0}", i, partner));
    line(absl::StrCat("e", i), array(i),
         absl::StrFormat("slice(cat%d), slice={[0:%d]}", i, sizes[i]));
  }
  for (int k = 0; k < spec.dead_instructions; ++k) {
    const int i = k % width;
    const int size = 1 + rng() % (sizes[i] - 1);
    line(absl::StrCat("dead", k), absl::StrFormat("f32[%d]", size),
         absl::StrFormat("slice(e%d), slice={[0:%d]}", i, size));
  }
  std::vector<std::string> results;
  for (int i = 0; i < width; ++i) {
    results.push_back(rng() % 3 == 0 ? absl::StrCat("g", i)
                                     : absl::StrCat("e", i));
  }
  results.push_back("next");
  absl::StrAppend(&hlo, "  ROOT out = ", carried_shape, " tuple(",
                  absl::StrJoin(results, ", "), ")\n}\n\n");

  absl::StrAppend(&hlo, "cond {\n  cp = ", carried_shape, " parameter(0)\n");
  absl::StrAppendFormat(&hlo, "  ci = s32[] get-tuple-element(cp), index=%d\n",
                        width);
  absl::StrAppend(&hlo,
                  "  limit = s32[] constant(3)\n"
                  "  ROOT lt = pred[] compare(ci, limit), direction=LT\n}\n\n");

  absl::StrAppend(&hlo, "ENTRY main {\n");
  std::vector<std::string> parameters;
  for (int i = 0; i < width; ++i) {
    parameters.push_back(absl::StrCat("p", i));
    absl::StrAppendFormat(&hlo, "  p%d = %s parameter(%d)\n", i, array(i), i);
  }
  parameters.push_back("zero");
  absl::StrAppend(&hlo, "  zero = s32[] constant(0)\n  init = ", carried_shape,
                  " tuple(", absl::StrJoin(parameters, ", "), ")\n");
  absl::StrAppend(&hlo, "  ROOT loop = ", carried_shape,
                  " while(init), condition=cond, body=body\n}\n");
  return hlo;
}

// The list scheduling heuristic written the direct way, as the reference for
// ListMemoryScheduler: the unscheduled use count of every buffer lives in a
// map keyed by HloValue, and an entry's priority is recomputed by scanning the
// buffers it uses whenever the heuristic consults it.
HloInstructionSequence ReferenceListSchedule(
    HloComputation* computation, const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function) {
  const HloDataflowAnalysis& dataflow = alias_analysis.dataflow_analysis();
  auto ignored = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kParameter ||
           instruction.opcode() == HloOpcode::kConstant;
  };

  // An instruction uses every value in the flattened value sets of its
  // unique operands; the root's values have one more use, the live out.
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloValue*>>
      uses;
  absl::flat_hash_map<const HloValue*, int64_t> unscheduled_uses;
  absl::flat_hash_map<const HloInstruction*, int64_t> bytes_defined;
  for (HloInstruction* instruction : computation->instructions()) {
    absl::flat_hash_set<const HloValue*>& used = uses[instruction];
    for (const HloInstruction* operand : instruction->unique_operands()) {
      const HloValueSet value_set = dataflow.GetFlattenedValueSet(operand);
      used.insert(value_set.values().begin(), value_set.values().end());
    }
    for (const HloValue* value : used) {
      ++unscheduled_uses[value];
    }
    int64_t& defined = bytes_defined[instruction];
    if (!ignored(*instruction)) {
      dataflow.GetInstructionValueSet(instruction)
          .ForEachElement([&](const ShapeIndex& index, const HloValueSet&) {
            if (dataflow.ValueIsDefinedAt(instruction, index)) {
              defined +=
                  size_function(dataflow.GetValueDefinedAt(instruction, index));
            }
          });
    }
  }
  const HloValueSet live_out =
      dataflow.GetFlattenedValueSet(computation->root_instruction());
  for (const HloValue* value : live_out.values()) {
    ++unscheduled_uses[value];
  }

  using Priority = std::pair<int64_t, int64_t>;
  auto priority = [&](const HloInstruction* instruction) -> Priority {
    if (ShapeUtil::IsEffectiveScalar(instruction->shape())) {
      return {std::numeric_limits<int64_t>::max(),
              std::numeric_limits<int64_t>::max()};
    }
    int64_t freed = 0;
    if (instruction->opcode() == HloOpcode::kOutfeed &&
        !instruction->outfeed_config().empty()) {
      freed = INT_MAX;
    } else if (instruction->opcode() == HloOpcode::kInfeed &&
               !instruction->infeed_config().empty()) {
      freed = INT_MIN;
    } else {
      for (const HloValue* value : uses.at(instruction)) {
        if (!ignored(*value->instruction()) &&
            unscheduled_uses.at(value) == 1) {
          freed += size_function(*value);
        }
      }
      freed -= bytes_defined.at(instruction);
    }
    return {freed, instruction->user_count()};
  };

  absl::flat_hash_map<const HloInstruction*, int64_t> unscheduled_predecessors;
  for (HloInstruction* instruction : computation->instructions()) {
    for (HloInstruction* user : instruction->users()) {
      ++unscheduled_predecessors[user];
    }
    for (HloInstruction* successor : instruction->control_successors()) {
      ++unscheduled_predecessors[successor];
    }
  }

  // Ready instructions ordered by priority; among equal priorities the most
  // recently inserted one is scheduled first.
  std::multimap<Priority, HloInstruction*> ready;
  absl::flat_hash_map<const HloInstruction*,
                      std::multimap<Priority, HloInstruction*>::iterator>
      ready_entry;
  auto make_ready = [&](HloInstruction* instruction) {
    ready_entry[instruction] =
        ready.emplace(priority(instruction), instruction);
  };
  for (HloInstruction* instruction : computation->instructions()) {
    if (instruction->operands().empty() &&
        instruction->control_predecessors().empty()) {
      make_ready(instruction);
    }
  }

  HloInstructionSequence sequence;
  while (!ready.empty()) {
    auto best_it = std::prev(ready.end());
    HloInstruction* best = best_it->second;
    ready.erase(best_it);
    ready_entry.erase(best);
    sequence.push_back(best);

    bool refresh = false;
    for (const HloValue* value : uses.at(best)) {
      int64_t& count = unscheduled_uses.at(value);
      --count;
      if (count == 1) {
        refresh = true;
      }
    }
    auto release = [&](HloInstruction* instruction) {
      if (--unscheduled_predecessors.at(instruction) == 0) {
        make_ready(instruction);
      }
    };
    for (HloInstruction* user : best->users()) {
      release(user);
    }
    for (HloInstruction* successor : best->control_successors()) {
      release(successor);
    }
    if (!refresh) {
      continue;
    }
    // Only the ready users of the scheduled instruction's operands get a
    // fresh priority; a changed one moves behind the entries of equal
    // priority. Every other ready instruction keeps its stale priority.
    for (HloInstruction* operand : best->operands()) {
      for (HloInstruction* user : operand->users()) {
        auto it = ready_entry.find(user);
        if (it == ready_entry.end()) {
          continue;
        }
        const Priority fresh = priority(user);
        if (fresh == it->second->first) {
          continue;
        }
        auto stale = it->second;
        it->second = ready.emplace(fresh, user);
        ready.erase(stale);
      }
    }
  }
  return sequence;
}

std::string InstructionNames(const HloInstructionSequence& sequence) {
  std::vector<absl::string_view> names;
  for (const HloInstruction* instruction : sequence.instructions()) {
    names.push_back(instruction->name());
  }
  return absl::StrJoin(names, " ");
}

class ListSchedulerStressTest
    : public HloSchedulingTest,
      public ::testing::WithParamInterface<TupleHeavyWhileBodySpec> {};

// ListMemoryScheduler must produce the reference's sequence on every
// computation of a generated tuple heavy while loop: same buffer use counts
// at every step, same bytes freed, same refresh points, same tie breaking.
TEST_P(ListSchedulerStressTest, MatchesReference) {
  const std::string hlo = TupleHeavyWhileBodyHlo(GetParam());
  SCOPED_TRACE(hlo);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  BufferValue::SizeFunction size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(),
                     ListMemoryScheduler(&alias_info_, &size_fn)));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                       HloAliasAnalysis::Run(module.get(), &alias_info_));
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    EXPECT_EQ(InstructionNames(schedule.sequence(computation)),
              InstructionNames(
                  ReferenceListSchedule(computation, *alias_analysis, size_fn)))
        << "computation " << computation->name();
  }
}

INSTANTIATE_TEST_SUITE_P(
    TupleHeavyWhileBodies, ListSchedulerStressTest,
    ::testing::ValuesIn(std::vector<TupleHeavyWhileBodySpec>{
        {"plain_w2", 2, false, false, false, 0, 0, 1},
        {"aliases_w4", 4, true, false, false, 0, 0, 2},
        {"overlap_w4", 4, true, true, false, 0, 0, 3},
        {"duplicates_w4", 4, true, true, true, 0, 0, 4},
        {"constants_w6", 6, true, true, true, 3, 0, 5},
        {"dead_w6", 6, true, true, true, 3, 4, 6},
        {"all_w8_seed7", 8, true, true, true, 4, 5, 7},
        {"all_w8_seed8", 8, true, true, true, 8, 6, 8},
        {"all_w8_seed9", 8, true, true, true, 8, 6, 9},
        {"all_w8_seed10", 8, true, true, true, 5, 8, 10},
        {"all_w12_seed11", 12, true, true, true, 6, 8, 11},
        {"all_w12_seed12", 12, true, true, true, 12, 10, 12},
        {"all_w16_seed13", 16, true, true, true, 8, 12, 13},
    }),
    [](const ::testing::TestParamInfo<TupleHeavyWhileBodySpec>& info) {
      return info.param.name;
    });

TEST_F(HloSchedulingTest, DefaultSchedulerRunsThreeSchedulers) {
  const char* module_str = R"(
HloModule test_aliasing_module

ENTRY root {
  param = s32[1000] parameter(0)
  p0 = s32[1000] copy(param)
  p1 = s32[1000] copy(param)
  t = (s32[1000], s32[1000]) tuple(p0, p1)
  a = s32[1000] get-tuple-element(t), index=0
  b = s32[1000] get-tuple-element(t), index=1
  c = s32[1000] add(a, b)
  d = s32[1000] add(c, b)
  e = s32[1000] add(c, c)
  f = s32[1000] add(e, e)
  ROOT result = (s32[1000], s32[1000], s32[1000]) tuple(d, e, f)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));

  BufferValue::SizeFunction size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  module->metadata()->RecordPassStart();
  int64_t peak_memory;
  ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(),
                     DefaultMemoryScheduler(&alias_info_, &size_fn),
                     /*execution_threads=*/{}, &peak_memory));
  ASSERT_OK(module->set_schedule(schedule));
  MemorySchedulerMetrics metrics;
  auto metadata = module->metadata()->GetCurrentHloPassMetadata();
  ASSERT_OK(metadata.status());
  metadata.value()->custom_metadata().UnpackTo(&metrics);

  EXPECT_EQ(metrics.schedulers_size(), 3);

  auto list_metrics = metrics.schedulers(0);
  auto dfs_metrics = metrics.schedulers(1);
  auto post_order_metrics = metrics.schedulers(2);

  EXPECT_GT(list_metrics.peak_memory(), 0);
  EXPECT_GT(dfs_metrics.peak_memory(), 0);
  EXPECT_GT(post_order_metrics.peak_memory(), 0);

  EXPECT_TRUE(list_metrics.valid_schedule());
  EXPECT_TRUE(dfs_metrics.valid_schedule());
  EXPECT_TRUE(post_order_metrics.valid_schedule());

  EXPECT_GE(metrics.selected_scheduler_idx(), 0);
  EXPECT_LE(metrics.selected_scheduler_idx(), 2);
}

TEST_F(HloSchedulingTest, HostSendDoneSchedule) {
  const char* const module_str = R"(
HloModule module

ENTRY entry {
  %p = f32[1000, 1000] parameter(0)
  %token.0 = token[] after-all()
  %send = (f32[1000, 1000], token[]) send(%p, %token.0),
    channel_id=1, is_host_transfer=true
  %n1 = f32[1000, 1000] negate(%p)
  %n2 = f32[1000, 1000] negate(%n1)
  %n3 = f32[1000, 1000] negate(%n2)
  %send-done = token[] send-done(%send), channel_id=1, is_host_transfer=true
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), ListMemoryScheduler(&alias_info_, size_fn)));
  // Verify that all instructions are in the sequence.
  const std::vector<HloInstruction*>& sequence =
      schedule.sequence(module->entry_computation()).instructions();
  EXPECT_EQ(module->entry_computation()->instruction_count(), sequence.size());

  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name;
  for (const HloInstruction* instruction : sequence) {
    instructions_by_name[instruction->name()] = instruction;
  }

  EXPECT_LT(absl::c_find(sequence, instructions_by_name.at("send-done")),
            absl::c_find(sequence, instructions_by_name.at("n1")));
}

TEST_F(HloSchedulingTest, TuplesAreAccountedCorrectly) {
  auto builder = HloComputation::Builder(TestName());
  const Shape r1f32 = ShapeUtil::MakeShape(xla::F32, {6});

  // Wrap lit in abs because constants are considered free by
  // IgnoreInstruction, and it skews the accounting.
  auto lit = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 1, 1, 1, 1, 1})));
  auto abs_const = builder.AddInstruction(
      HloInstruction::CreateUnary(r1f32, HloOpcode::kAbs, lit));

  auto abs_abs1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r1f32, HloOpcode::kAbs, abs_const));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple(
      absl::Span<HloInstruction* const>({abs_abs1})));
  auto tuple_elm = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(r1f32, tuple, 0));

  auto abs_abs2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r1f32, HloOpcode::kAbs, abs_const));

  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kAdd,
                                                      tuple_elm, abs_abs2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(
          module.get(),
          ListMemoryScheduler(&alias_info_, [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
          })));

  // Verify that all instructions are in the sequence.
  EXPECT_EQ(module->entry_computation()->instruction_count(),
            schedule.sequence(module->entry_computation()).size());
  SequentialHloOrdering ordering(schedule);
  // tuple allocates the tuple buffer and doesn't free anything.
  // abs_abs2 uses the same buffer for input/output, so its bytes-freed is 0.
  // abs_abs2 should be scheduled before tuple by List.
  EXPECT_TRUE(ordering.ExecutesBefore(abs_abs2, tuple));
}

TEST_F(HloSchedulingTest, MultiOutputFusionAccountedCorrectly) {
  const Shape r1f32 = ShapeUtil::MakeShape(xla::F32, {5});
  HloComputation::Builder builder(TestName());

  auto c1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 1, 1, 1, 1})));
  auto c2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 2, 3, 4, 5})));
  auto c3 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({0, 2, 4, 6, 8})));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kAdd, c1, c2));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kMultiply, add, c3));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({add, mul}));

  auto tuple_elm = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(r1f32, tuple, 0));

  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r1f32, HloOpcode::kExp, c3));

  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kAdd, tuple_elm, exp));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());

  auto fusion = computation->CreateFusionInstruction(
      {tuple, mul, add}, HloInstruction::FusionKind::kLoop);

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(
          module.get(),
          ListMemoryScheduler(&alias_info_, [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), 2);
          })));

  // Verify that all instructions are in the sequence.
  EXPECT_EQ(module->entry_computation()->instruction_count(),
            schedule.sequence(module->entry_computation()).size());
  SequentialHloOrdering ordering(schedule);
  // fusion allocates memory for the tuple elements and doesn't free anything,
  // so it's more expensive than exp.
  EXPECT_TRUE(ordering.ExecutesBefore(exp, fusion));
}

TEST_F(HloSchedulingTest, TrivialScheduler) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  param.b = (s32[], s32[]) parameter(0)
  gte.0 = s32[] get-tuple-element(param.b), index=0
  gte.1 = s32[] get-tuple-element(param.b), index=1
  add = s32[] add(gte.0, gte.1)
  ROOT tuple = (s32[], s32[]) tuple(gte.0, add)
}

cond {
  param.c = (s32[], s32[]) parameter(0)
  ROOT constant = pred[] constant(true)
}

ENTRY main {
  init = (s32[], s32[]) parameter(0)
  ROOT while = (s32[], s32[]) while(init), condition=cond, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(module->has_schedule());
  TF_ASSERT_OK(HloTrivialScheduler().Run(module.get()).status());
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());

  // Verify that a clone of the module also has a schedule.
  std::unique_ptr<HloModule> clone = module->Clone();
  ASSERT_TRUE(clone->has_schedule());
  TF_ASSERT_OK(clone->schedule().Verify());
}

TEST_F(HloSchedulingTest, BFSScheduler) {
  // When scheduling for maximum concurrency, we expect HLO operations to be
  // processed in wave-fronts: (1) all bcasts (2) followed by additions (3)
  // followed by reduction (4) final result accumulation. This would allow
  // XLA executor to overlap the execution of independent additions and
  // reductions at the cost of extra memory to keep temporaries alive.
  const char* const hlo_string = R"(
    HloModule m

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,512,256] parameter(0)
      c0 = f32[] constant(0)

      c1 = f32[] constant(1)
      bcast1 = f32[1,2,1,512,256] broadcast(c1), dimensions={}
      add1 = f32[1,2,1,512,256] add(p0, bcast1)

      c2 = f32[] constant(2)
      bcast2 = f32[1,2,1,512,256] broadcast(c2), dimensions={}
      add2 = f32[1,2,1,512,256] add(p0, bcast2)

      c3 = f32[] constant(3)
      bcast3 = f32[1,2,1,512,256] broadcast(c3), dimensions={}
      add3 = f32[1,2,1,512,256] add(p0, bcast3)

      c4 = f32[] constant(4)
      bcast4 = f32[1,2,1,512,256] broadcast(c4), dimensions={}
      add4 = f32[1,2,1,512,256] add(p0, bcast4)

      c5 = f32[] constant(5)
      bcast5 = f32[1,2,1,512,256] broadcast(c5), dimensions={}
      add5 = f32[1,2,1,512,256] add(p0, bcast5)

      r1 = f32[1,2] reduce(add1, c0), dimensions={2,3,4}, to_apply=add
      r2 = f32[1,2] reduce(add2, c0), dimensions={2,3,4}, to_apply=add
      r3 = f32[1,2] reduce(add3, c0), dimensions={2,3,4}, to_apply=add
      r4 = f32[1,2] reduce(add4, c0), dimensions={2,3,4}, to_apply=add
      r5 = f32[1,2] reduce(add5, c0), dimensions={2,3,4}, to_apply=add

      out0 = f32[1,2] add(r1, r2)
      out1 = f32[1,2] add(r3, r4)
      out2 = f32[1,2] add(out0, out1)
      ROOT out3 = f32[1,2] add(out2, r5)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(),
                     BFScheduler(&alias_info_, [](const BufferValue& buffer) {
                       return ShapeUtil::ByteSizeOf(buffer.shape());
                     })));

  const std::vector<HloInstruction*>& sequence =
      schedule.sequence(module->entry_computation()).instructions();

  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name;
  for (const HloInstruction* instruction : sequence) {
    instructions_by_name[instruction->name()] = instruction;
  }

  auto index = [&](absl::string_view name) -> size_t {
    const HloInstruction* instruction = instructions_by_name.at(name);
    return std::distance(sequence.begin(), absl::c_find(sequence, instruction));
  };

  std::vector<size_t> indices = {
      index("bcast1"), index("bcast2"), index("bcast3"), index("bcast4"),
      index("bcast5"), index("add1"),   index("add2"),   index("add3"),
      index("add4"),   index("add5"),   index("r1"),     index("r2"),
      index("r3"),     index("r4"),     index("r5"),     index("out0"),
      index("out1"),   index("out2"),   index("out3")};

  EXPECT_TRUE(absl::c_is_sorted(indices));
}

}  // namespace
}  // namespace xla
