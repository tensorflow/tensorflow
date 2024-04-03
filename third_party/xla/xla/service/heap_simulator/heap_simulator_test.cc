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

#include "xla/service/heap_simulator/heap_simulator.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/hlo_ordering.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_value.h"
#include "xla/service/tuple_points_to_analysis.h"
#include "xla/status_macros.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class MinimumMemoryForSequenceTest : public HloTestBase {};

TEST_F(MinimumMemoryForSequenceTest, MultiComputation) {
  auto module = CreateNewVerifiedModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 0));
  HloInstruction* cond_data = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  // Free cond_param[] (16 bytes), Alloc PRED[] (1 byte)
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_data, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  // Entry params: 8 bytes (4 bytes per param), TOTAL=8
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param_iter"));
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_data"));
  // Tuple: 16 bytes (8 bytes per pointer), TOTAL=24
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({iter, data}));
  // While: 8 bytes (4 bytes per element), TOTAL=32
  // Both cond and body use a max of 24 bytes, TOTAL=56
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  auto size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_data, cond_lt});
  schedule.set_sequence(body_computation, {body_param});
  schedule.set_sequence(entry_computation, {iter, data, tuple, while_op});
  TF_ASSERT_OK(schedule.Verify());

  EXPECT_EQ(25,
            HeapSimulator::MinimumMemoryForModule(schedule, size_fn).value());
}

TEST_F(MinimumMemoryForSequenceTest, SubcomputationAccounting) {
  // HloModule SubcomputationAccounting

  // %WhileBody (body_param: f32[4]) -> f32[4] {
  //   %body_param = f32[4]{0} parameter(0)
  //   %constant.1 = f32[4]{0} constant({1, 1, 1, 1})
  //   ROOT %subtract = f32[4]{0} subtract(f32[4]{0} %body_param, f32[4]{0}
  //   %constant.1)
  // }

  // %WhileCond (cond_param: f32[4]) -> pred[] {
  //   %cond_param = f32[4]{0} parameter(0)
  //   %slice = f32[1]{0} slice(f32[4]{0} %cond_param), slice={[0:1]}
  //   %reshape = f32[] reshape(f32[1]{0} %slice)
  //   %constant = f32[] constant(0)
  //   ROOT %not-equal-to = pred[] compare(f32[] %reshape, f32[] %constant),
  //   direction=NE
  // }

  // ENTRY %SubcomputationAccounting () -> f32[2,4] {
  //   %constant.3 = f32[2,4]{1,0} constant(f32[2,4] { { 1, 2, 3, 4 }, { 1, 2,
  //   3, 4 } }) %transpose = f32[2,4]{1,0} transpose(f32[2,4]{1,0}
  //   %constant.3), dimensions={0,1} %constant.2 = f32[4]{0} constant({1, 1, 1,
  //   1}) %while = f32[4]{0} while(f32[4]{0} %constant.2),
  //   condition=%WhileCond, body=%WhileBody %broadcast = f32[2,4]{1,0}
  //   broadcast(f32[4]{0} %while), dimensions={1} ROOT %add = f32[2,4]{1,0}
  //   add(f32[2,4]{1,0} %transpose, f32[2,4]{1,0} %broadcast)
  // }

  auto module = CreateNewVerifiedModule();
  const Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  const Shape r1f32 = ShapeUtil::MakeShape(F32, {4});
  const Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 4});

  // reshape(slice(param)) != 0
  // Needs 5 bytes
  auto cond_builder = HloComputation::Builder("WhileCond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "cond_param"));
  HloInstruction* slice =
      cond_builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {1}), cond_param, {0}, {1}, {1}));
  HloInstruction* reshape =
      cond_builder.AddInstruction(HloInstruction::CreateReshape(r0f32, slice));
  HloInstruction* zero = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  HloInstruction* cond_comparison = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), reshape,
                                    zero, ComparisonDirection::kNe));
  auto cond_computation = module->AddEmbeddedComputation(cond_builder.Build());

  // param - 1
  // Needs 16 bytes
  auto body_builder = HloComputation::Builder("WhileBody");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "body_param"));
  HloInstruction* one_vector =
      body_builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<float>({1, 1, 1, 1})));
  HloInstruction* subtract =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          r1f32, HloOpcode::kSubtract, body_param, one_vector));
  auto body_computation = module->AddEmbeddedComputation(body_builder.Build());

  // transpose(matrix) + bcast(while)
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* while_init =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<float>({1, 1, 1, 1})));
  // Creates 16 bytes, ignoring subcomputations
  HloInstruction* while_loop =
      builder.AddInstruction(HloInstruction::CreateWhile(
          r1f32, cond_computation, body_computation, while_init));

  // Creates 32 bytes and frees 16
  HloInstruction* bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r2f32, while_loop, {1}));

  HloInstruction* matrix = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>(
          {{1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}})));
  // Creates 32 bytes
  HloInstruction* transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(r2f32, matrix, {0, 1}));

  // Creates 32 bytes and frees 64
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, transpose, bcast));

  auto entry_computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  std::vector<HloInstruction*> cond_vec = {cond_param, slice, reshape, zero,
                                           cond_comparison};
  std::vector<HloInstruction*> while_body_vec = {body_param, one_vector,
                                                 subtract};
  std::vector<HloInstruction*> entry_comp_vec = {while_init, while_loop, bcast,
                                                 matrix,     transpose,  add};
  schedule.set_sequence(cond_computation, cond_vec);
  schedule.set_sequence(body_computation, while_body_vec);
  schedule.set_sequence(entry_computation, entry_comp_vec);

  auto size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape());
  };
  absl::flat_hash_map<const HloComputation*, int64_t> memory_by_computation;
  memory_by_computation[cond_computation] = 5;
  memory_by_computation[body_computation] = 16;

  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module.get()).value();

  // HeapSimulator accounts for subcomputations. The output buffer is aliased,
  // so we don't double count.
  EXPECT_EQ(64, HeapSimulator::MinimumMemoryForComputation(
                    *entry_computation, schedule.sequence(entry_computation),
                    *alias_analysis, size_fn, &memory_by_computation)
                    .value());
}

const char kAlloc[] = "Alloc";
const char kFree[] = "Free";
const char kShare[] = "Share";
const char kFinish[] = "Finish";

// CallSequence records a sequence of Alloc/Free/Finish calls.
using CallSequence = std::vector<std::pair<std::string, const HloValue*>>;

// HeapCallRecorder is a dummy heap algorithm that simply records its calls.
class HeapCallRecorder : public HeapAlgorithm<HloValue> {
 public:
  explicit HeapCallRecorder(CallSequence* calls) : calls_(calls) {}
  ~HeapCallRecorder() override {}

  void Alloc(const HloValue* buffer, int64_t size) override {
    calls_->emplace_back(kAlloc, buffer);
    // Instead of assigning a real offset, we set the cardinality of the Alloc
    // call.  This isn't a valid assignment, but allows us to easily test for
    // buffer sharing.
    const int64_t offset = result_.chunk_map.size();
    result_.chunk_map.emplace(buffer, Chunk::FromOffsetSize(offset, size));
  }

  void ShareWith(const HloValue* buffer, const HloValue* shared,
                 int64_t size) override {
    calls_->emplace_back(kShare, buffer);
    // Instead of assigning a real offset, we set the cardinality of the Alloc
    // call.  This isn't a valid assignment, but allows us to easily test for
    // buffer sharing.
    const int64_t offset = result_.chunk_map[shared].offset;
    result_.chunk_map.emplace(buffer, Chunk::FromOffsetSize(offset, size));
  }
  void Free(const HloValue* buffer, int64_t size) override {
    calls_->emplace_back(kFree, buffer);
  }
  absl::StatusOr<Result> Finish() override {
    calls_->emplace_back(kFinish, nullptr);
    HeapSimulator::Result<HloValue> result;
    result.heap_size = result_.heap_size;
    result.heap_results.emplace_back(std::move(result_));
    return result;
  }

 private:
  CallSequence* calls_;
  HeapSimulator::HeapResult<HloValue> result_;
};

// HeapSimulatorTracker runs the heap simulator, recording the sequence of calls
// made to the underlying heap algorithm.  Tests compare the actual call
// sequence against an expected sequence.
class HeapSimulatorTracker {
 public:
  explicit HeapSimulatorTracker(
      std::unique_ptr<HloModule> module,
      const std::vector<HloInstruction*>& instruction_sequence,
      const std::vector<HloInstruction*>& must_alias_set = {},
      const HloDataflowAnalysis::CanShareBuffer& can_share_buffer = nullptr) {
    module_ = std::move(module);
    Init(instruction_sequence, can_share_buffer);
  }

  // Constructor for testing a single entry computation.
  explicit HeapSimulatorTracker(
      const std::string& name,
      std::unique_ptr<HloComputation> entry_computation,
      const std::vector<HloInstruction*>& instruction_sequence,
      const std::vector<HloInstruction*>& must_alias_set = {},
      const HloDataflowAnalysis::CanShareBuffer& can_share_buffer = nullptr) {
    HloModuleConfig config;
    module_ = std::make_unique<HloModule>(name, config);
    module_->AddEntryComputation(std::move(entry_computation));
    Init(instruction_sequence, can_share_buffer);
  }

  explicit HeapSimulatorTracker(const std::string& name) {
    HloModuleConfig config;
    module_ = std::make_unique<HloModule>(name, config);
  }

  // Similar to the single entry computation constructor above, but runs the
  // simulation over the entire module.
  void RunWholeModule(
      const std::vector<HloInstruction*>& full_module_sequence) {
    alias_analysis_ = HloAliasAnalysis::Run(module_.get()).value();

    // Construct the module sequence grouped by computation.
    HloSchedule schedule(module_.get());
    absl::flat_hash_map<const HloInstruction*, int> reverse_position;
    for (int i = 0; i < full_module_sequence.size(); ++i) {
      HloInstruction* instruction = full_module_sequence[i];
      schedule.GetOrCreateSequence(instruction->parent())
          .push_back(instruction);
      reverse_position[instruction] = full_module_sequence.size() - i;
    }

    // Hack the size_fn so that it returns a decreasing value as we step through
    // the sequence. This lets us ensure the Alloc calls are in the sequence
    // order. The Free calls are sorted by BufferValue.id, which is at least
    // deterministic.
    auto size_fn = [&reverse_position](const BufferValue& buffer) {
      return reverse_position[buffer.instruction()];
    };
    auto algorithm = std::make_unique<HeapCallRecorder>(&actual_calls_);
    result_ = HeapSimulator::Run(std::move(algorithm), *module_, schedule,
                                 *alias_analysis_, size_fn)
                  .value();
  }

  HloModule* module() { return module_.get(); }

  // Returns the buffer defined at the given instruction and index.
  const HloValue* BufferAt(const HloInstruction* instruction,
                           const ShapeIndex& index) const {
    return &alias_analysis_->dataflow_analysis().GetUniqueValueAt(instruction,
                                                                  index);
  }

  int64_t OffsetAt(const HloInstruction* instruction, const ShapeIndex& index) {
    const HloValue* buffer = BufferAt(instruction, index);
    CHECK_EQ(1, result_.heap_results.size());
    return result_.heap_results.at(0).chunk_map.at(buffer).offset;
  }

  // Ensures the expected sequence of Alloc/Free/Finish calls was performed.
  void ExpectCallSequence(const CallSequence& expected) const {
    auto to_string = [](const CallSequence& sequence) {
      std::string output;
      for (int64_t i = 0; i < sequence.size(); ++i) {
        auto pair = sequence.at(i);
        absl::StrAppendFormat(&output, "%d", i);
        absl::StrAppendFormat(&output, " :%s", pair.first);
        if (pair.second != nullptr) {
          absl::StrAppendFormat(&output, " - %s{%s}\n",
                                pair.second->instruction()->name(),
                                pair.second->index().ToString());
        }
      }
      return output;
    };
    EXPECT_EQ(expected, actual_calls_) << "Expected:\n"
                                       << to_string(expected) << " \nActual:\n"
                                       << to_string(actual_calls_) << "\n";
  }

  // Ensures the buffers defined by the respective (instruction,index) pairs are
  // shared, relying on the unique offsets assigned in
  // HeapCallRecorder::Alloc.
  void ExpectSharedBuffers(const HloInstruction* instruction_a,
                           const ShapeIndex& index_a,
                           const HloInstruction* instruction_b,
                           const ShapeIndex& index_b) {
    int64_t offset_a = OffsetAt(instruction_a, index_a);
    int64_t offset_b = OffsetAt(instruction_b, index_b);
    EXPECT_EQ(offset_a, offset_b);
  }

 private:
  void Init(const std::vector<HloInstruction*>& instruction_sequence,
            const HloDataflowAnalysis::CanShareBuffer& can_share_buffer) {
    // Since we're only tracking the sequence of Alloc/Free calls, the actual
    // size of the buffers doesn't matter, so we always return 0.  We rely on
    // the secondary sorting criteria of DecreasingSizeRunsHeap to sort calls
    // by buffer id, for determinism in the tests.
    auto zero_size = [](const BufferValue& buffer) { return 0; };
    auto algorithm = std::make_unique<HeapCallRecorder>(&actual_calls_);

    alias_analysis_ =
        HloAliasAnalysis::Run(module_.get(), can_share_buffer).value();

    HeapSimulator::Options options;

    result_ =
        HeapSimulator::Run(std::move(algorithm), *module_->entry_computation(),
                           HloInstructionSequence(instruction_sequence),
                           *alias_analysis_, zero_size, options)
            .value();
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  CallSequence actual_calls_;
  HeapSimulator::Result<HloValue> result_;
};

class HeapSimulatorTest : public HloTestBase {
 protected:
  HeapSimulatorTest() {}
  ~HeapSimulatorTest() override {}

  // Shapes for use in the examples.
  Shape f32scalar_ = ShapeUtil::MakeShape(xla::F32, {});
  Shape f32vec4_ = ShapeUtil::MakeShape(F32, {4});
};

TEST_F(HeapSimulatorTest, ScalarConstant) {
  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));

  // Constants aren't assigned.  See b/32248867
  HeapSimulatorTracker tracker(TestName(), builder.Build(), {const0});
  tracker.ExpectCallSequence({{kFinish, nullptr}});
}

TEST_F(HeapSimulatorTest, OneParam) {
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32scalar_, "param0"));

  // A single parameter which is also the output.
  HeapSimulatorTracker tracker(TestName(), builder.Build(), {param0});
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(param0, {})},
      {kFree, tracker.BufferAt(param0, {})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, Multiply) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32scalar_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));

  // We must keep all parameters and outputs.
  HeapSimulatorTracker tracker(TestName(), builder.Build(),
                               {paramA, paramX, mul});
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(paramX, {})},
      {kAlloc, tracker.BufferAt(mul, {})},
      // All params and outputs are freed at the end.
      {kFree, tracker.BufferAt(paramA, {})},
      {kFree, tracker.BufferAt(paramX, {})},
      {kFree, tracker.BufferAt(mul, {})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, MultiplyAdd) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32scalar_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec4_, "paramY"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd, mul, paramY));

  // The buffer for add is the output, and it's shared with the buffer for
  // mul.
  HeapSimulatorTracker tracker(TestName(), builder.Build(),
                               {paramA, paramX, mul, paramY, add});
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(paramX, {})},
      {kAlloc, tracker.BufferAt(paramY, {})},
      {kAlloc, tracker.BufferAt(mul, {})},
      {kFree, tracker.BufferAt(mul, {})},
      {kShare, tracker.BufferAt(add, {})},
      // All params and outputs are freed at the end.
      {kFree, tracker.BufferAt(paramA, {})},
      {kFree, tracker.BufferAt(paramX, {})},
      {kFree, tracker.BufferAt(paramY, {})},
      {kFree, tracker.BufferAt(add, {})},
      {kFinish, nullptr},
  });
  tracker.ExpectSharedBuffers(add, {}, mul, {});
}

TEST_F(HeapSimulatorTest, FusionOutputsOnlyShareOnce) {
  // Test that only one output of a fusion node will be shared with its operand.
  auto can_share_buffer =
      [](const HloInstruction* instr, const HloInstruction* operand,
         const ShapeIndex& user_index) -> std::optional<bool> {
    return instr->opcode() == HloOpcode::kFusion &&
           operand->shape().IsArray() &&
           ShapeUtil::Equal(operand->shape(),
                            ShapeUtil::GetSubshape(instr->shape(), user_index));
  };

  HloModuleConfig config;
  auto module = std::make_unique<HloModule>(TestName(), config);

  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec4_, "paramA"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec4_, HloOpcode::kNegate, paramA));

  // The fusion node has two outputs, both are eligible for being reused with
  // operand.
  auto fusion_builder = HloComputation::Builder("simple_two_way_forwarding");
  {
    auto param = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(0, f32vec4_, "x"));
    fusion_builder.AddInstruction(HloInstruction::CreateTuple({param, param}));
  }
  auto fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  auto fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      ShapeUtil::MakeTupleShape({f32vec4_, f32vec4_}),
      HloInstruction::FusionKind::kLoop, {negate}, fusion_computation));

  auto element0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32scalar_, fusion, 0));

  auto element1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32scalar_, fusion, 1));

  auto negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec4_, HloOpcode::kNegate, element0));
  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec4_, HloOpcode::kNegate, element1));

  builder.AddInstruction(HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd,
                                                      negate0, negate1));

  module->AddEntryComputation(builder.Build());
  HeapSimulatorTracker tracker(
      std::move(module),
      {paramA, negate, fusion, element0, element1, negate0, negate1}, {},
      can_share_buffer);
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(negate, {})},
      {kAlloc, tracker.BufferAt(fusion, {})},
      {kFree, tracker.BufferAt(negate, {})},
      {kShare, tracker.BufferAt(fusion, {0})},
      {kAlloc, tracker.BufferAt(fusion, {1})},
      {kFree, tracker.BufferAt(fusion, {})},
      {kAlloc, tracker.BufferAt(negate0, {})},
      {kFree, tracker.BufferAt(fusion, {0})},
      {kFree, tracker.BufferAt(negate0, {})},
      {kAlloc, tracker.BufferAt(negate1, {})},
      {kFree, tracker.BufferAt(fusion, {1})},
      {kFree, tracker.BufferAt(negate1, {})},
      {kFree, tracker.BufferAt(paramA, {})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, FusionOutputsOnlyShareOnceOutputShortLived) {
  // Test that only one output of a fusion node will be shared with its operand.
  // This variant of the test has a fusion node that dies immediately.
  auto can_share_buffer =
      [](const HloInstruction* instr, const HloInstruction* operand,
         const ShapeIndex& user_index) -> std::optional<bool> {
    if (instr->opcode() == HloOpcode::kFusion) {
      return true;
    }
    return false;
  };

  HloModuleConfig config;
  auto module = std::make_unique<HloModule>(TestName(), config);

  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec4_, "paramA"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec4_, HloOpcode::kNegate, paramA));

  // The fusion node has two outputs, both are eligible for being reused with
  // operand.
  auto fusion_builder = HloComputation::Builder("simple_two_way_forwarding");
  {
    auto param = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(0, f32vec4_, "x"));
    fusion_builder.AddInstruction(HloInstruction::CreateTuple({param, param}));
  }
  auto fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  auto fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      ShapeUtil::MakeTupleShape({f32vec4_, f32vec4_}),
      HloInstruction::FusionKind::kLoop, {negate}, fusion_computation));

  auto element1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32scalar_, fusion, 1));

  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec4_, HloOpcode::kNegate, element1));

  module->AddEntryComputation(builder.Build());
  HeapSimulatorTracker tracker(std::move(module),
                               {paramA, negate, fusion, element1, negate1}, {},
                               can_share_buffer);
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(negate, {})},
      {kFree, tracker.BufferAt(negate, {})},
      {kShare, tracker.BufferAt(fusion, {0})},
      {kAlloc, tracker.BufferAt(fusion, {})},
      {kAlloc, tracker.BufferAt(fusion, {1})},
      {kFree, tracker.BufferAt(fusion, {0})},
      {kFree, tracker.BufferAt(fusion, {})},
      {kAlloc, tracker.BufferAt(negate1, {})},
      {kFree, tracker.BufferAt(fusion, {1})},
      {kFree, tracker.BufferAt(paramA, {})},
      {kFree, tracker.BufferAt(negate1, {})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, BufferReusedOnce) {
  HeapSimulatorTracker tracker(TestName());
  auto builder = HloComputation::Builder(TestName());

  HloComputation::Builder fusion_builder("fusion");
  {
    HloComputation::Builder& builder = fusion_builder;
    auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, f32vec4_, "A"));
    auto exp = builder.AddInstruction(
        HloInstruction::CreateUnary(f32vec4_, HloOpcode::kExp, a_param));
    auto neg = builder.AddInstruction(
        HloInstruction::CreateUnary(f32vec4_, HloOpcode::kNegate, a_param));

    builder.AddInstruction(HloInstruction::CreateTuple({exp, neg}));
  }
  auto fusion_computation =
      tracker.module()->AddEmbeddedComputation(fusion_builder.Build());
  auto a_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec4_, "paramA"));
  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec4_, HloOpcode::kNegate, a_param));
  auto fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      ShapeUtil::MakeTupleShape({f32vec4_, f32vec4_}),
      HloInstruction::FusionKind::kLoop, {neg}, fusion_computation));
  tracker.module()->AddEntryComputation(builder.Build());

  tracker.RunWholeModule({a_param, neg, fusion});

  auto neg_buffer = tracker.OffsetAt(neg, {});
  int64_t output_buffer_0 = tracker.OffsetAt(fusion, {0});
  int64_t output_buffer_1 = tracker.OffsetAt(fusion, {1});
  // Only one buffer should be shared.
  EXPECT_TRUE((neg_buffer == output_buffer_0) ^
              (neg_buffer == output_buffer_1));
}

TEST_F(HeapSimulatorTest, MultiplyDot) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32scalar_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32scalar_, "paramY"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      f32vec4_, mul, paramY, dot_dnums, DefaultPrecisionConfig(2)));

  // The buffer for dot is the output, and it cannot be shared with the buffer
  // for mul, since dot isn't elementwise.
  HeapSimulatorTracker tracker(TestName(), builder.Build(),
                               {paramA, paramX, mul, paramY, dot});
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(paramX, {})},
      {kAlloc, tracker.BufferAt(paramY, {})},
      {kAlloc, tracker.BufferAt(mul, {})},
      {kAlloc, tracker.BufferAt(dot, {})},
      // All params and outputs are freed at the end.
      {kFree, tracker.BufferAt(mul, {})},
      {kFree, tracker.BufferAt(paramA, {})},
      {kFree, tracker.BufferAt(paramX, {})},
      {kFree, tracker.BufferAt(paramY, {})},
      {kFree, tracker.BufferAt(dot, {})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, MultiplyDotAdd) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32scalar_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32scalar_, "paramY"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      f32vec4_, mul, paramY, dot_dnums, DefaultPrecisionConfig(2)));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd, dot, paramA));

  // The buffer for add is the output, and it's shared with the buffer for
  // dot.
  HeapSimulatorTracker tracker(TestName(), builder.Build(),
                               {paramA, paramX, mul, paramY, dot, add});
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(paramX, {})},
      {kAlloc, tracker.BufferAt(paramY, {})},
      {kAlloc, tracker.BufferAt(mul, {})},
      {kAlloc, tracker.BufferAt(dot, {})},
      {kFree, tracker.BufferAt(mul, {})},
      {kFree, tracker.BufferAt(dot, {})},
      {kShare, tracker.BufferAt(add, {})},
      // All params and outputs are freed at the end.
      {kFree, tracker.BufferAt(paramA, {})},
      {kFree, tracker.BufferAt(paramX, {})},
      {kFree, tracker.BufferAt(paramY, {})},
      {kFree, tracker.BufferAt(add, {})},
      {kFinish, nullptr},
  });
  tracker.ExpectSharedBuffers(add, {}, dot, {});
}

TEST_F(HeapSimulatorTest, MultiplyDotDot) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32scalar_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32scalar_, "paramY"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot0 = builder.AddInstruction(HloInstruction::CreateDot(
      f32vec4_, mul, paramY, dot_dnums, DefaultPrecisionConfig(2)));
  auto dot1 = builder.AddInstruction(HloInstruction::CreateDot(
      f32vec4_, dot0, paramY, dot_dnums, DefaultPrecisionConfig(2)));

  // The buffer for dot1 is the output.  No buffers can be shared.  The buffer
  // for mul is freed before the end, since it's no longer used after dot0
  // finishes.
  HeapSimulatorTracker tracker(TestName(), builder.Build(),
                               {paramA, paramX, mul, paramY, dot0, dot1});
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(paramX, {})},
      {kAlloc, tracker.BufferAt(paramY, {})},
      {kAlloc, tracker.BufferAt(mul, {})},
      {kAlloc, tracker.BufferAt(dot0, {})},
      {kFree, tracker.BufferAt(mul, {})},  // mul no longer used
      {kAlloc, tracker.BufferAt(dot1, {})},
      {kFree, tracker.BufferAt(dot0, {})},
      // All params and outputs are freed at the end.
      {kFree, tracker.BufferAt(paramA, {})},
      {kFree, tracker.BufferAt(paramX, {})},
      {kFree, tracker.BufferAt(paramY, {})},
      {kFree, tracker.BufferAt(dot1, {})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, MultiplyDotDotTuple) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32scalar_, "paramA"));
  auto paramX = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec4_, "paramX"));
  auto paramY = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32scalar_, "paramY"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec4_, HloOpcode::kMultiply, paramA, paramX));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot0 = builder.AddInstruction(HloInstruction::CreateDot(
      f32vec4_, mul, paramY, dot_dnums, DefaultPrecisionConfig(2)));
  auto dot1 = builder.AddInstruction(HloInstruction::CreateDot(
      f32vec4_, dot0, paramY, dot_dnums, DefaultPrecisionConfig(2)));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({dot0, dot1}));

  // The buffers for dot0, dot1 and tuple are the output.  No buffers can be
  // shared.  The buffer for mul is freed before the end, since it's no longer
  // used after dot0 finishes.
  HeapSimulatorTracker tracker(
      TestName(), builder.Build(),
      {paramA, paramX, mul, paramY, dot0, dot1, tuple});
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(paramX, {})},
      {kAlloc, tracker.BufferAt(paramY, {})},
      {kAlloc, tracker.BufferAt(mul, {})},
      {kAlloc, tracker.BufferAt(dot0, {})},
      {kFree, tracker.BufferAt(mul, {})},  // mul no longer used
      {kAlloc, tracker.BufferAt(dot1, {})},
      {kAlloc, tracker.BufferAt(tuple, {})},
      // All params and outputs are freed at the end.
      {kFree, tracker.BufferAt(paramA, {})},
      {kFree, tracker.BufferAt(paramX, {})},
      {kFree, tracker.BufferAt(paramY, {})},
      {kFree, tracker.BufferAt(dot0, {})},
      {kFree, tracker.BufferAt(dot1, {})},
      {kFree, tracker.BufferAt(tuple, {})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, IndependentTupleElements) {
  auto builder = HloComputation::Builder(TestName());
  auto paramA = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32scalar_, "paramA"));
  auto paramB = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32scalar_, "paramB"));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32scalar_, HloOpcode::kMultiply, paramA, paramB));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      f32scalar_, HloOpcode::kAdd, paramA, paramB));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({mul, add}));
  auto element0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32scalar_, tuple, 0));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(f32vec4_, element0, {0}));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      f32scalar_, HloOpcode::kSubtract, paramA, paramB));
  auto element1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32scalar_, tuple, 1));
  auto output = builder.AddInstruction(
      HloInstruction::CreateTuple({broadcast, sub, element1}));

  HeapSimulatorTracker tracker(TestName(), builder.Build(),
                               {paramA, paramB, mul, add, tuple, element0,
                                broadcast, sub, element1, output});
  tracker.ExpectCallSequence({
      {kAlloc, tracker.BufferAt(paramA, {})},
      {kAlloc, tracker.BufferAt(paramB, {})},
      {kAlloc, tracker.BufferAt(mul, {})},
      {kAlloc, tracker.BufferAt(add, {})},
      {kAlloc, tracker.BufferAt(tuple, {})},
      {kAlloc, tracker.BufferAt(broadcast, {})},
      // The mul can be freed right after the broadcast happens, even though
      // The other GetTupleElement is still alive.
      {kFree, tracker.BufferAt(mul, {})},
      {kAlloc, tracker.BufferAt(sub, {})},
      // The temporary tuple is now dead.
      {kFree, tracker.BufferAt(tuple, {})},
      {kAlloc, tracker.BufferAt(output, {})},
      // All params and outputs are freed at the end.
      {kFree, tracker.BufferAt(paramA, {})},
      {kFree, tracker.BufferAt(paramB, {})},
      {kFree, tracker.BufferAt(add, {})},
      {kFree, tracker.BufferAt(broadcast, {})},
      {kFree, tracker.BufferAt(sub, {})},
      {kFree, tracker.BufferAt(output, {})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, WholeModule) {
  HeapSimulatorTracker tracker(TestName());

  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 0));
  HloInstruction* cond_data = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_data, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      tracker.module()->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloComputation* body_computation =
      tracker.module()->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, param));
  tracker.module()->AddEntryComputation(builder.Build());

  tracker.RunWholeModule(
      {param, while_op, body_param, cond_param, cond_iter, cond_data, cond_lt});
  tracker.ExpectCallSequence({
      // The entry computation param and while_op are allocated first.
      {kAlloc, tracker.BufferAt(param, {})},
      {kAlloc, tracker.BufferAt(param, {0})},
      {kAlloc, tracker.BufferAt(param, {1})},

      // Now the final cond less-than buffer is allocated.
      {kAlloc, tracker.BufferAt(cond_lt, {})},

      // The order of the remaining Free calls is based on the BufferValue.id,
      // which is deterministic, but not obvious.
      {kFree, tracker.BufferAt(cond_lt, {})},
      {kFree, tracker.BufferAt(param, {})},
      {kFree, tracker.BufferAt(param, {0})},
      {kFree, tracker.BufferAt(param, {1})},
      {kFinish, nullptr},
  });
}

TEST_F(HeapSimulatorTest, AsyncCallImplicitSharding) {
  std::string hlo_string = R"(
  HloModule module, is_scheduled=true

  called_computation {
    param0 = f32[4] parameter(0)
    constant = f32[1] constant(1)
    dynamic-update-slice = f32[4] dynamic-update-slice(param0, constant, constant)
    ROOT negate = f32[4] negate(dynamic-update-slice)
  }

  ENTRY entry {
    p0 = f32[8] parameter(0)
    call-start = ((f32[8]), f32[8], s32[]) call-start(p0), async_execution_thread="foo", to_apply=called_computation
    ROOT call-done = f32[8] call-done(call-start)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  auto size_fn = [](const BufferValue& buffer) -> int64_t {
    const Shape& shape = buffer.shape();
    if (!shape.IsArray()) {
      return 0;
    }
    return ShapeUtil::ByteSizeOf(shape);
  };
  auto algorithm = std::make_unique<GlobalDecreasingSizeBestFitHeap<HloValue>>(
      /*alignment=*/1);

  HeapSimulator::Result<HloValue> result =
      HeapSimulator::Run(std::move(algorithm), *module, module->schedule(),
                         *alias_analysis, size_fn)
          .value();
  for (const auto& [value, chunk] : result.heap_results[0].chunk_map) {
    if (value->instruction()->name() == "dynamic-update-slice") {
      EXPECT_EQ(chunk.size, 32);
    }
  }
}

// Base class for heap algorithm tests.
class HeapAlgorithmTestBase : public ::testing::Test {
 protected:
  HeapAlgorithmTestBase() : builder_("heap_simulator_test") {
    buffer_a_ = DummyBufferValue();
    buffer_b_ = DummyBufferValue();
    buffer_c_ = DummyBufferValue();
    buffer_d_ = DummyBufferValue();
    buffer_e_ = DummyBufferValue();
    buffer_f_ = DummyBufferValue();
    buffer_g_ = DummyBufferValue();
    buffer_h_ = DummyBufferValue();
    buffer_i_ = DummyBufferValue();
  }
  ~HeapAlgorithmTestBase() override {}

  const HloValue* buffer_a_;
  const HloValue* buffer_b_;
  const HloValue* buffer_c_;
  const HloValue* buffer_d_;
  const HloValue* buffer_e_;
  const HloValue* buffer_f_;
  const HloValue* buffer_g_;
  const HloValue* buffer_h_;
  const HloValue* buffer_i_;

 private:
  // Create a dummy HloValue to pass to the heap algorithm.
  const HloValue* DummyBufferValue() {
    const HloValue::Id id = buffers_.size();
    auto const0 = builder_.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    buffers_.emplace_back(std::make_unique<HloValue>(id, const0, ShapeIndex{}));
    return buffers_.back().get();
  }

  HloComputation::Builder builder_;
  std::vector<std::unique_ptr<HloValue>> buffers_;
};

class NoFragmentationStatsHeapTest : public HeapAlgorithmTestBase {};

TEST_F(NoFragmentationStatsHeapTest, Empty) {
  NoFragmentationStatsHeap<HloValue> heap;
  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> result,
                          heap.Finish());
  EXPECT_EQ(0, result.heap_size);
}

TEST_F(NoFragmentationStatsHeapTest, Simple) {
  NoFragmentationStatsHeap<HloValue> heap;
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 20);
  heap.Alloc(buffer_c_, 30);
  heap.Alloc(buffer_d_, 30);
  heap.Free(buffer_a_, 10);
  heap.Free(buffer_b_, 20);
  heap.Free(buffer_c_, 30);
  heap.Free(buffer_d_, 30);
  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> result,
                          heap.Finish());
  EXPECT_EQ(90, result.heap_size);
}

TEST_F(NoFragmentationStatsHeapTest, Mixed) {
  NoFragmentationStatsHeap<HloValue> heap;
  heap.Alloc(buffer_a_, 10);  // max: A

  heap.Alloc(buffer_b_, 20);  // max: A+B
  heap.Free(buffer_b_, 20);

  heap.Alloc(buffer_c_, 30);  // max: A+C
  heap.Free(buffer_c_, 30);

  heap.Alloc(buffer_d_, 5);  // max: A+C
  heap.Free(buffer_d_, 5);

  heap.Free(buffer_a_, 10);
  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> result,
                          heap.Finish());
  EXPECT_EQ(40, result.heap_size);
}

class GlobalDecreasingSizeBestFitHeapTest : public HeapAlgorithmTestBase {};

TEST_F(GlobalDecreasingSizeBestFitHeapTest, Empty) {
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/1);
  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> result,
                          heap.Finish());
  EXPECT_EQ(0, result.heap_size);
  EXPECT_EQ(1, result.heap_results.size());
  EXPECT_EQ(0, result.heap_results.at(0).chunk_map.size());
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, DecreasingSize) {
  // space
  //   ^
  //   |  +---a---+
  //   |      +-------+
  //   |      +---c---+
  //   |    +-------+
  //   |    |   b   |
  //   |    +-------+
  //   |         +-------+
  //   |         |       |
  //   |         |   d   |
  //   |         +-------+
  //   -----------------> time
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 30);
  heap.Alloc(buffer_c_, 20);
  heap.Alloc(buffer_d_, 40);
  heap.Free(buffer_a_, 10);
  heap.Free(buffer_b_, 30);
  heap.Free(buffer_c_, 20);
  heap.Free(buffer_d_, 40);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> results,
                          heap.Finish());
  EXPECT_EQ(1, results.heap_results.size());
  const HeapSimulator::HeapResult<HloValue>& result =
      results.heap_results.at(0);
  EXPECT_EQ(100, result.heap_size);
  EXPECT_EQ(10, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(30, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(20, result.chunk_map.at(buffer_c_).size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_d_).size);

  EXPECT_EQ(90, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(40, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(70, result.chunk_map.at(buffer_c_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_d_).offset);
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, DecreasingSizeWithAlignment) {
  // space
  //   ^
  //   |      +-------+
  //   |      +---b---+
  //   |            +-------+
  //   |            |       |
  //   |            |   d   |
  //   |  +---a---+ +-------+
  //   |
  //   |         +-------+
  //   |         |       |
  //   |         |   c   |
  //   |         |       |
  //   |         +-------+
  //   ---------------------> time
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/20);
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 20);
  heap.Alloc(buffer_c_, 50);
  heap.Free(buffer_a_, 10);
  heap.Alloc(buffer_d_, 40);
  heap.Free(buffer_b_, 20);
  heap.Free(buffer_c_, 50);
  heap.Free(buffer_d_, 40);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> results,
                          heap.Finish());
  EXPECT_EQ(1, results.heap_results.size());
  const HeapSimulator::HeapResult<HloValue>& result =
      results.heap_results.at(0);
  EXPECT_EQ(120, result.heap_size);
  EXPECT_EQ(10, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(20, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(50, result.chunk_map.at(buffer_c_).size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_d_).size);

  EXPECT_EQ(60, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(100, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_c_).offset);
  EXPECT_EQ(60, result.chunk_map.at(buffer_d_).offset);
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, BestFit) {
  // space
  //   ^
  //   |    +-------+
  //   |    +---b---+
  //   |         +-------+
  //   |         |   d   |
  //   | +--a--+ +-------+
  //   |      +-------+
  //   |      |       |
  //   |      |   c   |
  //   |      +-------+
  //   |           +-------+
  //   |           |       |
  //   |           |   e   |
  //   |           |       |
  //   |           +-------+
  //   ---------------------> time
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 20);
  heap.Alloc(buffer_c_, 40);
  heap.Free(buffer_a_, 10);
  heap.Alloc(buffer_d_, 30);
  heap.Alloc(buffer_e_, 50);
  heap.Free(buffer_b_, 20);
  heap.Free(buffer_c_, 40);
  heap.Free(buffer_d_, 30);
  heap.Free(buffer_e_, 50);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> results,
                          heap.Finish());
  EXPECT_EQ(1, results.heap_results.size());
  const HeapSimulator::HeapResult<HloValue>& result =
      results.heap_results.at(0);
  EXPECT_EQ(140, result.heap_size);
  EXPECT_EQ(10, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(20, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_c_).size);
  EXPECT_EQ(30, result.chunk_map.at(buffer_d_).size);
  EXPECT_EQ(50, result.chunk_map.at(buffer_e_).size);

  EXPECT_EQ(90, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(120, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(50, result.chunk_map.at(buffer_c_).offset);
  EXPECT_EQ(90, result.chunk_map.at(buffer_d_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_e_).offset);
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, Colocated) {
  // space      colocate
  //   ^   +--------------+
  //   |   v              v
  //   |+------+      +-------+
  //   ||      |      |       |
  //   ||      |+----+|       |
  //   |+--a---++-b--++---c---+
  //   ---------------------> time
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 40);
  heap.Free(buffer_a_, 40);
  heap.Alloc(buffer_b_, 20);
  heap.Free(buffer_b_, 20);
  heap.ShareWith(buffer_c_, buffer_a_, 40);
  heap.Free(buffer_c_, 40);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> results,
                          heap.Finish());
  EXPECT_EQ(1, results.heap_results.size());
  const HeapSimulator::HeapResult<HloValue>& result =
      results.heap_results.at(0);
  EXPECT_EQ(40, result.heap_size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(20, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_c_).size);

  EXPECT_EQ(0, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_c_).offset);
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, ColocatedII) {
  // space
  //   ^       +---------------+
  //   |       +-------b-------+
  //   |+------+      +-------+
  //   ||      |      |       |
  //   ||      |      |       | <--- colocate with a
  //   |+--a---+      +---c---+
  //   ---------------------> time
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 40);
  heap.Free(buffer_a_, 40);
  heap.Alloc(buffer_b_, 20);

  heap.ShareWith(buffer_c_, buffer_a_, 40);
  heap.Free(buffer_c_, 40);
  heap.Free(buffer_b_, 20);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> results,
                          heap.Finish());
  EXPECT_EQ(1, results.heap_results.size());
  const HeapSimulator::HeapResult<HloValue>& result =
      results.heap_results.at(0);
  EXPECT_EQ(60, result.heap_size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(20, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_c_).size);

  EXPECT_EQ(0, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(40, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_c_).offset);
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, ColocatedIII) {
  // space
  //   ^+------+      +-------+
  //   ||      |      |       | <--- colocate with a
  //   |+--a---+      +---c---+
  //   |       +---------------+
  //   |       |               |
  //   |       |               |
  //   |       +-------b-------+
  //   ---------------------> time
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 10);
  heap.Free(buffer_a_, 10);
  heap.Alloc(buffer_b_, 30);

  heap.ShareWith(buffer_c_, buffer_a_, 10);
  heap.Free(buffer_c_, 10);
  heap.Free(buffer_b_, 30);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> results,
                          heap.Finish());
  EXPECT_EQ(1, results.heap_results.size());
  const HeapSimulator::HeapResult<HloValue>& result =
      results.heap_results.at(0);
  EXPECT_EQ(40, result.heap_size);
  EXPECT_EQ(10, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(30, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(10, result.chunk_map.at(buffer_c_).size);

  EXPECT_EQ(30, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(30, result.chunk_map.at(buffer_c_).offset);
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, ColocatedDifferentSize1) {
  // space
  //   ^
  //   |         +---------------+
  //   |+------+ +-------b-------+
  //   ||      |      +-------+
  //   ||      |      |       | <--- colocate with a
  //   |+--a---+      +---c---+
  //   ---------------------> time
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 40);
  heap.Free(buffer_a_, 40);
  heap.Alloc(buffer_b_, 20);

  heap.ShareWith(buffer_c_, buffer_a_, 30);
  heap.Free(buffer_c_, 30);
  heap.Free(buffer_b_, 20);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> results,
                          heap.Finish());
  EXPECT_EQ(1, results.heap_results.size());
  const HeapSimulator::HeapResult<HloValue>& result =
      results.heap_results.at(0);
  EXPECT_EQ(50, result.heap_size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(20, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(30, result.chunk_map.at(buffer_c_).size);

  EXPECT_EQ(0, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(30, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_c_).offset);
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, ColocatedDifferentSize2) {
  // space
  //   ^         +-------------+
  //   |         +-----b-------+
  //   |              +-------+
  //   |+------+      |       |
  //   ||      |      |       |
  //   ||      |      |       | <--- colocate with a
  //   |+--a---+      +---c---+
  //   ---------------------> time
  GlobalDecreasingSizeBestFitHeap<HloValue> heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 40);
  heap.Free(buffer_a_, 40);
  heap.Alloc(buffer_b_, 20);

  heap.ShareWith(buffer_c_, buffer_a_, 50);
  heap.Free(buffer_c_, 50);
  heap.Free(buffer_b_, 20);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> results,
                          heap.Finish());
  EXPECT_EQ(1, results.heap_results.size());
  const HeapSimulator::HeapResult<HloValue>& result =
      results.heap_results.at(0);
  EXPECT_EQ(70, result.heap_size);
  EXPECT_EQ(40, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(20, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(50, result.chunk_map.at(buffer_c_).size);

  EXPECT_EQ(0, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(50, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_c_).offset);
}

class FindGlobalDecreasingSizeBestFitTest : public HeapAlgorithmTestBase {
 protected:
  class InheritedGlobalDecreasingSizeBestFitHeap
      : public GlobalDecreasingSizeBestFitHeap<HloValue> {
   public:
    InheritedGlobalDecreasingSizeBestFitHeap()
        : GlobalDecreasingSizeBestFitHeap(/*alignment=*/1) {}

    // Makes a BufferInterval from the input specifications, find a chunk
    // candidate for it, with the preferred_offset if > -1, and commit that
    // chunk. Returns the offset and the new heap size.
    std::pair<int64_t, int64_t> MakeFindAndCommit(
        const HloValue* buffer, int64_t size, int64_t start, int64_t end,
        int64_t preferred_offset = -1) {
      // Make the BufferInterval.
      MakeBufferInterval(buffer, size, start, end);
      BufferInterval* buffer_interval = &GetBufferInterval(buffer);

      // Find a chunk candidate.
      Chunk chunk_candidate =
          FindChunkCandidate(*buffer_interval, preferred_offset);
      EXPECT_EQ(chunk_candidate.size, size);
      std::pair<int64_t, int64_t> result = std::make_pair(
          chunk_candidate.offset, result_.UpdatedHeapSize(chunk_candidate));

      // Commit the chunk.
      CommitChunk(*buffer_interval, chunk_candidate);

      return result;
    }

    // Creates a BufferInterval from the inputs and adds it to
    // buffer_intervals_.
    void MakeBufferInterval(const HloValue* buffer, int64_t size, int64_t start,
                            int64_t end) {
      BufferInterval* buffer_interval = &buffer_intervals_[buffer];
      buffer_interval->buffer = buffer;
      buffer_interval->size = size;
      buffer_interval->start = start;
      buffer_interval->end = end;
    }

    // Adds a colocation to buffer_intervals_[buffer] for colocation.
    void AddColocationToBuffer(const HloValue* buffer,
                               const HloValue* colocation) {
      CHECK(buffer_intervals_.contains(buffer));
      buffer_intervals_[buffer].colocations.push_back(colocation);
    }

    // Returns buffer_intervals_[buffer]. The returned reference is invalidated
    // if any elements are added or removed from buffer_intervals_, e.g., if
    // MakeBufferInterval() is called.
    BufferInterval& GetBufferInterval(const HloValue* buffer) {
      CHECK(buffer_intervals_.contains(buffer));
      return buffer_intervals_[buffer];
    }

    // Expose protected function.
    std::vector<Chunk> FindChunkCandidates(
        const SlicedBufferInterval& sliced_buffer_interval,
        int64_t preferred_offset = -1) const {
      return GlobalDecreasingSizeBestFitHeap<HloValue>::FindChunkCandidates(
          sliced_buffer_interval, preferred_offset);
    }

    // Expose protected function.
    void CommitChunk(const BufferInterval& buffer_interval, Chunk chunk) {
      GlobalDecreasingSizeBestFitHeap<HloValue>::CommitChunk(buffer_interval,
                                                             chunk);
    }

    // Typically, only one chunk is allowed to be assigned to each buffer in
    // HeapSimulator. That limitation is insufficient for slices. However, MSA
    // is the only code that generates slices, and it gets around that
    // limitation by making this method a no-op. For testing, we'll allow
    // multiple chunks to be assigned to a buffer (as would be allowed in MSA).
    void AddToChunkMap(const HloValue* buffer, Chunk chunk) override {
      committed_[buffer].push_back(chunk);
    }

    const absl::flat_hash_map<const HloValue*, std::vector<Chunk>>& committed()
        const {
      return committed_;
    }

    int64_t heap_size() const { return result_.heap_size; }

   private:
    absl::flat_hash_map<const HloValue*, std::vector<Chunk>> committed_;
  };

  using BufferInterval =
      InheritedGlobalDecreasingSizeBestFitHeap::BufferInterval;
  using SlicedBufferInterval =
      InheritedGlobalDecreasingSizeBestFitHeap::SlicedBufferInterval;
  using Chunk = InheritedGlobalDecreasingSizeBestFitHeap::Chunk;

  InheritedGlobalDecreasingSizeBestFitHeap heap_;
};

TEST_F(FindGlobalDecreasingSizeBestFitTest, ChunkCandidate) {
  // space
  //   ^
  // 35|
  //   |            +-----------+
  //   |            |           |
  // 30|            |           |
  //   |            |  po: 15   |
  //   |            |           |
  // 25|            +-----g-----+
  //   |         +-----+
  //   |         |po:20|
  // 20|         +--f--+
  //   |                                +-----+
  //   |                                |     |
  // 15|                                |     |
  //   |      +-----------------+       |po:10|
  //   |      |                 |       |     |
  // 10|      +-------c---------+       +--e--+
  //   |         +-----+  +-----------+
  //   |         |     |  |   po: 5   |
  //  5|         |     |  +-----a-----+
  //   |+-----+  |     |
  //   ||po:10|  |     |
  //  0|+--d--+  +--b--+
  //   -----------------------------------------> time
  //    0  1  2  3  4  5  6  7  8  9 10 11 12 13
  using pair = std::pair<int64_t, int64_t>;

  // offset: 5, size: 5, start: 6, end: 10
  // Preferred offset 5 is returned.
  EXPECT_EQ(pair(5, 10), heap_.MakeFindAndCommit(buffer_a_, 5, 6, 10, 5));

  // offset: 0, size: 10, start: 3, end: 5
  EXPECT_EQ(pair(0, 10), heap_.MakeFindAndCommit(buffer_b_, 10, 3, 5));

  // offset: 10, size: 5, start: 2, end: 8
  EXPECT_EQ(pair(10, 15), heap_.MakeFindAndCommit(buffer_c_, 5, 2, 8));

  // offset: 0, size: 5, start: 0, end: 2
  // Preferred offset 10 could not be given because it is occupied.
  EXPECT_EQ(pair(0, 15), heap_.MakeFindAndCommit(buffer_d_, 5, 0, 2, 10));

  // offset: 10, size: 10, start: 11, end: 13
  // Preferred offset 10 is returned.
  EXPECT_EQ(pair(10, 20), heap_.MakeFindAndCommit(buffer_e_, 10, 11, 13, 10));

  // offset: 20, size: 5, start: 3, end: 5
  // Preferred offset 20 is returned.
  EXPECT_EQ(pair(20, 25), heap_.MakeFindAndCommit(buffer_f_, 5, 3, 5, 20));

  // offset: 25, size: 10, start: 4, end: 8
  // Preferred offset 15 could not be given because it is occupied.
  EXPECT_EQ(pair(25, 35), heap_.MakeFindAndCommit(buffer_g_, 10, 4, 8, 15));
}

TEST_F(FindGlobalDecreasingSizeBestFitTest, FindChunkCandidates) {
  //  space
  //    ^
  //    |
  // 30 -                             +----+
  //    |                             |    |
  //    -         +---------+    +---+|  E |
  //    |         |         |    |   ||    |
  // 20 -         |         |    | F |+----+
  //    |         |    C    |    |   ||    |
  //    -         |         |    +---++    |
  //    |         |         |    |     B   |
  // 10 -    +----+----+----+    +---------+
  //    |    |         |
  //    -    |    A    |         +---------+
  //    |    |         |         |    D    |
  //    +----|----|----|----|----|----|----|----> time
  //              10        20        30

  // Place and commit A.
  {  // Force sliced buffers to go out of scope before they are invalidated by
     // calls to MakeBufferInterval.
    heap_.MakeBufferInterval(buffer_a_, 10, 5, 15);
    auto sliced_buffer_a = SlicedBufferInterval::CreateMutableInterval(
        heap_.GetBufferInterval(buffer_a_));
    auto chunks = heap_.FindChunkCandidates(sliced_buffer_a);
    EXPECT_THAT(chunks, ::testing::ElementsAre(Chunk::FromOffsetSize(0, 10)));
    heap_.CommitChunk(sliced_buffer_a.full_buffer_interval(),
                      Chunk::FromOffsetSize(0, 10));
    EXPECT_THAT(
        heap_.committed(),
        ::testing::UnorderedElementsAre(::testing::Pair(
            buffer_a_, ::testing::ElementsAre(Chunk::FromOffsetSize(0, 10)))));
    EXPECT_EQ(heap_.heap_size(), 10);
  }

  // Colocate B and C.
  {  // Force sliced buffers to go out of scope before they are invalidated by
     // calls to MakeBufferInterval.
    heap_.MakeBufferInterval(buffer_b_, 10, 25, 35);
    heap_.MakeBufferInterval(buffer_c_, 15, 10, 20);
    // Note, HeapSimulator uses GetTransitiveColocations(), so we can colocate
    // b with c, without doing the reverse.
    heap_.AddColocationToBuffer(buffer_b_, buffer_c_);
    auto sliced_buffer_b = SlicedBufferInterval::CreateMutableInterval(
        heap_.GetBufferInterval(buffer_b_));
    auto sliced_buffer_c = SlicedBufferInterval::CreateMutableInterval(
        heap_.GetBufferInterval(buffer_c_));

    // // Slice B.
    sliced_buffer_b.Slice({5, 5});
    sliced_buffer_b.UpdateInclusiveSliceStartTimes({25, 30});

    // Place and commit B (and C transitively via colocation). B should be
    // placed at an offset that accommodates C; however, it should not have the
    // size of C.
    auto chunks = heap_.FindChunkCandidates(sliced_buffer_b);
    EXPECT_THAT(chunks, ::testing::ElementsAre(Chunk::FromOffsetSize(10, 5),
                                               Chunk::FromOffsetSize(15, 5)));
    // In today's code, MSA would massage the SlicedBufferInterval and returned
    // chunks before calling CommitChunks. We hard-code simulations of those
    // changes here.
    //
    // We turn:
    //      +----+          +----+
    //      |    |          |    |
    // +----+----+  => +----+    |
    // |         |     |    |    |
    // +---------+     +----+----+
    heap_.CommitChunk(BufferInterval{buffer_b_, 5, 25, 30, /*colocations=*/{},
                                     /*need_allocation=*/true},
                      Chunk::FromOffsetSize(10, 5));
    heap_.CommitChunk(
        BufferInterval{buffer_b_, 10, 30, 35, /*colocations=*/{buffer_c_},
                       /*need_allocation=*/true},
        Chunk::FromOffsetSize(10, 10));
    EXPECT_THAT(
        heap_.committed(),
        ::testing::UnorderedElementsAre(
            ::testing::Pair(buffer_a_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(0, 10))),
            ::testing::Pair(buffer_b_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(10, 5),
                                           Chunk::FromOffsetSize(10, 10))),
            ::testing::Pair(buffer_c_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(10, 15)))));
    EXPECT_EQ(heap_.heap_size(), 25);
  }

  // Place and commit D.
  {  // Force sliced buffers to go out of scope before they are invalidated by
     // calls to MakeBufferInterval.
    heap_.MakeBufferInterval(buffer_d_, 5, 25, 35);
    auto sliced_buffer_d = SlicedBufferInterval::CreateMutableInterval(
        heap_.GetBufferInterval(buffer_d_));
    auto chunks = heap_.FindChunkCandidates(sliced_buffer_d);
    EXPECT_THAT(chunks, ::testing::ElementsAre(Chunk::FromOffsetSize(0, 5)));
    heap_.CommitChunk(sliced_buffer_d.full_buffer_interval(),
                      Chunk::FromOffsetSize(0, 5));
    EXPECT_THAT(
        heap_.committed(),
        ::testing::UnorderedElementsAre(
            ::testing::Pair(buffer_a_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(0, 10))),
            ::testing::Pair(buffer_b_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(10, 5),
                                           Chunk::FromOffsetSize(10, 10))),
            ::testing::Pair(buffer_c_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(10, 15))),
            ::testing::Pair(buffer_d_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(0, 5)))));
    EXPECT_EQ(heap_.heap_size(), 25);
  }

  // Place and commit E. It should fit just on top of B.
  {  // Force sliced buffers to go out of scope before they are invalidated by
     // calls to MakeBufferInterval.
    heap_.MakeBufferInterval(buffer_e_, 10, 30, 35);
    auto sliced_buffer_e = SlicedBufferInterval::CreateMutableInterval(
        heap_.GetBufferInterval(buffer_e_));
    auto chunks = heap_.FindChunkCandidates(sliced_buffer_e);
    EXPECT_THAT(chunks, ::testing::ElementsAre(Chunk::FromOffsetSize(20, 10)));
    heap_.CommitChunk(sliced_buffer_e.full_buffer_interval(),
                      Chunk::FromOffsetSize(20, 10));
    EXPECT_THAT(
        heap_.committed(),
        ::testing::UnorderedElementsAre(
            ::testing::Pair(buffer_a_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(0, 10))),
            ::testing::Pair(buffer_b_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(10, 5),
                                           Chunk::FromOffsetSize(10, 10))),
            ::testing::Pair(buffer_c_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(10, 15))),
            ::testing::Pair(
                buffer_d_, ::testing::ElementsAre(Chunk::FromOffsetSize(0, 5))),
            ::testing::Pair(buffer_e_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(20, 10)))));
    EXPECT_EQ(heap_.heap_size(), 30);
  }

  // Place and commit F. It should fit on top of B's first slice.
  {  // Force sliced buffers to go out of scope before they are invalidated by
     // calls to MakeBufferInterval.
    heap_.MakeBufferInterval(buffer_f_, 10, 25, 29);
    auto sliced_buffer_f = SlicedBufferInterval::CreateMutableInterval(
        heap_.GetBufferInterval(buffer_f_));
    auto chunks = heap_.FindChunkCandidates(sliced_buffer_f);
    EXPECT_THAT(chunks, ::testing::ElementsAre(Chunk::FromOffsetSize(15, 10)));
    heap_.CommitChunk(sliced_buffer_f.full_buffer_interval(),
                      Chunk::FromOffsetSize(15, 10));
    EXPECT_THAT(
        heap_.committed(),
        ::testing::UnorderedElementsAre(
            ::testing::Pair(buffer_a_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(0, 10))),
            ::testing::Pair(buffer_b_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(10, 5),
                                           Chunk::FromOffsetSize(10, 10))),
            ::testing::Pair(buffer_c_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(10, 15))),
            ::testing::Pair(
                buffer_d_, ::testing::ElementsAre(Chunk::FromOffsetSize(0, 5))),
            ::testing::Pair(buffer_e_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(20, 10))),
            ::testing::Pair(buffer_f_, ::testing::ElementsAre(
                                           Chunk::FromOffsetSize(15, 10)))));
    EXPECT_EQ(heap_.heap_size(), 30);
  }
}

class ConstrainedGlobalDecreasingSizeBestFitHeapTest
    : public HeapAlgorithmTestBase {};

TEST_F(ConstrainedGlobalDecreasingSizeBestFitHeapTest, DecreasingSize) {
  // space
  //   ^
  //   |      +-------+
  //   |      +---c---+
  //   |    +-------+
  //   |    |   b   |
  //   |    +-------+
  //   | ................ // split into two allocations.
  //   |  +---a---+
  //   |         +-------+
  //   |         |       |
  //   |         |   d   |
  //   |         +-------+
  //   -----------------> time
  ConstrainedGlobalDecreasingSizeBestFitHeap heap(/*size_limit_per_heap=*/50,
                                                  /*alignment=*/1);
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 30);
  heap.Alloc(buffer_c_, 20);
  heap.Alloc(buffer_d_, 40);
  heap.Free(buffer_a_, 10);
  heap.Free(buffer_b_, 30);
  heap.Free(buffer_c_, 20);
  heap.Free(buffer_d_, 40);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> result,
                          heap.Finish());
  EXPECT_EQ(100, result.heap_size);
  EXPECT_EQ(2, result.heap_results.size());

  EXPECT_TRUE(result.heap_results[0].chunk_map.contains(buffer_a_));
  EXPECT_TRUE(result.heap_results[0].chunk_map.contains(buffer_d_));
  EXPECT_EQ(10, result.heap_results[0].chunk_map.at(buffer_a_).size);
  EXPECT_EQ(40, result.heap_results[0].chunk_map.at(buffer_d_).size);
  EXPECT_EQ(40, result.heap_results[0].chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(0, result.heap_results[0].chunk_map.at(buffer_d_).offset);
}

TEST_F(ConstrainedGlobalDecreasingSizeBestFitHeapTest,
       DecreasingSizeWithAlignment) {
  // space
  //   ^
  //   |      +-------+
  //   |      +---b---+
  //   |            +-------+
  //   |            |       |
  //   |            |   d   |
  //   |            +-------+
  //   | ...................
  //   |  +---a---+
  //   |
  //   |         +-------+
  //   |         |       |
  //   |         |   c   |
  //   |         |       |
  //   |         +-------+
  //   ---------------------> time
  ConstrainedGlobalDecreasingSizeBestFitHeap heap(/*size_limit_per_heap=*/70,
                                                  /*alignment=*/20);
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 20);
  heap.Alloc(buffer_c_, 50);
  heap.Free(buffer_a_, 10);
  heap.Alloc(buffer_d_, 40);
  heap.Free(buffer_b_, 20);
  heap.Free(buffer_c_, 50);
  heap.Free(buffer_d_, 40);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> result,
                          heap.Finish());
  EXPECT_EQ(130, result.heap_size);  // 70 + 60
  EXPECT_EQ(2, result.heap_results.size());

  EXPECT_TRUE(result.heap_results[0].chunk_map.contains(buffer_a_));
  EXPECT_TRUE(result.heap_results[0].chunk_map.contains(buffer_c_));
  EXPECT_EQ(10, result.heap_results[0].chunk_map.at(buffer_a_).size);
  EXPECT_EQ(50, result.heap_results[0].chunk_map.at(buffer_c_).size);
  EXPECT_EQ(60, result.heap_results[0].chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(0, result.heap_results[0].chunk_map.at(buffer_c_).offset);
}

TEST_F(ConstrainedGlobalDecreasingSizeBestFitHeapTest, ColocatedII) {
  // space
  //   ^
  //   |       +---------------+
  //   |       +-------b-------+
  //   | ....................
  //   |+------+      +-------+
  //   ||      |      |       |
  //   ||      |      |       | <--- colocate with a
  //   |+--a---+      +---c---+
  //   ---------------------> time
  ConstrainedGlobalDecreasingSizeBestFitHeap heap(/*size_limit_per_heap=*/50,
                                                  /*alignment=*/20);
  heap.Alloc(buffer_a_, 30);
  heap.Free(buffer_a_, 30);
  heap.Alloc(buffer_b_, 20);

  heap.ShareWith(buffer_c_, buffer_a_, 40);
  heap.Free(buffer_c_, 40);
  heap.Free(buffer_b_, 20);

  TF_ASSERT_OK_AND_ASSIGN(const HeapSimulator::Result<HloValue> result,
                          heap.Finish());
  EXPECT_EQ(60, result.heap_size);  // 40 + 20
  EXPECT_EQ(2, result.heap_results.size());

  EXPECT_TRUE(result.heap_results[0].chunk_map.contains(buffer_a_));
  EXPECT_TRUE(result.heap_results[0].chunk_map.contains(buffer_c_));
  EXPECT_EQ(30, result.heap_results[0].chunk_map.at(buffer_a_).size);
  EXPECT_EQ(40, result.heap_results[0].chunk_map.at(buffer_c_).size);
  EXPECT_EQ(0, result.heap_results[0].chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(0, result.heap_results[0].chunk_map.at(buffer_c_).offset);
}

class IntervalTreeTest : public ::testing::Test {};

TEST_F(IntervalTreeTest, InsertAndRemove) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(1, 2);
  BufferIntervalTree tree;
  tree.Add(1, 2, chunk);
  EXPECT_TRUE(tree.Remove(1, 2, chunk));
  EXPECT_FALSE(tree.Remove(1, 2, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
  // Do it again.
  tree.Add(1, 2, chunk);
  EXPECT_TRUE(tree.Remove(1, 2, chunk));
  EXPECT_FALSE(tree.Remove(1, 2, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

TEST_F(IntervalTreeTest, InsertAndRemoveTwoLevelsLeft) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(
      1, 2);  // Value in chunk doesn't matter here.
  //    [20, 36] (45)
  //     /
  //  [1, 45] (45)

  BufferIntervalTree tree;
  tree.Add(20, 36, chunk);
  tree.Add(1, 45, chunk);
  EXPECT_TRUE(tree.Remove(1, 45, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 36);
  EXPECT_TRUE(tree.Remove(20, 36, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

TEST_F(IntervalTreeTest, InsertAndRemoveTwoLevelsRight) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(
      1, 2);  // Value in chunk doesn't matter here.
  //    [20, 36] (45)
  //          \
  //         [21, 45] (45)
  BufferIntervalTree tree;
  tree.Add(20, 36, chunk);
  tree.Add(21, 45, chunk);
  EXPECT_TRUE(tree.Remove(21, 45, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 36);
  EXPECT_TRUE(tree.Remove(20, 36, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

TEST_F(IntervalTreeTest, TwoLevelsRight_RootFirst) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(
      1, 2);  // Value in chunk doesn't matter here.
  //    [20, 36] (45)
  //          \
  //         [21, 45] (45)
  BufferIntervalTree tree;
  tree.Add(20, 36, chunk);
  tree.Add(21, 45, chunk);
  EXPECT_TRUE(tree.Remove(20, 36, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 45);
  EXPECT_EQ(tree.GetRoot()->start, 21);
  EXPECT_EQ(tree.GetRoot()->end, 45);
  EXPECT_EQ(tree.GetRoot()->left, nullptr);
  EXPECT_EQ(tree.GetRoot()->right, nullptr);
  EXPECT_TRUE(tree.Remove(21, 45, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

TEST_F(IntervalTreeTest, TwoLevelsLeft_RootFirst) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(
      1, 2);  // Value in chunk doesn't matter here.
  //    [20, 36] (45)
  //      /
  //  [1, 45] (45)
  BufferIntervalTree tree;
  tree.Add(20, 36, chunk);
  tree.Add(1, 45, chunk);
  EXPECT_TRUE(tree.Remove(20, 36, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 45);
  EXPECT_EQ(tree.GetRoot()->start, 1);
  EXPECT_EQ(tree.GetRoot()->end, 45);
  EXPECT_EQ(tree.GetRoot()->left, nullptr);
  EXPECT_EQ(tree.GetRoot()->right, nullptr);
  EXPECT_TRUE(tree.Remove(1, 45, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

TEST_F(IntervalTreeTest, ThreeLevelsRight) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(
      1, 2);  // Value in chunk doesn't matter here.
  //    [20, 36] (45)
  //          \
  //         [21, 45] (45)
  //              \
  //              [22, 40] (40)
  BufferIntervalTree tree;
  tree.Add(20, 36, chunk);
  tree.Add(21, 45, chunk);
  tree.Add(22, 40, chunk);
  EXPECT_TRUE(tree.Remove(21, 45, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 40);
  EXPECT_TRUE(tree.Remove(20, 36, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 40);
  EXPECT_TRUE(tree.Remove(22, 40, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}
TEST_F(IntervalTreeTest, ThreeLevelsLeftLeft) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(
      1, 2);  // Value in chunk doesn't matter here.
  //    [20, 36] (45)
  //       /
  //  [10, 45] (45)
  //      /
  // [1, 40] (40)
  BufferIntervalTree tree;
  tree.Add(20, 36, chunk);
  tree.Add(10, 45, chunk);
  tree.Add(1, 40, chunk);
  EXPECT_TRUE(tree.Remove(10, 45, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 40);
  EXPECT_TRUE(tree.Remove(1, 40, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 36);
  EXPECT_TRUE(tree.Remove(20, 36, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

TEST_F(IntervalTreeTest, ThreeLevelsLeftRight) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(
      1, 2);  // Value in chunk doesn't matter here.
  //    [20, 36] (45)
  //       /
  //  [10, 45] (45)
  //      \
  //     [15, 40] (40)
  BufferIntervalTree tree;
  tree.Add(20, 36, chunk);
  tree.Add(10, 45, chunk);
  tree.Add(15, 40, chunk);
  EXPECT_TRUE(tree.Remove(10, 45, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 40);
  EXPECT_TRUE(tree.Remove(15, 40, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 36);
  EXPECT_TRUE(tree.Remove(20, 36, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

TEST_F(IntervalTreeTest, ThreeLevelsRightLeft) {
  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(
      1, 2);  // Value in chunk doesn't matter here.
  //    [20, 36] (45)
  //          \
  //         [25, 45] (45)
  //           /
  //       [22, 40] (40)
  BufferIntervalTree tree;
  tree.Add(20, 36, chunk);
  tree.Add(25, 45, chunk);
  tree.Add(22, 40, chunk);
  EXPECT_TRUE(tree.Remove(25, 45, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 40);
  EXPECT_TRUE(tree.Remove(20, 36, chunk));
  EXPECT_EQ(tree.GetRoot()->subtree_end, 40);
  EXPECT_TRUE(tree.Remove(22, 40, chunk));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

TEST_F(IntervalTreeTest, ThreeLevelsRightLeftChunkDifferent) {
  HeapSimulator::Chunk chunk1 = HeapSimulator::Chunk::FromOffsetSize(1, 2);
  HeapSimulator::Chunk chunk2 = HeapSimulator::Chunk::FromOffsetSize(2, 3);
  HeapSimulator::Chunk chunk3 = HeapSimulator::Chunk::FromOffsetSize(3, 4);
  //    [20, 36] (45) Chunk1({1, 2})
  //          \
  //         [25, 45] (45) Chunk2({2, 3})
  //           /
  //       [22, 40] (40) Chunk3({3, 4})
  BufferIntervalTree tree;
  tree.Add(20, 36, chunk1);
  tree.Add(25, 45, chunk2);
  tree.Add(22, 40, chunk3);
  EXPECT_TRUE(tree.Remove(25, 45, chunk2));
  // Chunk 1 is till the root after removing chunk 2.
  EXPECT_EQ(tree.GetRoot()->subtree_end, 40);
  EXPECT_EQ(tree.GetRoot()->chunk.offset, 1);
  EXPECT_EQ(tree.GetRoot()->chunk.size, 2);
  EXPECT_TRUE(tree.Remove(20, 36, chunk1));
  // Chunk 3 becomes the root now.
  EXPECT_EQ(tree.GetRoot()->subtree_end, 40);
  EXPECT_EQ(tree.GetRoot()->chunk.offset, 3);
  EXPECT_EQ(tree.GetRoot()->chunk.size, 4);
  EXPECT_TRUE(tree.Remove(22, 40, chunk3));
  ASSERT_EQ(tree.GetRoot(), nullptr);
}

class SlicedBufferIntervalTest : public ::testing::Test {
 public:
  using HeapTy = GlobalDecreasingSizeBestFitHeap<HloValue>;
  using ColocationTy = absl::InlinedVector<const HloValue*, 2>;

  static std::tuple<const HloValue*, int64_t, int64_t, int64_t,
                    const ColocationTy&, bool>
  BufferIntervalToTuple(const HeapTy::BufferInterval& buffer_interval) {
    return std::make_tuple(buffer_interval.buffer, buffer_interval.size,
                           buffer_interval.start, buffer_interval.end,
                           std::ref(buffer_interval.colocations),
                           buffer_interval.need_allocation);
  }

  SlicedBufferIntervalTest() {
    HloModuleConfig config;
    module_ = std::make_unique<HloModule>("TestModule", config);

    Shape f32vec4 = ShapeUtil::MakeShape(F32, {4});

    auto builder = HloComputation::Builder("TestComputation");
    auto p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, f32vec4, "p0"));
    auto p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, f32vec4, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(f32vec4, HloOpcode::kAdd, p0, p1));

    module_->AddEntryComputation(builder.Build());

    p0_value_ = std::make_unique<HloValue>(0, p0, ShapeIndex{});
    p1_value_ = std::make_unique<HloValue>(0, p1, ShapeIndex{});

    full_buffer_interval_ = HeapTy::BufferInterval({
        p0_value_.get(),
        /*size=*/20,
        /*start=*/100,
        /*end=*/200,
        /*colocations=*/{p1_value_.get()},
        /*need_allocation=*/true,
    });
    sliced_buffer_interval_ = std::make_unique<HeapTy::SlicedBufferInterval>(
        HeapTy::SlicedBufferInterval::CreateConstInterval(
            full_buffer_interval_));
    mutable_sliced_buffer_interval_ =
        std::make_unique<HeapTy::SlicedBufferInterval>(
            HeapTy::SlicedBufferInterval::CreateMutableInterval(
                full_buffer_interval_));
  }

 protected:
  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloValue> p0_value_;
  std::unique_ptr<HloValue> p1_value_;
  HeapTy::BufferInterval full_buffer_interval_;
  std::unique_ptr<const HeapTy::SlicedBufferInterval> sliced_buffer_interval_;
  std::unique_ptr<HeapTy::SlicedBufferInterval> mutable_sliced_buffer_interval_;
};

TEST_F(SlicedBufferIntervalTest, NoSlices) {
  EXPECT_EQ(
      BufferIntervalToTuple(sliced_buffer_interval_->full_buffer_interval()),
      BufferIntervalToTuple(full_buffer_interval_));
  EXPECT_EQ(sliced_buffer_interval_->num_slices(), 1);
  EXPECT_THAT(sliced_buffer_interval_->SliceSizesSortedByOffset(),
              ::testing::ElementsAre(20));
  EXPECT_EQ(BufferIntervalToTuple(
                sliced_buffer_interval_->IntervalForMakeFreeChunks(0)),
            BufferIntervalToTuple(full_buffer_interval_));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->full_buffer_interval()),
            BufferIntervalToTuple(full_buffer_interval_));
}

TEST_F(SlicedBufferIntervalTest, Sliced) {
  std::vector<int64_t> slice_sizes = {4, 5, 5, 6};
  mutable_sliced_buffer_interval_->Slice(absl::Span<int64_t>(slice_sizes));

  EXPECT_EQ(mutable_sliced_buffer_interval_->num_slices(), 4);
  EXPECT_THAT(mutable_sliced_buffer_interval_->SliceSizesSortedByOffset(),
              ::testing::ElementsAre(4, 5, 5, 6));

  mutable_sliced_buffer_interval_->UpdateInclusiveSliceStartTimes(
      {100, 125, 150, 175});

  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(0)),
            BufferIntervalToTuple(
                {p0_value_.get(), 4, 100, 124, ColocationTy(), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(1)),
            BufferIntervalToTuple(
                {p0_value_.get(), 4, 125, 149, ColocationTy(), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(2)),
            BufferIntervalToTuple(
                {p0_value_.get(), 4, 150, 174, ColocationTy(), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(3)),
            BufferIntervalToTuple({p0_value_.get(), 20, 175, 200,
                                   ColocationTy({p1_value_.get()}), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->full_buffer_interval()),
            BufferIntervalToTuple({p0_value_.get(), 20, 100, 200,
                                   ColocationTy({p1_value_.get()}), true}));

  mutable_sliced_buffer_interval_->UpdateExclusiveSliceStartTimes(
      {100, 125, 150, 175});

  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(0)),
            BufferIntervalToTuple(
                {p0_value_.get(), 4, 101, 125, ColocationTy(), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(1)),
            BufferIntervalToTuple(
                {p0_value_.get(), 4, 126, 150, ColocationTy(), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(2)),
            BufferIntervalToTuple(
                {p0_value_.get(), 4, 151, 175, ColocationTy(), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(3)),
            BufferIntervalToTuple({p0_value_.get(), 20, 176, 200,
                                   ColocationTy({p1_value_.get()}), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->full_buffer_interval()),
            BufferIntervalToTuple({p0_value_.get(), 20, 101, 200,
                                   ColocationTy({p1_value_.get()}), true}));

  mutable_sliced_buffer_interval_->UpdateEndTime(300);

  // Only the BufferInterval for the last slice time should have changed end
  // times.
  EXPECT_EQ(mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(2).end,
            175);
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->IntervalForMakeFreeChunks(3)),
            BufferIntervalToTuple({p0_value_.get(), 20, 176, 300,
                                   ColocationTy({p1_value_.get()}), true}));
  EXPECT_EQ(BufferIntervalToTuple(
                mutable_sliced_buffer_interval_->full_buffer_interval()),
            BufferIntervalToTuple({p0_value_.get(), 20, 101, 300,
                                   ColocationTy({p1_value_.get()}), true}));
}

class SlicedAllocationFinderTest : public ::testing::Test {
 public:
  using HeapTy = GlobalDecreasingSizeBestFitHeap<HloValue>;
  using FreeChunks = typename HeapTy::FreeChunks;
  using Chunk = HeapSimulator::Chunk;
  using Finder = typename HeapTy::SlicedAllocationFinder;

 protected:
  std::unique_ptr<SliceTimePermutationIterator> NewPermutationIterator(
      int64_t num_slices) {
    // For these tests, map each slice time to a unique incrementing start time.
    std::vector<int64_t> inclusive_start_times;
    inclusive_start_times.reserve(num_slices);
    for (int64_t start_time = 0; start_time < num_slices; ++start_time) {
      inclusive_start_times.push_back(start_time);
    }

    return SliceTimePermutationIterator::CreateForNewAllocation(
        SliceTimePermutationIterator::Ty::kAll, inclusive_start_times);
  }
};

TEST_F(SlicedAllocationFinderTest, NoSlices) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t0 |xxxxx  xxx                              xxxxx000xxxxxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space

The full buffer goes in the smallest chunk that fits.
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {45, 48},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(Chunk::FromOffsetSize(45, 3),
                                     Chunk::FromOffsetSize(48, 0)));
}

TEST_F(SlicedAllocationFinderTest, NoSlicesLargerMaxColloc) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t0 |xxxxx  xxx                              xxxxx   xxxxxxxxxxxx000       x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space

The max colocation size does not fit in the smallest free chunk.
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {45, 48},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3};
  int64_t max_colocation_size = 6;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(Chunk::FromOffsetSize(60, 3),
                                     Chunk::FromOffsetSize(63, 3)));
}

TEST_F(SlicedAllocationFinderTest, NoSlicesSmallestTie) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t0 |xxxxx  xxx000xx                         xxxxx   xxxxxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space

Multiple free chunks have size 3. We pick the one with the smallest offset.
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 13},
          {15, 40},
          {45, 48},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(Chunk::FromOffsetSize(10, 3),
                                     Chunk::FromOffsetSize(13, 0)));
}

TEST_F(SlicedAllocationFinderTest, LeftHole) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx000111222xxxxxx          x
t1 |xxxxx  xxx                              xxxxx000111xxxxxxxxx          x
t0 |xxxxx  xxx                              xxxxx000xxxxxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {45, 48},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {45, 51},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(45, 3), Chunk::FromOffsetSize(48, 3),
                  Chunk::FromOffsetSize(51, 3), Chunk::FromOffsetSize(54, 0)));
}

TEST_F(SlicedAllocationFinderTest, RightHole) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx000111222xxxxxx          x
t1 |xxxxx  xxx                              xxxxxxxx111222xxxxxx          x
t0 |xxxxx  xxx                              xxxxxxxxxxx222xxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {51, 54},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {48, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(51, 3), Chunk::FromOffsetSize(48, 3),
                  Chunk::FromOffsetSize(45, 3), Chunk::FromOffsetSize(54, 0)));
}

TEST_F(SlicedAllocationFinderTest, MiddleHole) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx000111222xxxxxx          x
t1 |xxxxx  xxx                              xxxxxxxx111222xxxxxx          x
t0 |xxxxx  xxx                              xxxxxxxx111xxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {48, 51},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {48, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(48, 3), Chunk::FromOffsetSize(51, 3),
                  Chunk::FromOffsetSize(45, 3), Chunk::FromOffsetSize(54, 0)));
}

TEST_F(SlicedAllocationFinderTest, ManyHoles) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx                          xxxxx    000111222              xx  xxxxxxx
t1 |xxxxx                          xxxxxxx  000 xx222  xxx     xxx  xxxxxxx
t0 |xxxxx                          xxxxxxxx   xxxx222  xxx     xxx  xxxxxxx
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space

Note, the free chunk @t1 offset 38 is not aligned with the free chunk @t0
offset 39 in a way that would fit any offset of the slices. (A slice can't be
subsliced by MSA.)
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 31},
          {39, 42},
          {46, 51},
          {54, 60},
          {62, 64},
      },
      // Slice time 1
      {
          {5, 31},
          {38, 44},
          {46, 51},
          {54, 59},
          {62, 64},
      },
      // Slice time 2
      {
          {5, 31},
          {36, 59},
          {62, 64},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(46, 3), Chunk::FromOffsetSize(40, 3),
                  Chunk::FromOffsetSize(43, 3), Chunk::FromOffsetSize(49, 0)));
}

TEST_F(SlicedAllocationFinderTest, EarlySliceTimesHaveLargeFreeChunks) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx000111222xxxxxx          x
t1 |xxxxx                    xxx            xxxxxxxx111222xxxxxx          x
t0 |xxxxxx                                          111                 xxx
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {6, 68},
      },
      // Slice time 1
      {
          {5, 25},
          {28, 40},
          {48, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(48, 3), Chunk::FromOffsetSize(51, 3),
                  Chunk::FromOffsetSize(45, 3), Chunk::FromOffsetSize(54, 0)));
}

TEST_F(SlicedAllocationFinderTest, DifferentSliceSizes1) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xx000001112222xxxxxx          x
t1 |xxxxx  xxx                              xxxxxx 1112222xxxxxx          x
t0 |xxxxx  xxx                              xxxxxx 111 xxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {46, 51},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {46, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {42, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {5, 3, 4};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(47, 3), Chunk::FromOffsetSize(50, 4),
                  Chunk::FromOffsetSize(42, 5), Chunk::FromOffsetSize(54, 0)));
}

TEST_F(SlicedAllocationFinderTest, DifferentSliceSizes2) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx000001112222                  xx            xxxxxx          x
t1 |xxxxx  xxx00000111                      xxxxxx        xxxxxx          x
t0 |xxxxx  xxx00000                         xxxxxx   xxxxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {46, 49},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {46, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {42, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {5, 3, 4};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(10, 5), Chunk::FromOffsetSize(15, 3),
                  Chunk::FromOffsetSize(18, 4), Chunk::FromOffsetSize(22, 0)));
}

TEST_F(SlicedAllocationFinderTest, ZeroSizeFreeChunk) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxxxxxxx                              xxxxx         xxxxxx000111222 x
t1 |xxxxxxxxxx                              xxxxx      xxxxxxxxx000111    x
t0 |xxxxxxxxxx                              xxxxxxxxxxxxxxxxxxxx000       x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 5},
          {10, 40},
          {45, 48},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {45, 51},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 45},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(60, 3), Chunk::FromOffsetSize(63, 3),
                  Chunk::FromOffsetSize(66, 3), Chunk::FromOffsetSize(69, 0)));
}

TEST_F(SlicedAllocationFinderTest, LargerMaxColloc) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx         xxxxxx000111222 x
t1 |xxxxx  xxx                              xxxxxxxx      xxxxxx000111    x
t0 |xxxxx  xxx                              xxxxxxxx   xxxxxxxxx000       x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
  */
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {48, 51},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {48, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = 10;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(60, 3), Chunk::FromOffsetSize(63, 3),
                  Chunk::FromOffsetSize(66, 3), Chunk::FromOffsetSize(69, 1)));
}

TEST_F(SlicedAllocationFinderTest, PreferredOffsetFit) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx          000111222           xxxxx         xxxxxx          x
t1 |xxxxx  xxx          000111              xxxxxxxx      xxxxxx          x
t0 |xxxxx  xxx          000                 xxxxxxxx   xxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {48, 51},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {48, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = 20;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(20, 3), Chunk::FromOffsetSize(23, 3),
                  Chunk::FromOffsetSize(26, 3), Chunk::FromOffsetSize(29, 0)));
}

TEST_F(SlicedAllocationFinderTest, PreferredOffsetNoFit) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx000111222xxxxxx          x
t1 |xxxxx  xxx                              xxxxxxxx111222xxxxxx          x
t0 |xxxxx  xxx                              xxxxxxxx111xxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space

The sliced allocation does not fit at the preferred offset.
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {48, 51},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {48, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = 35;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(48, 3), Chunk::FromOffsetSize(51, 3),
                  Chunk::FromOffsetSize(45, 3), Chunk::FromOffsetSize(54, 0)));
}

TEST_F(SlicedAllocationFinderTest, Misaligned) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxx 000011112222 xxx          x
t1 |xxxxx  xxx                              xxxxxxx 11112222 xxx          x
t0 |xxxxx  xxx                              xxxxxxx 1111 xxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space

The IsSliceOffsetAllowedFn is set such that we are only allowed to start slices
on spatial boundaries of 2.
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {47, 53},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {47, 57},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {43, 57},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {4, 4, 4};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 2;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(48, 4), Chunk::FromOffsetSize(52, 4),
                  Chunk::FromOffsetSize(44, 4), Chunk::FromOffsetSize(56, 0)));
}

TEST_F(SlicedAllocationFinderTest, PreferredOffsetMisaligned) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxx 000011112222 xxx          x
t1 |xxxxx  xxx                              xxxxxxx 11112222 xxx          x
t0 |xxxxx  xxx                              xxxxxxx 1111 xxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space

The IsSliceOffsetAllowedFn is set such that we are only allowed to start slices
on spatial boundaries of 2.
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {47, 53},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {47, 57},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {43, 57},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {4, 4, 4};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = 21;
  int64_t alignment = 2;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(48, 4), Chunk::FromOffsetSize(52, 4),
                  Chunk::FromOffsetSize(44, 4), Chunk::FromOffsetSize(56, 0)));
}

TEST_F(SlicedAllocationFinderTest, CorrectInitialization1) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t1 |xxxxx000111xxxxxxxxxxxxxx      xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
t0 |xxxxx000   xxxx      xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 11},
          {15, 21},
      },
      // Slice time 1
      {
          {5, 11},
          {25, 31},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(Chunk::FromOffsetSize(5, 3),
                                     Chunk::FromOffsetSize(8, 3),
                                     Chunk::FromOffsetSize(11, 0)));
}

TEST_F(SlicedAllocationFinderTest, CorrectInitialization2) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t1 |xxxxx000111     xxxxxxxxxx      xxxxxxxxxx   xxxxxxxxxxxxxxxxxxxxxxxxxx
t0 |xxxxx000        xxxx      xxxxxxxxxxxxxx   xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 16},
          {20, 26},
          {40, 43},
      },
      // Slice time 1
      {
          {5, 16},
          {26, 32},
          {42, 45},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(free_chunks_per_slice_time, sorted_slice_sizes,
                max_colocation_size, preferred_offset, alignment,
                NewPermutationIterator(sorted_slice_sizes.size()));

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(Chunk::FromOffsetSize(5, 3),
                                     Chunk::FromOffsetSize(8, 3),
                                     Chunk::FromOffsetSize(11, 0)));
}

TEST_F(SlicedAllocationFinderTest, LeftHoleNotAllowedToStartAtFirstOffset) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx 000111222xxxxx          x
t1 |xxxxx  xxx                              xxxxx 000111xxxxxxxx          x
t0 |xxxxx  xxx                              xxxxx 000xxxxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {45, 49},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {45, 52},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 55},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(
      free_chunks_per_slice_time, sorted_slice_sizes, max_colocation_size,
      preferred_offset, alignment,
      NewPermutationIterator(sorted_slice_sizes.size()),
      /*is_offset_allowed=*/[](int64_t offset) { return offset != 45; });

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(46, 3), Chunk::FromOffsetSize(49, 3),
                  Chunk::FromOffsetSize(52, 3), Chunk::FromOffsetSize(55, 0)));
}

TEST_F(SlicedAllocationFinderTest, LeftHoleAllowedToIncludeNoStartOffset) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx000111222xxxxxx          x
t1 |xxxxx  xxx                              xxxxx000111xxxxxxxxx          x
t0 |xxxxx  xxx                              xxxxx000xxxxxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {45, 48},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {45, 51},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(
      free_chunks_per_slice_time, sorted_slice_sizes, max_colocation_size,
      preferred_offset, alignment,
      NewPermutationIterator(sorted_slice_sizes.size()),
      // We're not allowed to start at offset 46, but we can include it.
      /*is_offset_allowed=*/[](int64_t offset) { return offset != 46; });

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(45, 3), Chunk::FromOffsetSize(48, 3),
                  Chunk::FromOffsetSize(51, 3), Chunk::FromOffsetSize(54, 0)));
}

TEST_F(SlicedAllocationFinderTest, RightHoleNotAllowedToStartAtFirstOffset) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx 000111222xxxxx          x
t1 |xxxxx  xxx                              xxxxxxxx 111222xxxxx          x
t0 |xxxxx  xxx                              xxxxxxxxxxx 222xxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {51, 55},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {48, 55},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 55},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(
      free_chunks_per_slice_time, sorted_slice_sizes, max_colocation_size,
      preferred_offset, alignment,
      NewPermutationIterator(sorted_slice_sizes.size()),
      /*is_offset_allowed=*/[](int64_t offset) { return offset != 45; });

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(52, 3), Chunk::FromOffsetSize(49, 3),
                  Chunk::FromOffsetSize(46, 3), Chunk::FromOffsetSize(55, 0)));
}

TEST_F(SlicedAllocationFinderTest, RightHoleNotAllowedOffsetsFindsNewHole) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx         xxxxxx000111222 x
t1 |xxxxx  xxx                              xxxxxxxx      xxxxxx000111    x
t0 |xxxxx  xxx                              xxxxxxxxxxx   xxxxxx000       x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {51, 54},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {48, 54},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 54},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(
      free_chunks_per_slice_time, sorted_slice_sizes, max_colocation_size,
      preferred_offset, alignment,
      NewPermutationIterator(sorted_slice_sizes.size()),
      /*is_offset_allowed=*/[](int64_t offset) { return offset != 45; });

  EXPECT_THAT(finder.Find(),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(60, 3), Chunk::FromOffsetSize(63, 3),
                  Chunk::FromOffsetSize(66, 3), Chunk::FromOffsetSize(69, 0)));
}

TEST_F(SlicedAllocationFinderTest, FindForOffset) {
  /*
Slice time vs allocation space (x = previously allocated, <number> = index of
                                the slice that will be allocated at the
                                specified position and time):
   ^
t2 |xxxxx  xxx                              xxxxx          xxxxx          x
t1 |xxxxx  xxx                              xxxxx       xxxxxxxx          x
t0 |xxxxx  xxx                              xxxxx    xxxxxxxxxxx          x
   +!----|----!----|----!----|----!----|----!----|----!----|----!----|----!>
         space
*/
  std::vector<FreeChunks> free_chunks_per_slice_time = {
      // Slice time 0
      {
          {5, 7},
          {10, 40},
          {45, 49},
          {60, 70},
      },
      // Slice time 1
      {
          {5, 7},
          {10, 40},
          {45, 52},
          {60, 70},
      },
      // Slice time 2
      {
          {5, 7},
          {10, 40},
          {45, 55},
          {60, 70},
      },
  };
  std::vector<int64_t> sorted_slice_sizes = {3, 3, 3};
  int64_t max_colocation_size = -1;
  int64_t preferred_offset = -1;
  int64_t alignment = 1;

  Finder finder(
      free_chunks_per_slice_time, sorted_slice_sizes, max_colocation_size,
      preferred_offset, alignment,
      NewPermutationIterator(sorted_slice_sizes.size()),
      /*is_offset_allowed=*/[](int64_t offset) { return offset != 45; });

  EXPECT_THAT(finder.FindForOffset(10),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(10, 3), Chunk::FromOffsetSize(13, 3),
                  Chunk::FromOffsetSize(16, 3), Chunk::FromOffsetSize(19, 0)));
  EXPECT_THAT(finder.FindForOffset(20),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(20, 3), Chunk::FromOffsetSize(23, 3),
                  Chunk::FromOffsetSize(26, 3), Chunk::FromOffsetSize(29, 0)));
  EXPECT_THAT(finder.FindForOffset(45),
              /* Disallowed */
              ::testing::IsEmpty());
  EXPECT_THAT(finder.FindForOffset(46),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(46, 3), Chunk::FromOffsetSize(49, 3),
                  Chunk::FromOffsetSize(52, 3), Chunk::FromOffsetSize(55, 0)));
  EXPECT_THAT(finder.FindForOffset(59),
              /* No space */
              ::testing::IsEmpty());
  EXPECT_THAT(finder.FindForOffset(61),
              ::testing::ElementsAre(
                  Chunk::FromOffsetSize(61, 3), Chunk::FromOffsetSize(64, 3),
                  Chunk::FromOffsetSize(67, 3), Chunk::FromOffsetSize(70, 0)));
}

class SliceTimePermutationIteratorTest : public ::testing::Test {
 protected:
  struct NewAllocationTestCase {
    void Test() const {
      auto iterator = SliceTimePermutationIterator::CreateForNewAllocation(
          ty, inclusive_start_times);

      // Run the iterator multiple times to make sure it can be reused.
      for (int i = 0; i < 5; ++i) {
        VLOG(2) << "Test case try #" << i << ": NewAllocation, " << name;
        EXPECT_THAT(GetPermutations(iterator.get()),
                    ::testing::ElementsAreArray(expected_permutations))
            << "Failed NewAllocation, " << name;
      }
    }

    std::string name;
    SliceTimePermutationIterator::Ty ty;
    std::vector<int64_t> inclusive_start_times;
    std::vector<std::vector<int64_t>> expected_permutations;
  };

  struct RepackTestCase {
    void Test() const {
      auto iterator = SliceTimePermutationIterator::CreateForRepack(
          ty, (original_slice_data.has_value() ? &(*original_slice_data)
                                               : nullptr));

      // Run the iterator multiple times to make sure it can be reused.
      for (int i = 0; i < 5; ++i) {
        VLOG(2) << "Test case try #" << i << ": Repack, " << name;
        EXPECT_THAT(GetPermutations(iterator.get()),
                    ::testing::ElementsAreArray(expected_permutations))
            << "Failed Repack, " << name;
      }
    }

    std::string name;
    SliceTimePermutationIterator::Ty ty;
    std::optional<SlicedAllocationData> original_slice_data;
    std::vector<std::vector<int64_t>> expected_permutations;
  };

  static std::vector<std::vector<int64_t>> GetPermutations(
      SliceTimePermutationIterator* it) {
    std::vector<std::vector<int64_t>> results;
    for (it->Begin(); !it->Done(); it->Next()) {
      absl::Span<const int64_t> permutation = it->Get();
      results.push_back(
          std::vector<int64_t>(permutation.begin(), permutation.end()));
    }

    return results;
  }
};

TEST_F(SliceTimePermutationIteratorTest, NewAllocations) {
  std::vector<NewAllocationTestCase> test_cases = {
      {
          "0 slices, all permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*inclusive_start_times=*/{},
          /*expected_permutations=*/{},
      },
      {
          "1 slice, all permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*inclusive_start_times=*/{0},
          /*expected_permutations=*/{{0}},
      },
      {
          "2 slices, all permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*inclusive_start_times=*/{10, 20},
          /*expected_permutations=*/{{0, 1}, {1, 0}},
      },
      {
          "many slices, all permutations, unique start times",
          SliceTimePermutationIterator::Ty::kAll,
          /*inclusive_start_times=*/{40, 10, 450},
          /*expected_permutations=*/
          {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}},
      },
      {
          "many slices, all permutations, non-unique start times",
          SliceTimePermutationIterator::Ty::kAll,
          /*inclusive_start_times=*/{40, 10, 450, 10},
          /*expected_permutations=*/
          {
              // The two smallest start times are the same. Thus, when we
              // compare permutations for equivalence, if index i is assigned
              // slice time 0, and index j is assigned slice time 1, its
              // equivalent to i being assigned 1, and j being assigned 0.
              //
              // Note, the order of inclusive start times is irrelevant. The ith
              // earliest slice time is associated with the ith earliest
              // inclusive start time.
              {0, 1, 2, 3},
              {0, 1, 3, 2},
              {0, 2, 1, 3},
              {0, 2, 3, 1},
              {0, 3, 1, 2},
              {0, 3, 2, 1},
              // {1, 0, 2, 3}, equivalent emitted
              // {1, 0, 3, 2}, equivalent emitted
              // {1, 2, 0, 3}, equivalent emitted
              // {1, 2, 3, 0}, equivalent emitted
              // {1, 3, 0, 2}, equivalent emitted
              // {1, 3, 2, 0}, equivalent emitted
              {2, 0, 1, 3},
              {2, 0, 3, 1},
              // {2, 1, 0, 3}, equivalent emitted
              // {2, 1, 3, 0}, equivalent emitted
              {2, 3, 0, 1},
              // {2, 3, 1, 0}, equivalent emitted
              {3, 0, 1, 2},
              {3, 0, 2, 1},
              // {3, 1, 0, 2}, equivalent emitted
              // {3, 1, 2, 0}, equivalent emitted
              {3, 2, 0, 1},
              // {3, 2, 1, 0}, equivalent emitted
          },
      },
      {
          "0 slices, preferred permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*inclusive_start_times=*/{},
          /*expected_permutations=*/{},
      },
      {
          "1 slice, preferred permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*inclusive_start_times=*/{0},
          /*expected_permutations=*/{{0}},
      },
      {
          "2 slices, preferred permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*inclusive_start_times=*/{10, 20},
          /*expected_permutations=*/{{0, 1}, {1, 0}},
      },
      {
          "many slices, preferred permutations, unique start times",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*inclusive_start_times=*/{40, 10, 450, 12, 14},
          /*expected_permutations=*/
          {{0, 1, 2, 3, 4}, {4, 3, 2, 1, 0}, {3, 1, 0, 2, 4}},
      },
      {
          "many slices, preferred permutations, non-unique start times 1",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*inclusive_start_times=*/{40, 10, 450, 10},
          /*expected_permutations=*/
          {// This case is not impacted by non-unique start times.
           {0, 1, 2, 3},
           {3, 2, 1, 0},
           {3, 1, 0, 2}},
      },
      {
          "many slices, preferred permutations, non-unique start times 2",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*inclusive_start_times=*/{40, 40},
          /*expected_permutations=*/
          {
              // The two smallest start times are the same. Thus, we must ignore
              // duplicate permutations, when we ignore the order of slice times
              // 0 and 1.
              {0, 1},
              // This is a duplicate of {0, 1}, when ignoring the order of 0 and
              // 1.
              // {1, 0},
          },
      },
  };

  for (const NewAllocationTestCase& test_case : test_cases) {
    test_case.Test();
  }
}

TEST_F(SliceTimePermutationIteratorTest, Repacks) {
  std::vector<RepackTestCase> test_cases = {
      {
          "no slice data, all permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*original_slice_data=*/std::nullopt,
          /*expected_permutations=*/{{0}},
      },
      {
          "0 slices, all permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*original_slice_data=*/SlicedAllocationData{},
          /*expected_permutations=*/{},
      },
      {
          "1 slice, all permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
          }},
          /*expected_permutations=*/{{0}},
      },
      {
          "2 slices, uniform slice size, all permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
              {/*size=*/1, /*offset=*/2, /*inclusive_start_time=*/2},
          }},
          /*expected_permutations=*/{{0, 1}, {1, 0}},
      },
      {
          "many slices, uniform slice size, unique start times, all "
          "permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
              {/*size=*/1, /*offset=*/2, /*inclusive_start_time=*/2},
              {/*size=*/1, /*offset=*/3, /*inclusive_start_time=*/3},
          }},
          /*expected_permutations=*/
          {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}},
      },
      {
          "many slices, non-uniform slice size, unique start times, all "
          "permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
              {/*size=*/2, /*offset=*/2, /*inclusive_start_time=*/3},
              {/*size=*/1, /*offset=*/3, /*inclusive_start_time=*/2},
          }},
          /*expected_permutations=*/
          {
              // The slice at index 0 has a different size than any other slice,
              // so it's invalid to give it any slice time other than its
              // original slice time of 2.
              {0, 2, 1},
              {1, 2, 0},
          },
      },
      {
          "many slices, non-uniform slice size, non-unique start times, all "
          "permutations",
          SliceTimePermutationIterator::Ty::kAll,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
              {/*size=*/1, /*offset=*/2, /*inclusive_start_time=*/2},
              {/*size=*/2, /*offset=*/3, /*inclusive_start_time=*/1},
              {/*size=*/1, /*offset=*/5, /*inclusive_start_time=*/1},
              {/*size=*/2, /*offset=*/6, /*inclusive_start_time=*/3},
              {/*size=*/3, /*offset=*/8, /*inclusive_start_time=*/4},
          }},
          /*expected_permutations=*/
          {
              // All permutations such that:
              // * The first 3 slice times hold 2 slices with size 1, and 1
              //   slice with size 2.
              // * Slice time 3 holds a slice with size 1.
              // * Slice time 4 holds a slice with size 2.
              // * Slice time 5 holds a slice with size 3, which can only be the
              //   slice at index 5.
              // * We throw away permutations where the first 3 slice times are
              //   given to the same slice offsets.
              {0, 1, 2, 3, 4, 5},
              {0, 1, 4, 3, 2, 5},
              {0, 3, 1, 2, 4, 5},
              {0, 3, 4, 1, 2, 5},
              {3, 0, 1, 2, 4, 5},
              {3, 0, 4, 1, 2, 5},
          },
      },
      {
          "no slice data, preferred permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*original_slice_data=*/std::nullopt,
          /*expected_permutations=*/{{0}},
      },
      {
          "0 slices, preferred permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*original_slice_data=*/SlicedAllocationData{},
          /*expected_permutations=*/{},
      },
      {
          "1 slice, preferred permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
          }},
          /*expected_permutations=*/{{0}},
      },
      {
          "2 slices, uniform slice size, preferred permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
              {/*size=*/1, /*offset=*/2, /*inclusive_start_time=*/2},
          }},
          /*expected_permutations=*/{{0, 1}, {1, 0}},
      },
      {
          "many slices, uniform slice size, unique start times, preferred "
          "permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
              {/*size=*/1, /*offset=*/2, /*inclusive_start_time=*/2},
              {/*size=*/1, /*offset=*/3, /*inclusive_start_time=*/3},
          }},
          /*expected_permutations=*/
          {{0, 1, 2}, {2, 1, 0}, {1, 0, 2}},
      },
      {
          "many slices, non-uniform slice size, unique start times, preferred "
          "permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
              {/*size=*/2, /*offset=*/2, /*inclusive_start_time=*/3},
              {/*size=*/1, /*offset=*/3, /*inclusive_start_time=*/2},
          }},
          /*expected_permutations=*/
          {
              // The 2nd slice has a different size than the first, so we must
              // fix it to its original slice time, i.e., slice time 2.
              {0, 2, 1},
              {1, 2, 0},
          },
      },
      {
          "many slices, non-uniform slice size, non-unique start times, "
          "preferred permutations",
          SliceTimePermutationIterator::Ty::kPreferred,
          /*original_slice_data=*/
          SlicedAllocationData{/*slices_sorted_by_offset=*/{
              {/*size=*/1, /*offset=*/1, /*inclusive_start_time=*/1},
              {/*size=*/1, /*offset=*/2, /*inclusive_start_time=*/2},
              {/*size=*/2, /*offset=*/3, /*inclusive_start_time=*/1},
              {/*size=*/1, /*offset=*/5, /*inclusive_start_time=*/1},
              {/*size=*/2, /*offset=*/6, /*inclusive_start_time=*/3},
              {/*size=*/3, /*offset=*/8, /*inclusive_start_time=*/4},
          }},
          /*expected_permutations=*/
          {
              // First we fix the slice times of slices that have different
              // sizes than the first slice, i.e., slices 3, 5, and 6. If we
              // sort the slices by <inclusive_start_time, offset>, the fixed
              // slice time for those slices will be their index in the sorted
              // order. Thus, slice 3 is fixed to slice time 1. Slice 5 is fixed
              // to slice time 4. And, slice 6 is fixed to slice time 5.
              //
              // The remaining slices are given preferred slice times, throwing
              // out any equivalent permutations. Two permutations are
              // equivalent if they are equal, after ignoring permutations of
              // the slice times that map to the same inclusive start time. In
              // our case, slice times 0, 1, and 2 map to inclusive start
              // time 1. Thus, if indices i, j, and k are given slice times 0,
              // 1, and 2, it doesn't matter which of i, j, and k maps to 0, 1,
              // and 2 (for the purposes of equivalence).
              {0, 2, 1, 3, 4, 5},
              {3, 2, 1, 0, 4, 5},
              // The next permutation is the same as the previous, except slice
              // times 0 and 2 are permuted, so we throw it out.
              // {3, 0, 1, 2, 4, 5},
          },
      },
  };

  for (const RepackTestCase& test_case : test_cases) {
    test_case.Test();
  }
}

}  // namespace
}  // namespace xla
