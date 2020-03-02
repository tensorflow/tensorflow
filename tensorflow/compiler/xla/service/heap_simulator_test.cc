/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/heap_simulator.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

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

  EXPECT_EQ(
      25,
      HeapSimulator::MinimumMemoryForModule(schedule, size_fn).ValueOrDie());
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
  absl::flat_hash_map<const HloComputation*, int64> memory_by_computation;
  memory_by_computation[cond_computation] = 5;
  memory_by_computation[body_computation] = 16;

  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module.get()).ValueOrDie();

  // HeapSimulator accounts for subcomputations. The output buffer is aliased,
  // so we don't double count.
  EXPECT_EQ(64, HeapSimulator::MinimumMemoryForComputation(
                    *entry_computation, schedule.sequence(entry_computation),
                    *alias_analysis, size_fn, &memory_by_computation)
                    .ValueOrDie());
}

const char kAlloc[] = "Alloc";
const char kFree[] = "Free";
const char kShare[] = "Share";
const char kFinish[] = "Finish";

// CallSequence records a sequence of Alloc/Free/Finish calls.
using CallSequence = std::vector<std::pair<string, const HloValue*>>;

// HeapCallRecorder is a dummy heap algorithm that simply records its calls.
class HeapCallRecorder : public HeapAlgorithm {
 public:
  explicit HeapCallRecorder(CallSequence* calls) : calls_(calls) {}
  ~HeapCallRecorder() override {}

  void Alloc(const HloValue* buffer, int64 size) override {
    calls_->emplace_back(kAlloc, buffer);
    // Instead of assigning a real offset, we set the cardinality of the Alloc
    // call.  This isn't a valid assignment, but allows us to easily test for
    // buffer sharing.
    const int64 offset = result_.chunk_map.size();
    result_.chunk_map.emplace(buffer, Chunk{offset, size});
  }

  void ShareWith(const HloValue* buffer, const HloValue* shared,
                 int64 size) override {
    calls_->emplace_back(kShare, buffer);
    // Instead of assigning a real offset, we set the cardinality of the Alloc
    // call.  This isn't a valid assignment, but allows us to easily test for
    // buffer sharing.
    const int64 offset = result_.chunk_map[shared].offset;
    result_.chunk_map.emplace(buffer, Chunk{offset, size});
  }
  void Free(const HloValue* buffer, int64 size) override {
    calls_->emplace_back(kFree, buffer);
  }
  Result Finish() override {
    calls_->emplace_back(kFinish, nullptr);
    return result_;
  }

 private:
  CallSequence* calls_;
  Result result_;
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
      const string& name, std::unique_ptr<HloComputation> entry_computation,
      const std::vector<HloInstruction*>& instruction_sequence,
      const std::vector<HloInstruction*>& must_alias_set = {},
      const HloDataflowAnalysis::CanShareBuffer& can_share_buffer = nullptr) {
    HloModuleConfig config;
    module_ = absl::make_unique<HloModule>(name, config);
    module_->AddEntryComputation(std::move(entry_computation));
    Init(instruction_sequence, can_share_buffer);
  }

  explicit HeapSimulatorTracker(const string& name) {
    HloModuleConfig config;
    module_ = absl::make_unique<HloModule>(name, config);
  }

  // Similar to the single entry computation constructor above, but runs the
  // simulation over the entire module.
  void RunWholeModule(
      const std::vector<HloInstruction*>& full_module_sequence) {
    alias_analysis_ = HloAliasAnalysis::Run(module_.get()).ConsumeValueOrDie();

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
    auto algorithm = absl::make_unique<HeapCallRecorder>(&actual_calls_);
    result_ = HeapSimulator::Run(std::move(algorithm), *module_, schedule,
                                 *alias_analysis_, size_fn)
                  .ConsumeValueOrDie();
  }

  HloModule* module() { return module_.get(); }

  // Returns the buffer defined at the given instruction and index.
  const HloValue* BufferAt(const HloInstruction* instruction,
                           const ShapeIndex& index) const {
    return &alias_analysis_->dataflow_analysis().GetUniqueValueAt(instruction,
                                                                  index);
  }

  int64 OffsetAt(const HloInstruction* instruction, const ShapeIndex& index) {
    const HloValue* buffer = BufferAt(instruction, index);
    return result_.chunk_map.at(buffer).offset;
  }

  // Ensures the expected sequence of Alloc/Free/Finish calls was performed.
  void ExpectCallSequence(const CallSequence& expected) const {
    auto to_string = [](const CallSequence& sequence) {
      std::string output;
      for (int64 i = 0; i < sequence.size(); ++i) {
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
    int64 offset_a = OffsetAt(instruction_a, index_a);
    int64 offset_b = OffsetAt(instruction_b, index_b);
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
    auto algorithm = absl::make_unique<HeapCallRecorder>(&actual_calls_);

    alias_analysis_ =
        HloAliasAnalysis::Run(module_.get(), can_share_buffer).ValueOrDie();

    HeapSimulator::Options options;

    result_ =
        HeapSimulator::Run(std::move(algorithm), *module_->entry_computation(),
                           HloInstructionSequence(instruction_sequence),
                           *alias_analysis_, zero_size, options)
            .ConsumeValueOrDie();
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  CallSequence actual_calls_;
  HeapSimulator::Result result_;
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
         const ShapeIndex& user_index) -> absl::optional<bool> {
    if (instr->opcode() == HloOpcode::kFusion) {
      return true;
    }
    return false;
  };

  HloModuleConfig config;
  auto module = absl::make_unique<HloModule>(TestName(), config);

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
         const ShapeIndex& user_index) -> absl::optional<bool> {
    if (instr->opcode() == HloOpcode::kFusion) {
      return true;
    }
    return false;
  };

  HloModuleConfig config;
  auto module = absl::make_unique<HloModule>(TestName(), config);

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
  int64 output_buffer_0 = tracker.OffsetAt(fusion, {0});
  int64 output_buffer_1 = tracker.OffsetAt(fusion, {1});
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
    buffers_.emplace_back(
        absl::make_unique<HloValue>(id, const0, ShapeIndex{}));
    return buffers_.back().get();
  }

  HloComputation::Builder builder_;
  std::vector<std::unique_ptr<HloValue>> buffers_;
};

class NoFragmentationStatsHeapTest : public HeapAlgorithmTestBase {};

TEST_F(NoFragmentationStatsHeapTest, Empty) {
  NoFragmentationStatsHeap heap;
  EXPECT_EQ(0, heap.Finish().heap_size);
}

TEST_F(NoFragmentationStatsHeapTest, Simple) {
  NoFragmentationStatsHeap heap;
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 20);
  heap.Alloc(buffer_c_, 30);
  heap.Alloc(buffer_d_, 30);
  heap.Free(buffer_a_, 10);
  heap.Free(buffer_b_, 20);
  heap.Free(buffer_c_, 30);
  heap.Free(buffer_d_, 30);
  EXPECT_EQ(90, heap.Finish().heap_size);
}

TEST_F(NoFragmentationStatsHeapTest, Mixed) {
  NoFragmentationStatsHeap heap;
  heap.Alloc(buffer_a_, 10);  // max: A

  heap.Alloc(buffer_b_, 20);  // max: A+B
  heap.Free(buffer_b_, 20);

  heap.Alloc(buffer_c_, 30);  // max: A+C
  heap.Free(buffer_c_, 30);

  heap.Alloc(buffer_d_, 5);  // max: A+C
  heap.Free(buffer_d_, 5);

  heap.Free(buffer_a_, 10);
  EXPECT_EQ(40, heap.Finish().heap_size);
}

class GlobalDecreasingSizeBestFitHeapTest : public HeapAlgorithmTestBase {
 protected:
  class InheritedGlobalDecreasingSizeBestFitHeap
      : public GlobalDecreasingSizeBestFitHeap {
   public:
    InheritedGlobalDecreasingSizeBestFitHeap()
        : GlobalDecreasingSizeBestFitHeap(/*alignment=*/1) {}

    // Finds a chunk candidate and returns the offset and the new heap size.
    std::pair<int64, int64> FindChunkCandidate(const HloValue* buffer,
                                               int64 size, int64 start,
                                               int64 end,
                                               int64 preferred_offset = -1) {
      buffer_interval_.buffer = buffer;
      buffer_interval_.size = size;
      buffer_interval_.start = start;
      buffer_interval_.end = end;
      chunk_candidate_ = GlobalDecreasingSizeBestFitHeap::FindChunkCandidate(
          buffer_interval_, preferred_offset);
      EXPECT_EQ(chunk_candidate_.chunk.size, size);
      return {chunk_candidate_.chunk.offset, chunk_candidate_.heap_size};
    }

    // Commits the previously found chunk candidate.
    void CommitChunk() {
      GlobalDecreasingSizeBestFitHeap::CommitChunk(buffer_interval_,
                                                   chunk_candidate_);
    }

   private:
    BufferInterval buffer_interval_;
    ChunkCandidate chunk_candidate_;
  };

  InheritedGlobalDecreasingSizeBestFitHeap heap_;
};

TEST_F(GlobalDecreasingSizeBestFitHeapTest, Empty) {
  GlobalDecreasingSizeBestFitHeap heap(/*alignment=*/1);
  const HeapSimulator::Result result = heap.Finish();
  EXPECT_EQ(0, result.heap_size);
  EXPECT_EQ(0, result.chunk_map.size());
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
  GlobalDecreasingSizeBestFitHeap heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 30);
  heap.Alloc(buffer_c_, 20);
  heap.Alloc(buffer_d_, 40);
  heap.Free(buffer_a_, 10);
  heap.Free(buffer_b_, 30);
  heap.Free(buffer_c_, 20);
  heap.Free(buffer_d_, 40);

  const HeapSimulator::Result result = heap.Finish();
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
  GlobalDecreasingSizeBestFitHeap heap(/*alignment=*/20);
  heap.Alloc(buffer_a_, 10);
  heap.Alloc(buffer_b_, 20);
  heap.Alloc(buffer_c_, 50);
  heap.Free(buffer_a_, 10);
  heap.Alloc(buffer_d_, 40);
  heap.Free(buffer_b_, 20);
  heap.Free(buffer_c_, 50);
  heap.Free(buffer_d_, 40);

  const HeapSimulator::Result result = heap.Finish();
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
  GlobalDecreasingSizeBestFitHeap heap(/*alignment=*/1);
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

  const HeapSimulator::Result result = heap.Finish();
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
  GlobalDecreasingSizeBestFitHeap heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 40);
  heap.Free(buffer_a_, 40);
  heap.Alloc(buffer_b_, 20);
  heap.Free(buffer_b_, 20);
  heap.ShareWith(buffer_c_, buffer_a_, 40);
  heap.Free(buffer_c_, 40);

  const HeapSimulator::Result result = heap.Finish();
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
  GlobalDecreasingSizeBestFitHeap heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 40);
  heap.Free(buffer_a_, 40);
  heap.Alloc(buffer_b_, 20);

  heap.ShareWith(buffer_c_, buffer_a_, 40);
  heap.Free(buffer_c_, 40);
  heap.Free(buffer_b_, 20);

  const HeapSimulator::Result result = heap.Finish();
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
  GlobalDecreasingSizeBestFitHeap heap(/*alignment=*/1);
  heap.Alloc(buffer_a_, 10);
  heap.Free(buffer_a_, 10);
  heap.Alloc(buffer_b_, 30);

  heap.ShareWith(buffer_c_, buffer_a_, 10);
  heap.Free(buffer_c_, 10);
  heap.Free(buffer_b_, 30);

  const HeapSimulator::Result result = heap.Finish();
  EXPECT_EQ(40, result.heap_size);
  EXPECT_EQ(10, result.chunk_map.at(buffer_a_).size);
  EXPECT_EQ(30, result.chunk_map.at(buffer_b_).size);
  EXPECT_EQ(10, result.chunk_map.at(buffer_c_).size);

  EXPECT_EQ(30, result.chunk_map.at(buffer_a_).offset);
  EXPECT_EQ(0, result.chunk_map.at(buffer_b_).offset);
  EXPECT_EQ(30, result.chunk_map.at(buffer_c_).offset);
}

TEST_F(GlobalDecreasingSizeBestFitHeapTest, ChunkCandidate) {
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
  using pair = std::pair<int64, int64>;
  EXPECT_EQ(pair(5, 10), heap_.FindChunkCandidate(buffer_a_, 5, 6, 10, 5));
  heap_.CommitChunk();  // offset: 5, size: 5, start: 6, end: 10
  // Preferred offset 5 is returned.
  EXPECT_EQ(pair(0, 10), heap_.FindChunkCandidate(buffer_b_, 10, 3, 5));
  heap_.CommitChunk();  // offset: 0, size: 10, start: 3, end: 5
  EXPECT_EQ(pair(10, 15), heap_.FindChunkCandidate(buffer_c_, 5, 2, 8));
  heap_.CommitChunk();  // offset: 10, size: 5, start: 2, end: 8
  EXPECT_EQ(pair(0, 15), heap_.FindChunkCandidate(buffer_d_, 5, 0, 2, 10));
  heap_.CommitChunk();  // offset: 0, size: 5, start: 0, end: 2
  // Preferred offset 10 could not be given because it is occupied.
  EXPECT_EQ(pair(10, 20), heap_.FindChunkCandidate(buffer_e_, 10, 11, 13, 10));
  heap_.CommitChunk();  // offset: 10, size: 10, start: 11, end: 13
  // Preferred offset 10 is returned.
  EXPECT_EQ(pair(20, 25), heap_.FindChunkCandidate(buffer_f_, 5, 3, 5, 20));
  heap_.CommitChunk();  // offset: 20, size: 5, start: 3, end: 5
  // Preferred offset 20 is returned.
  EXPECT_EQ(pair(25, 35), heap_.FindChunkCandidate(buffer_g_, 10, 4, 8, 15));
  heap_.CommitChunk();  // offset: 25, size: 10, start: 4, end: 8
  // Preferred offset 15 could not be given because it is occupied.
}

class IntervalTreeTest : public ::testing::Test {};

TEST_F(IntervalTreeTest, InsertAndRemove) {
  HeapSimulator::Chunk chunk({1, 2});
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
  HeapSimulator::Chunk chunk({1, 2});  // Value in chunk doesn't matter here.
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
  HeapSimulator::Chunk chunk({1, 2});  // Value in chunk doesn't matter here.
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
  HeapSimulator::Chunk chunk({1, 2});  // Value in chunk doesn't matter here.
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
  HeapSimulator::Chunk chunk({1, 2});  // Value in chunk doesn't matter here.
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
  HeapSimulator::Chunk chunk({1, 2});  // Value in chunk doesn't matter here.
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
  HeapSimulator::Chunk chunk({1, 2});  // Value in chunk doesn't matter here.
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
  HeapSimulator::Chunk chunk({1, 2});  // Value in chunk doesn't matter here.
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
  HeapSimulator::Chunk chunk({1, 2});  // Value in chunk doesn't matter here.
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
  HeapSimulator::Chunk chunk1({1, 2});
  HeapSimulator::Chunk chunk2({2, 3});
  HeapSimulator::Chunk chunk3({3, 4});
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

}  // namespace
}  // namespace xla
