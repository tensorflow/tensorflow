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

#include "xla/hlo/analysis/tuple_points_to_analysis.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

class TuplePointsToAnalysisTest : public HloHardwareIndependentTestBase {
 protected:
  // Builds a module with the given entry computation and runs points to
  // analysis.
  void BuildModuleAndRunAnalysis(std::unique_ptr<HloComputation> computation) {
    BuildModule(std::move(computation));
    RunAnalysis();
  }

  void BuildModule(std::unique_ptr<HloComputation> computation) {
    module_ = CreateNewVerifiedModule();
    module_->AddEntryComputation(std::move(computation));
  }

  void RunAnalysis() {
    CHECK_NOTNULL(module_.get());
    points_to_analysis_ = TuplePointsToAnalysis::Run(module_.get()).value();
  }

  // Returns the LogicalBuffer defined at the given instruction and
  // index. CHECKs if no buffer is defined at that point.
  const LogicalBuffer* GetBuffer(const HloInstruction* instruction,
                                 const ShapeIndex& index) {
    const auto& pointed_to =
        points_to_analysis_->GetPointsToSet(instruction).element(index);
    CHECK_EQ(1, pointed_to.size());
    CHECK_EQ(instruction, pointed_to[0]->instruction());
    CHECK(index == pointed_to[0]->index());
    return pointed_to[0];
  }

  // Checks that the given points-to set contains exactly (unordered) the given
  // LogicalBuffers.
  void ExpectHasBuffers(const PointsToSet::BufferList& points_to_set,
                        absl::Span<const LogicalBuffer* const> buffers) {
    std::vector<const LogicalBuffer*> vec(buffers.begin(), buffers.end());
    EXPECT_THAT(points_to_set, UnorderedElementsAreArray(vec));
  }

  // Checks that the given points-to set contains exactly (unordered) the
  // top-level buffers of the given instructions.
  void ExpectHasTopLevelBuffers(
      const PointsToSet::BufferList& points_to_set,
      absl::Span<HloInstruction* const> instructions) {
    PointsToSet::BufferList buffers;
    for (auto instruction : instructions) {
      buffers.push_back(GetBuffer(instruction, /*index=*/{}));
    }
    ExpectHasBuffers(points_to_set, buffers);
  }

  // Overload which takes a set instead of a vector.
  void ExpectHasTopLevelBuffers(
      const PointsToSet::BufferSet& points_to_set,
      absl::Span<HloInstruction* const> instructions) {
    ExpectHasTopLevelBuffers(
        PointsToSet::BufferList(points_to_set.begin(), points_to_set.end()),
        instructions);
  }

  // Checks that the buffer defined at the given instruction and index has
  // aliases which are exactly (unordered) the given instruction/index pairs.
  void ExpectHasBufferAliases(
      const HloInstruction* instruction, const ShapeIndex& index,
      absl::Span<const std::pair<HloInstruction*, ShapeIndex>> expected) {
    const LogicalBuffer* buffer =
        points_to_analysis_->GetBufferDefinedAt(instruction, index).value();
    std::vector<BufferAlias> expected_aliases;
    expected_aliases.reserve(expected.size());
    for (auto& pair : expected) {
      expected_aliases.push_back(BufferAlias(pair.first, pair.second));
    }
    EXPECT_THAT(points_to_analysis_->GetBufferAliases(*buffer),
                UnorderedElementsAreArray(expected_aliases));
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;
};

TEST_F(TuplePointsToAnalysisTest, SimpleTuple) {
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));

  BuildModuleAndRunAnalysis(builder.Build());
  EXPECT_EQ(1, points_to_analysis_->GetPointsToSet(constant1).size());
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(constant1).element({}), {constant1});
  EXPECT_TRUE(
      points_to_analysis_->GetPointsToSet(constant1).tuple_sources({}).empty());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(tuple).IsDistinct());

  EXPECT_EQ(1, points_to_analysis_->GetPointsToSet(constant2).size());
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(constant2).element({}), {constant2});
  EXPECT_TRUE(
      points_to_analysis_->GetPointsToSet(constant2).tuple_sources({}).empty());

  EXPECT_EQ(3, points_to_analysis_->GetPointsToSet(tuple).size());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(tuple).IsAmbiguous());
  EXPECT_THAT(points_to_analysis_->GetPointsToSet(tuple).tuple_sources({}),
              UnorderedElementsAre(tuple));

  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).CreateFlattenedSet(),
      {constant1, constant2, tuple});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({}), {tuple});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({0}), {constant1});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({1}), {constant2});

  const PointsToSet& tuple_points_to_set =
      points_to_analysis_->GetPointsToSet(tuple);
  EXPECT_TRUE(tuple_points_to_set.ContainsBufferAtIndex(
      *GetBuffer(constant1, {}), {0}));
  EXPECT_TRUE(tuple_points_to_set.ContainsBufferAtIndex(
      *GetBuffer(constant2, {}), {1}));
  EXPECT_FALSE(tuple_points_to_set.ContainsBufferAtIndex(
      *GetBuffer(constant2, {}), {0}));
  EXPECT_TRUE(tuple_points_to_set.ContainsBuffer(*GetBuffer(constant1, {})));
  EXPECT_TRUE(tuple_points_to_set.ContainsBuffer(*GetBuffer(constant2, {})));
}

TEST_F(TuplePointsToAnalysisTest, NestedTuple) {
  // Create a (nested) tuple containing an inner tuple. The points-to set of the
  // outer tuple should contain all elements of the points-to set of the inner
  // tuple.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto inner_tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));

  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({inner_tuple, constant3}));

  BuildModuleAndRunAnalysis(builder.Build());
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(constant1).element({}), {constant1});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(constant2).element({}), {constant2});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(constant3).element({}), {constant3});

  EXPECT_EQ(3, points_to_analysis_->GetPointsToSet(inner_tuple).size());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(inner_tuple).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(inner_tuple).IsDistinct());
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(inner_tuple).CreateFlattenedSet(),
      {constant1, constant2, inner_tuple});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(inner_tuple).element({}),
      {inner_tuple});
  EXPECT_THAT(
      points_to_analysis_->GetPointsToSet(inner_tuple).tuple_sources({}),
      UnorderedElementsAre(inner_tuple));

  EXPECT_EQ(5, points_to_analysis_->GetPointsToSet(tuple).size());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(tuple).IsAmbiguous());
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).CreateFlattenedSet(),
      {constant1, constant2, constant3, inner_tuple, tuple});

  EXPECT_THAT(points_to_analysis_->GetPointsToSet(tuple).tuple_sources({}),
              UnorderedElementsAre(tuple));
  EXPECT_THAT(points_to_analysis_->GetPointsToSet(tuple).tuple_sources({0}),
              UnorderedElementsAre(inner_tuple));
  EXPECT_TRUE(
      points_to_analysis_->GetPointsToSet(tuple).tuple_sources({1}).empty());

  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({0}), {inner_tuple});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({0, 0}), {constant1});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({0, 1}), {constant2});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({1}), {constant3});
}

TEST_F(TuplePointsToAnalysisTest, GetTupleElement) {
  // Create a nested tuple, then extract the inner tuple with GetTupleElement.
  // The points-to set of the GetTupleElement should be the same as the inner
  // tuple.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto inner_tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));

  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({inner_tuple, constant3}));

  auto get_tuple_element = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(inner_tuple->shape(), tuple, 0));

  BuildModuleAndRunAnalysis(builder.Build());

  auto& points_to_set = points_to_analysis_->GetPointsToSet(get_tuple_element);
  EXPECT_EQ(3, points_to_set.size());
  EXPECT_FALSE(points_to_set.IsAmbiguous());
  EXPECT_TRUE(points_to_set.IsDistinct());
  ExpectHasTopLevelBuffers(points_to_set.CreateFlattenedSet(),
                           {constant1, constant2, inner_tuple});
  ExpectHasTopLevelBuffers(points_to_set.element({}), {inner_tuple});

  EXPECT_THAT(points_to_set.tuple_sources({}),
              UnorderedElementsAre(inner_tuple));
}

TEST_F(TuplePointsToAnalysisTest, AddDependency) {
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto add_dependency = builder.AddInstruction(
      HloInstruction::CreateAddDependency(constant, token));
  BuildModuleAndRunAnalysis(builder.Build());

  auto& points_to_set = points_to_analysis_->GetPointsToSet(add_dependency);
  EXPECT_EQ(1, points_to_set.size());
  EXPECT_FALSE(points_to_set.IsAmbiguous());
  EXPECT_TRUE(points_to_set.IsDistinct());
  ExpectHasTopLevelBuffers(points_to_set.CreateFlattenedSet(), {constant});
}

TEST_F(TuplePointsToAnalysisTest, DuplicatedElement) {
  // Create a tuple which contains duplicate elements.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant, constant, constant}));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_EQ(2, points_to_analysis_->GetPointsToSet(tuple).size());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(tuple).IsAmbiguous());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(tuple).IsDistinct());
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({}), {tuple});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).CreateFlattenedSet(),
      {constant, tuple});
}

TEST_F(TuplePointsToAnalysisTest, TupleCopy) {
  // Create a copy (HloOpcode::kCopy) of a tuple. The points to sets should be
  // the same.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(tuple->shape(), HloOpcode::kCopy, tuple));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(copy).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(copy).IsDistinct());
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).CreateFlattenedSet(),
      {constant1, constant2, tuple});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(copy).element({}), {copy});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(copy).CreateFlattenedSet(),
      {constant1, constant2, copy});
}

TEST_F(TuplePointsToAnalysisTest, CopyStartAndCopyDone) {
  // CopyDone forwards its operand to the output tuple at {0}.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto copy_start = builder.AddInstruction(HloInstruction::CreateCopyStart(
      ShapeUtil::MakeTupleShape({constant->shape(), constant->shape(),
                                 ShapeUtil::MakeShape(U32, {})}),
      constant));
  auto copy_done = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopyDone, copy_start));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(copy_start).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(copy_start).IsDistinct());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(copy_done).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(copy_done).IsDistinct());

  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(copy_start).element({}),
      {copy_start});
  ExpectHasBufferAliases(copy_start, {0}, {{copy_start, {0}}, {copy_done, {}}});
  ExpectHasBufferAliases(constant, {}, {{constant, {}}, {copy_start, {1}}});
}

TEST_F(TuplePointsToAnalysisTest, AsyncOps) {
  std::string hlo_str = R"(
  HloModule module

  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[2,3], u32[]) custom-call-start(p0), custom_call_target="foo"
    async-update = ((f32[2,3]), f32[2,3], u32[]) custom-call-update(async-start)
    ROOT async-done = f32[2,3] custom-call-done(async-update)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(hlo_str, GetModuleConfigForTest()));

  HloInstruction* param =
      module_->entry_computation()->parameter_instruction(0);
  HloInstruction* async_start = FindInstruction(module_.get(), "async-start");
  HloInstruction* async_update = FindInstruction(module_.get(), "async-update");
  HloInstruction* async_done = FindInstruction(module_.get(), "async-done");

  RunAnalysis();
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(async_start).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(async_start).IsDistinct());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(async_update).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(async_update).IsDistinct());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(async_done).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(async_done).IsDistinct());

  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(async_start).element({}),
      {async_start});
  ExpectHasBufferAliases(
      param, {}, {{param, {}}, {async_start, {0, 0}}, {async_update, {0, 0}}});
  ExpectHasBufferAliases(
      async_start, {1},
      {{async_start, {1}}, {async_update, {1}}, {async_done, {}}});
  ExpectHasBufferAliases(async_start, {2},
                         {{async_start, {2}}, {async_update, {2}}});
}

TEST_F(TuplePointsToAnalysisTest, SendAndSendDone) {
  // Send forwards its operand to the output tuple at {0}.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(HloInstruction::CreateSend(
      constant, token, /*channel_id=*/0, /*is_host_transfer=*/false));
  auto send_done = builder.AddInstruction(HloInstruction::CreateSendDone(
      send, send->channel_id(), /*is_host_transfer=*/false));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(send).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(send).IsDistinct());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(send_done).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(send_done).IsDistinct());

  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(send).element({}), {send});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(send).element({0}), {constant});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(send_done).CreateFlattenedSet(),
      {send_done});
  ExpectHasBufferAliases(constant, {}, {{constant, {}}, {send, {0}}});
}

TEST_F(TuplePointsToAnalysisTest, RecvAndRecvDone) {
  // RecvDone forwards its operand tuple element at {0} to the output.
  auto builder = HloComputation::Builder(TestName());
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto recv = builder.AddInstruction(
      HloInstruction::CreateRecv(ShapeUtil::MakeShape(F32, {1, 2, 3}), token,
                                 /*channel_id=*/0, /*is_host_transfer=*/false));
  auto recv_done = builder.AddInstruction(HloInstruction::CreateRecvDone(
      recv, recv->channel_id(), /*is_host_transfer=*/false));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(recv).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(recv).IsDistinct());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(recv_done).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(recv_done).IsDistinct());

  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(recv).element({}), {recv});
  ExpectHasBufferAliases(recv, {0}, {{recv, {0}}, {recv_done, {0}}});
}

TEST_F(TuplePointsToAnalysisTest, TupleWithBitcast) {
  // Bitcast is an alias of its operand. A tuple with a bitcast element should
  // have the operand of the bitcast in its points-to set.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(constant2->shape(), constant2));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant1, bitcast}));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_EQ(1, points_to_analysis_->GetPointsToSet(bitcast).size());
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(bitcast).element({}), {constant2});
  EXPECT_TRUE(
      points_to_analysis_->GetPointsToSet(bitcast).tuple_sources({}).empty());

  EXPECT_EQ(3, points_to_analysis_->GetPointsToSet(tuple).size());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(tuple).IsAmbiguous());
  EXPECT_THAT(points_to_analysis_->GetPointsToSet(tuple).tuple_sources({}),
              UnorderedElementsAre(tuple));

  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).CreateFlattenedSet(),
      {constant1, constant2, tuple});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({}), {tuple});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({0}), {constant1});
  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(tuple).element({1}), {constant2});
}

TEST_F(TuplePointsToAnalysisTest, PointsToTupleConstantElements) {
  // Construct a tuple constant and kCopy it. Verify the points-to set of the
  // copy correctly points into the nested elements of the constant.
  auto builder = HloComputation::Builder(TestName());
  Literal elements[] = {LiteralUtil::CreateR2<float>({{1.0}, {2.0}}),
                        LiteralUtil::CreateR1<float>({2.0, 42})};
  auto tuple_constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::MakeTuple({&elements[0], &elements[1]})));
  auto copy = builder.AddInstruction(HloInstruction::CreateUnary(
      tuple_constant->shape(), HloOpcode::kCopy, tuple_constant));

  BuildModuleAndRunAnalysis(builder.Build());

  auto& points_to_set = points_to_analysis_->GetPointsToSet(copy);

  ExpectHasBuffers(points_to_set.element({}), {GetBuffer(copy, {})});
  ExpectHasBuffers(points_to_set.element({0}),
                   {GetBuffer(tuple_constant, {0})});
  ExpectHasBuffers(points_to_set.element({1}),
                   {GetBuffer(tuple_constant, {1})});
}

TEST_F(TuplePointsToAnalysisTest, BufferAliases) {
  // Create a nested tuple in which individual elements appear multiple
  // times. Verify buffer alias sets.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto inner_tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({inner_tuple, constant2}));

  BuildModuleAndRunAnalysis(builder.Build());

  ExpectHasBufferAliases(
      constant1, /*index=*/{},
      {{constant1, {}}, {inner_tuple, {0}}, {tuple, {0, 0}}});
  ExpectHasBufferAliases(
      constant2, /*index=*/{},
      {{constant2, {}}, {inner_tuple, {1}}, {tuple, {0, 1}}, {tuple, {1}}});
  ExpectHasBufferAliases(inner_tuple, /*index=*/{},
                         {{inner_tuple, {}}, {tuple, {0}}});
  ExpectHasBufferAliases(tuple, /*index=*/{}, {{tuple, {}}});
}

// TODO(b/277597692): The fix for this bug causes this test to fail.
TEST_F(TuplePointsToAnalysisTest, DISABLED_CustomCall) {
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  Shape data_shape = ShapeUtil::MakeShape(F32, {});
  auto ccall = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeTupleShape({data_shape, data_shape}), {constant},
      "TestOp"));
  Cast<HloCustomCallInstruction>(ccall)->set_output_to_operand_aliasing(
      {std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>{
          ShapeIndex{1}, std::pair<int64_t, ShapeIndex>(0, {})}});
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, ccall, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, ccall, 1));

  BuildModuleAndRunAnalysis(builder.Build());

  ExpectHasBufferAliases(ccall, /*index=*/{0}, {{gte0, {}}, {ccall, {0}}});
  ExpectHasBufferAliases(constant, /*index=*/{},
                         {{constant, {}}, {gte1, {}}, {ccall, {1}}});
}

class FusionPointsToAnalysisTest : public TuplePointsToAnalysisTest {
 protected:
  // Takes a HLO string, parses it, runs points-to analysis, then checks for
  // expected results (see unit test cases for example computation graphs).
  void Run(const std::string& hlo_str, int64_t expected_num_users) {
    TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo_str));
    // Get computation root instruction (should be a kFusion).
    auto* fusion = module_->entry_computation()->root_instruction();
    auto* tuple_param0 = fusion->operand(0);
    // Run points-to analysis (should include fused instructions from 'fusion').
    RunAnalysis();

    // Check points-to set of fusion parameter associated with 'tuple_param0'.
    auto* fusion_param = GetFusionParameterForOperand(fusion, tuple_param0);
    ExpectHasBuffers(
        points_to_analysis_->GetPointsToSet(fusion_param).element({}),
        {GetBuffer(fusion_param, {})});
    ExpectHasBuffers(
        points_to_analysis_->GetPointsToSet(fusion_param).element({0}),
        {GetBuffer(fusion_param, {0})});
    ExpectHasBuffers(
        points_to_analysis_->GetPointsToSet(fusion_param).element({1}),
        {GetBuffer(fusion_param, {1})});

    // Check that Gte at tuple_index = 0 points-to fusion_param({0})
    auto fused_gte0 = GetUniqueFusionParameterUserAt(fusion_param, 0);
    ExpectHasBuffers(
        points_to_analysis_->GetPointsToSet(fused_gte0).element({}),
        {GetBuffer(fusion_param, {0})});
    // Check that Gte at tuple_index = 1 points-to fusion_param({1})
    auto fused_gte1 = GetUniqueFusionParameterUserAt(fusion_param, 1);
    ExpectHasBuffers(
        points_to_analysis_->GetPointsToSet(fused_gte1).element({}),
        {GetBuffer(fusion_param, {1})});

    // Check buffer aliases of 'fusion_param' at shape index {0}.
    ExpectHasBufferAliases(fusion_param, /*index=*/{0},
                           {{fusion_param, {0}}, {fused_gte0, {}}});
    // Check buffer aliases of 'fusion_param' at shape index {1}.
    ExpectHasBufferAliases(fusion_param, /*index=*/{1},
                           {{fusion_param, {1}}, {fused_gte1, {}}});

    // Check number of users of 'fusion_param' aliases at shape index {0}.
    ExpectNumUsersOfAliases(fusion_param, {0}, expected_num_users);
  }

  // Returns fusion parameter (from 'fusion.fused_instructions') corresponding
  // to fusion 'operand'.
  HloInstruction* GetFusionParameterForOperand(HloInstruction* fusion,
                                               const HloInstruction* operand) {
    const auto& fused_instructions = fusion->fused_instructions();
    auto it =
        absl::c_find_if(fused_instructions, [&](const HloInstruction* fused) {
          return fused->opcode() == HloOpcode::kParameter &&
                 fusion->operand(fused->parameter_number()) == operand;
        });
    CHECK(it != fusion->fused_instructions().end());
    return *it;
  }

  // Returns all users of 'fusion_paran' at 'tuple_index'.
  std::vector<HloInstruction*> GetFusionParameterUsersAt(
      HloInstruction* fusion_param, int64_t tuple_index) {
    CHECK(fusion_param->shape().IsTuple());
    std::vector<HloInstruction*> users_at_tuple_index;
    for (auto user : fusion_param->users()) {
      CHECK_EQ(HloOpcode::kGetTupleElement, user->opcode());
      if (user->tuple_index() == tuple_index) {
        users_at_tuple_index.push_back(user);
      }
    }
    return users_at_tuple_index;
  }

  // Returns the unique user of 'fusion_param' at 'tuple_index'.
  HloInstruction* GetUniqueFusionParameterUserAt(HloInstruction* fusion_param,
                                                 int64_t tuple_index) {
    std::vector<HloInstruction*> users =
        GetFusionParameterUsersAt(fusion_param, tuple_index);
    CHECK_EQ(1, users.size());
    return users[0];
  }

  // Checks that the count of all users of all aliases of 'instruction' at
  // 'index' match 'expected_num_users'.
  void ExpectNumUsersOfAliases(const HloInstruction* instruction,
                               const ShapeIndex& index,
                               const int64_t expected_num_users) {
    const auto* buffer = GetBuffer(instruction, index);
    int64_t num_users = 0;
    for (const auto& alias : points_to_analysis_->GetBufferAliases(*buffer)) {
      for (auto user : alias.instruction()->users()) {
        if (user->opcode() == HloOpcode::kGetTupleElement && !index.empty()) {
          // Gte instructions only access the top-level buffer of their operand.
          continue;
        }
        ++num_users;
      }
    }
    EXPECT_EQ(expected_num_users, num_users);
  }
};

// Tests the points-to set of tuple-shaped fusion parameter 0 and all GTE users.
// Tests the alias set of tuple-shaped fusion parameter 0 at all shape indices.
// Tests that there is a single user of the aliases of tuple-shaped fusion
// parameter 0 at shape index {0}.
//
//             Param0    Const
//                 \      /
//                  Fusion
//                 /      \
//        FusionParam0   FusionParam1
//        /          \       |
//     Gte(0) Const  Gte(1)  /
//        \     |      \    /
//         \    |       Add
//          \   |        /
//           \0 |2      /1
//          DynamicUpdateSlice  // fused root.
//
TEST_F(FusionPointsToAnalysisTest, FusionParam0OneUser) {
  std::string hlo_str = R"(
HloModule FusionParam0OneUser

%fused_computation (param_1.2: (f32[8], f32[3])) -> f32[8] {
  %param_1.2 = (f32[8]{0}, f32[3]{0}) parameter(0)
  %get-tuple-element.1 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[3]{0}) %param_1.2), index=0
  %get-tuple-element.2 = f32[3]{0} get-tuple-element((f32[8]{0}, f32[3]{0}) %param_1.2), index=1
  %constant.3 = f32[3]{0} constant({1, 1, 1})
  %add.1 = f32[3]{0} add(f32[3]{0} %get-tuple-element.2, f32[3]{0} %constant.3)
  %constant.2 = s32[] constant(0)
  ROOT %dynamic-update-slice.1 = f32[8]{0} dynamic-update-slice(f32[8]{0} %get-tuple-element.1, f32[3]{0} %add.1, s32[] %constant.2)
}

ENTRY %FusionParam0OneUser (param0: (f32[8], f32[3])) -> f32[8] {
  %param0 = (f32[8]{0}, f32[3]{0}) parameter(0)
  ROOT %fusion = f32[8]{0} fusion((f32[8]{0}, f32[3]{0}) %param0), kind=kLoop, calls=%fused_computation
}
)";
  Run(hlo_str, /*expected_num_users=*/1);
}

TEST_F(FusionPointsToAnalysisTest,
       FusionParam0OneUserWithUnreachableInstructionInFusion) {
  std::string hlo_str = R"(
HloModule FusionParam0OneUser

%fused_computation (param_1.2: (f32[8], f32[3], f32[7])) -> f32[8] {
  %param_1.2 = (f32[8]{0}, f32[3]{0}, f32[7]{0}) parameter(0)
  %get-tuple-element.1 = f32[8]{0} get-tuple-element(%param_1.2), index=0
  %get-tuple-element.2 = f32[3]{0} get-tuple-element(%param_1.2), index=1
  %get-tuple-element.3 = f32[7]{0} get-tuple-element(%param_1.2), index=2
  %placeholder = f32[7]{0} custom-call(%get-tuple-element.3), custom_call_target="IntermediateBufferDummyConsumer", custom_call_has_side_effect=true
  %constant.3 = f32[3]{0} constant({1, 1, 1})
  %add.1 = f32[3]{0} add(f32[3]{0} %get-tuple-element.2, f32[3]{0} %constant.3)
  %constant.2 = s32[] constant(0)
  ROOT %dynamic-update-slice.1 = f32[8]{0} dynamic-update-slice(f32[8]{0} %get-tuple-element.1, f32[3]{0} %add.1, s32[] %constant.2)
}

ENTRY %FusionParam0OneUser (param0: (f32[8], f32[3], f32[7])) -> f32[8] {
  %param0 = (f32[8]{0}, f32[3]{0}, f32[7]{0}) parameter(0)
  ROOT %fusion = f32[8]{0} fusion(%param0), kind=kLoop, calls=%fused_computation
}
)";
  Run(hlo_str, /*expected_num_users=*/1);
}

// Tests the points-to set of tuple-shaped fusion parameter 0 and all GTE users.
// Tests the alias set of tuple-shaped fusion parameter 0 at all shape indices.
// Tests that there are two users of the aliases of tuple-shaped fusion
// parameter 0 at shape index {0}.
//
//             Param0    Const
//                 \      /
//                  Fusion
//                 /      \
//        FusionParam0   FusionParam1
//              |    \       |
//            Gte(0) Gte(1)  /
//              |      \    /
//              |\      Add
//        Const | \      /
//           |  | Slice /
//           |  |   \  /
//           |  |   Add
//           |  |    |
//           |2 |0   |1
//          DynamicUpdateSlice  // fused root.
//
TEST_F(FusionPointsToAnalysisTest, FusionParam0TwoUsers) {
  std::string hlo_str = R"(
HloModule FusionParam0TwoUsers

%fused_computation (param_1.2: (f32[8], f32[3])) -> f32[8] {
  %param_1.2 = (f32[8]{0}, f32[3]{0}) parameter(0)
  %get-tuple-element.1 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[3]{0}) %param_1.2), index=0
  %get-tuple-element.2 = f32[3]{0} get-tuple-element((f32[8]{0}, f32[3]{0}) %param_1.2), index=1
  %constant.3 = f32[3]{0} constant({1, 1, 1})
  %add.1 = f32[3]{0} add(f32[3]{0} %get-tuple-element.2, f32[3]{0} %constant.3)
  %slice = f32[3]{0} slice(f32[8]{0} %get-tuple-element.1), slice={[0:3]}
  %add.2 = f32[3]{0} add(f32[3]{0} %add.1, f32[3]{0} %slice)
  %constant.2 = s32[] constant(0)
  ROOT %dynamic-update-slice.1 = f32[8]{0} dynamic-update-slice(f32[8]{0} %get-tuple-element.1, f32[3]{0} %add.2, s32[] %constant.2)
}

ENTRY %FusionParam0TwoUsers (param0: (f32[8], f32[3])) -> f32[8] {
  %param0 = (f32[8]{0}, f32[3]{0}) parameter(0)
  ROOT %fusion = f32[8]{0} fusion((f32[8]{0}, f32[3]{0}) %param0), kind=kLoop, calls=%fused_computation
}
)";
  Run(hlo_str, /*expected_num_users=*/2);
}

class PointsToAnalysisTestBase : public HloHardwareIndependentTestBase {
 protected:
  void BuildModule(std::unique_ptr<HloComputation> computation) {
    module_ = CreateNewVerifiedModule();
    computation_ = module_->AddEntryComputation(std::move(computation));
  }

  void RunAnalysis() {
    CHECK_NOTNULL(module_.get());
    points_to_analysis_ = TuplePointsToAnalysis::Run(module_.get()).value();
  }

  void BuildModuleAndRunAnalysis(std::unique_ptr<HloComputation> computation) {
    BuildModule(std::move(computation));
    RunAnalysis();
  }

  std::unique_ptr<HloModule> module_;
  HloComputation* computation_ = nullptr;
  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;
};

class DoesNotUseOperandBufferTest : public PointsToAnalysisTestBase {};

TEST_F(DoesNotUseOperandBufferTest, GetTupleElement) {
  auto builder = HloComputation::Builder(TestName());

  Shape elem_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({elem_shape, elem_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(elem_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(elem_shape, tuple, 1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(elem_shape, HloOpcode::kAdd, gte0, gte1));

  BuildModuleAndRunAnalysis(builder.Build());

  // GetTupleElement instructions only access the top-level buffer of their
  // operand.
  EXPECT_TRUE(points_to_analysis_->DoesNotUseOperandBuffer(tuple, {0}, gte0));
  EXPECT_TRUE(points_to_analysis_->DoesNotUseOperandBuffer(tuple, {1}, gte1));
  EXPECT_FALSE(points_to_analysis_->DoesNotUseOperandBuffer(tuple, {}, gte0));
  EXPECT_FALSE(points_to_analysis_->DoesNotUseOperandBuffer(tuple, {}, gte1));
}

TEST_F(DoesNotUseOperandBufferTest, FusedDynamicUpdateSlice) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  auto tuple = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "tuple"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, tuple, 1));

  // Create a DynamicUpdateSlice instruction of tuple element 1.
  auto starts = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(2)));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape, gte1, update, {starts}));
  builder.AddInstruction(
      HloInstruction::CreateTuple({gte0, dynamic_update_slice}));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {dynamic_update_slice, starts, update, gte1},
      HloInstruction::FusionKind::kLoop);
  RunAnalysis();

  // The fusion instruction never uses tuple element 0, but does use element 1.
  EXPECT_TRUE(points_to_analysis_->DoesNotUseOperandBuffer(tuple, {0}, fusion));
  EXPECT_FALSE(
      points_to_analysis_->DoesNotUseOperandBuffer(tuple, {1}, fusion));
}
}  // namespace
}  // namespace xla
