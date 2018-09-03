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

#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"

#include <map>
#include <memory>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

class TuplePointsToAnalysisTest : public HloTestBase {
 protected:
  // Builds a module with the given entry computation and runs points to
  // analysis.
  void BuildModuleAndRunAnalysis(std::unique_ptr<HloComputation> computation) {
    BuildModule(std::move(computation));
    RunAnalysis();
  }

  void BuildModule(std::unique_ptr<HloComputation> computation) {
    module_ = CreateNewModule();
    module_->AddEntryComputation(std::move(computation));
  }

  void RunAnalysis() {
    CHECK_NOTNULL(module_.get());
    points_to_analysis_ =
        TuplePointsToAnalysis::Run(module_.get()).ConsumeValueOrDie();
  }

  // Returns the LogicalBuffer defined at the given instruction and
  // index. CHECKs if no buffer is defined at that point.
  const LogicalBuffer* const GetBuffer(const HloInstruction* instruction,
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
        points_to_analysis_->GetBufferDefinedAt(instruction, index)
            .ValueOrDie();
    std::vector<BufferAlias> expected_aliases;
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

TEST_F(TuplePointsToAnalysisTest, SendAndSendDone) {
  // Send forwards its operand to the output tuple at {0}.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(
      HloInstruction::CreateSend(constant, token, /*channel_id=*/0));
  auto send_done = builder.AddInstruction(HloInstruction::CreateSendDone(send));

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
  auto recv = builder.AddInstruction(HloInstruction::CreateRecv(
      ShapeUtil::MakeShape(F32, {1, 2, 3}), token, /*channel_id=*/0));
  auto recv_done = builder.AddInstruction(HloInstruction::CreateRecvDone(recv));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(recv).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(recv).IsDistinct());
  EXPECT_FALSE(points_to_analysis_->GetPointsToSet(recv_done).IsAmbiguous());
  EXPECT_TRUE(points_to_analysis_->GetPointsToSet(recv_done).IsDistinct());

  ExpectHasTopLevelBuffers(
      points_to_analysis_->GetPointsToSet(recv).element({}), {recv});
  ExpectHasBufferAliases(recv, {0}, {{recv, {0}}, {recv_done, {0}}});
}

TEST_F(TuplePointsToAnalysisTest, TupleSelect) {
  // Select from two different tuples. This should create an ambiguous points to
  // set containing the union of both sides.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto tuple2 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant2, constant2}));

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple1->shape(), HloOpcode::kTupleSelect, pred, tuple1, tuple2));

  BuildModuleAndRunAnalysis(builder.Build());

  auto& points_to_set = points_to_analysis_->GetPointsToSet(select);
  EXPECT_EQ(3, points_to_set.size());
  EXPECT_TRUE(points_to_set.IsAmbiguous());
  EXPECT_FALSE(points_to_set.IsDistinct());
  ExpectHasTopLevelBuffers(points_to_set.element({}), {select});
  ExpectHasTopLevelBuffers(points_to_set.element({0}), {constant1, constant2});
  ExpectHasTopLevelBuffers(points_to_set.element({1}), {constant2});
  ExpectHasTopLevelBuffers(points_to_set.CreateFlattenedSet(),
                           {constant1, constant2, select});
}

TEST_F(TuplePointsToAnalysisTest, SelectTupleParameters) {
  // Create a Select which selects between two tuple parameters. Verify the
  // points-to sets and tuple sources are properly set.
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {1, 2, 3}), ShapeUtil::MakeShape(U32, {5})});

  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, tuple_shape, "param1"));
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kTupleSelect, pred, param0, param1));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(tuple_shape, HloOpcode::kCopy, select));

  BuildModuleAndRunAnalysis(builder.Build());

  // The points-to set of each element of a tuple parameters should be itself
  // with the appropriate index.
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(param0).element({}),
                   {GetBuffer(param0, {})});
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(param0).element({0}),
                   {GetBuffer(param0, {0})});
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(param0).element({1}),
                   {GetBuffer(param0, {1})});

  // Select's point-to set of its subelements should be the respective
  // subelements of param0 and param1. The top-level buffer, however, does not
  // alias as it is created by the select instruction.
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(select).element({}),
                   {GetBuffer(select, {})});
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(select).element({0}),
                   {GetBuffer(param0, {0}), GetBuffer(param1, {0})});
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(select).element({1}),
                   {GetBuffer(param0, {1}), GetBuffer(param1, {1})});

  // Copy should be identical to select other than the top-level buffer.
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(copy).element({}),
                   {GetBuffer(copy, {})});
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(copy).element({0}),
                   {GetBuffer(param0, {0}), GetBuffer(param1, {0})});
  ExpectHasBuffers(points_to_analysis_->GetPointsToSet(copy).element({1}),
                   {GetBuffer(param0, {1}), GetBuffer(param1, {1})});
}

TEST_F(TuplePointsToAnalysisTest, UnambiguousTupleSelect) {
  // Select from two identical tuples. The result should not be ambiguous.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto tuple2 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple1->shape(), HloOpcode::kTupleSelect, pred, tuple1, tuple2));

  BuildModuleAndRunAnalysis(builder.Build());

  auto& points_to_set = points_to_analysis_->GetPointsToSet(select);
  EXPECT_EQ(3, points_to_set.size());
  EXPECT_FALSE(points_to_set.IsAmbiguous());
  EXPECT_TRUE(points_to_set.IsDistinct());
  ExpectHasTopLevelBuffers(points_to_set.element({}), {select});
  ExpectHasTopLevelBuffers(points_to_set.element({0}), {constant1});
  ExpectHasTopLevelBuffers(points_to_set.element({1}), {constant2});
  ExpectHasTopLevelBuffers(points_to_set.CreateFlattenedSet(),
                           {constant1, constant2, select});
}

TEST_F(TuplePointsToAnalysisTest, NestedTupleSelect) {
  // Select from nested tuples. Verify that the nested points-to sets contain
  // the right values.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto inner_tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto inner_tuple2 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant2, constant2}));

  auto tuple1 =
      builder.AddInstruction(HloInstruction::CreateTuple({inner_tuple1}));
  auto tuple2 =
      builder.AddInstruction(HloInstruction::CreateTuple({inner_tuple2}));

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple1->shape(), HloOpcode::kTupleSelect, pred, tuple1, tuple2));

  BuildModuleAndRunAnalysis(builder.Build());

  auto& points_to_set = points_to_analysis_->GetPointsToSet(select);
  EXPECT_EQ(5, points_to_set.size());
  EXPECT_TRUE(points_to_set.IsAmbiguous());
  EXPECT_FALSE(points_to_set.IsDistinct());

  // Verify points-to set.
  ExpectHasTopLevelBuffers(points_to_set.element({}), {select});
  ExpectHasTopLevelBuffers(points_to_set.element({0}),
                           {inner_tuple1, inner_tuple2});
  ExpectHasTopLevelBuffers(points_to_set.element({0, 0}),
                           {constant1, constant2});
  ExpectHasTopLevelBuffers(points_to_set.element({0, 1}), {constant2});

  // Verify tuple sources.
  EXPECT_THAT(points_to_set.tuple_sources({}),
              UnorderedElementsAre(tuple1, tuple2));
  EXPECT_THAT(points_to_set.tuple_sources({0}),
              UnorderedElementsAre(inner_tuple1, inner_tuple2));
  EXPECT_EQ(0, points_to_set.tuple_sources({0, 0}).size());
  EXPECT_EQ(0, points_to_set.tuple_sources({0, 1}).size());
}

TEST_F(TuplePointsToAnalysisTest, TupleWithBitcast) {
  // Bitcast is an alias of its operand. A tuple with a bitcast element should
  // have the operand of the bitcast in its points-to set.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto bitcast = builder.AddInstruction(HloInstruction::CreateUnary(
      constant2->shape(), HloOpcode::kBitcast, constant2));
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
  // copy correctly correctly points into the nested elements of the constant.
  auto builder = HloComputation::Builder(TestName());
  auto tuple_constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MakeTuple(
          {LiteralUtil::CreateR2<float>({{1.0}, {2.0}}).get(),
           LiteralUtil::CreateR1<float>({2.0, 42}).get()})));
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

class FusionPointsToAnalysisTest : public TuplePointsToAnalysisTest {
 protected:
  // Builds a computation, runs instruction fusion HloPass, runs points-to
  // analysis, then checks for expected results (see unit test cases for
  // example computation graphs).
  void Run(const bool add_additional_gte0_user) {
    Shape input_shape = ShapeUtil::MakeShape(F32, {8});
    Shape update_shape = ShapeUtil::MakeShape(F32, {3});
    Shape starts_shape = ShapeUtil::MakeShape(S32, {1});
    Shape tuple_shape =
        ShapeUtil::MakeTupleShape({input_shape, update_shape, starts_shape});

    auto builder = HloComputation::Builder(TestName());
    // Create tuple-shaped parameter.
    auto tuple_param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "param0"));
    // Create 'tuple_element1' = GetTupleElement(tuple_param0, 1).
    auto tuple_element1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(update_shape, tuple_param0, 1));
    auto ones = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f, 1.f})));
    // Create 'update' = Add(GetTupleElement(tuple_param0, 1), ones)
    auto update = builder.AddInstruction(HloInstruction::CreateBinary(
        update_shape, HloOpcode::kAdd, tuple_element1, ones));
    // Create 'input' = GetTupleElement(tuple_param0, 0).
    auto input = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape, tuple_param0, 0));

    if (add_additional_gte0_user) {
      // Create 'slice' as an additional user of 'input'.
      auto slice = builder.AddInstruction(
          HloInstruction::CreateSlice(update_shape, input, {0}, {3}, {1}));
      // Modify 'update' to take 'slice' output.
      update = builder.AddInstruction(HloInstruction::CreateBinary(
          update_shape, HloOpcode::kAdd, update, slice));
    }

    // Create slice 'starts' = GetTupleElement(tuple_param0, 2).
    auto starts = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(starts_shape, tuple_param0, 2));
    // Update 'input' with 'update' at dynamic 'starts' indices.
    builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
        input_shape, input, update, starts));

    // Build computation and add it to module as entry computation.
    BuildModule(builder.Build());
    // Run instruction fusion HloPass.
    EXPECT_TRUE(InstructionFusion(InstructionFusion::IsExpensive)
                    .Run(module_.get())
                    .ValueOrDie());
    // Get computation root instruction (should be a kFusion).
    auto* fusion = module_->entry_computation()->root_instruction();
    EXPECT_THAT(fusion, op::Fusion(tuple_param0));
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
    ExpectHasBuffers(
        points_to_analysis_->GetPointsToSet(fusion_param).element({2}),
        {GetBuffer(fusion_param, {2})});

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
    // Check that Gte at tuple_index = 2 points-to fusion_param({2})
    auto fused_gte2 = GetUniqueFusionParameterUserAt(fusion_param, 2);
    ExpectHasBuffers(
        points_to_analysis_->GetPointsToSet(fused_gte2).element({}),
        {GetBuffer(fusion_param, {2})});

    // Check buffer aliases of 'fusion_param' at shape index {0}.
    ExpectHasBufferAliases(fusion_param, /*index=*/{0},
                           {{fusion_param, {0}}, {fused_gte0, {}}});
    // Check buffer aliases of 'fusion_param' at shape index {1}.
    ExpectHasBufferAliases(fusion_param, /*index=*/{1},
                           {{fusion_param, {1}}, {fused_gte1, {}}});
    // Check buffer aliases of 'fusion_param' at shape index {2}.
    ExpectHasBufferAliases(fusion_param, /*index=*/{2},
                           {{fusion_param, {2}}, {fused_gte2, {}}});

    // Check number of users of 'fusion_param' aliases at shape index {0}.
    ExpectNumUsersOfAliases(fusion_param, {0},
                            add_additional_gte0_user ? 2 : 1);
  }

  // Returns fusion parameter (from 'fusion.fused_instructions') corresponding
  // to fusion 'operand'.
  HloInstruction* GetFusionParameterForOperand(HloInstruction* fusion,
                                               HloInstruction* operand) {
    auto it = std::find_if(
        fusion->fused_instructions().begin(),
        fusion->fused_instructions().end(), [=](const HloInstruction* fused) {
          return fused->opcode() == HloOpcode::kParameter &&
                 fusion->operand(fused->parameter_number()) == operand;
        });
    CHECK(it != fusion->fused_instructions().end());
    return *it;
  }

  // Returns all users of 'fusion_paran' at 'tuple_index'.
  std::vector<HloInstruction*> GetFusionParameterUsersAt(
      HloInstruction* fusion_param, int64 tuple_index) {
    CHECK(ShapeUtil::IsTuple(fusion_param->shape()));
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
                                                 int64 tuple_index) {
    std::vector<HloInstruction*> users =
        GetFusionParameterUsersAt(fusion_param, tuple_index);
    CHECK_EQ(1, users.size());
    return users[0];
  }

  // Checks that the count of all users of all aliases of 'instruction' at
  // 'index' match 'expected_num_users'.
  void ExpectNumUsersOfAliases(const HloInstruction* instruction,
                               const ShapeIndex& index,
                               const int64 expected_num_users) {
    const auto* buffer = GetBuffer(instruction, index);
    int64 num_users = 0;
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
//        /     |    \       |
//     Gte(0) Gte(2) Gte(1)  /
//        \     |      \    /
//         \    |       Add
//          \   |        /
//           \0 |2      /1
//          DynamicUpdateSlice  // fused root.
//
TEST_F(FusionPointsToAnalysisTest, FusionParam0OneUser) {
  Run(/*add_additional_gte0_user=*/false);
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
//        /     |    \       |
//     Gte(2) Gte(0) Gte(1)  /
//        \     |      \    /
//         \    |\      Add
//          \   | \      /
//           |  | Slice /
//           |  |   \  /
//           |  |   Add
//           |  |    |
//           |2 |0   |1
//          DynamicUpdateSlice  // fused root.
//
TEST_F(FusionPointsToAnalysisTest, FusionParam0TwoUsers) {
  Run(/*add_additional_gte0_user=*/true);
}

class PointsToAnalysisTestBase : public HloTestBase {
 protected:
  void BuildModule(std::unique_ptr<HloComputation> computation) {
    module_ = CreateNewModule();
    computation_ = module_->AddEntryComputation(std::move(computation));
  }

  void RunAnalysis() {
    CHECK_NOTNULL(module_.get());
    points_to_analysis_ =
        TuplePointsToAnalysis::Run(module_.get()).ConsumeValueOrDie();
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32>({2})));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape, gte1, update, starts));
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

class CanShareOperandBufferWithUserTest : public PointsToAnalysisTestBase {};

TEST_F(CanShareOperandBufferWithUserTest, ElementWiseSameShape) {
  auto builder = HloComputation::Builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {8});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kLog, exp));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_TRUE(
      points_to_analysis_->CanShareOperandBufferWithUser(param, {}, exp, {}));
  EXPECT_TRUE(
      points_to_analysis_->CanShareOperandBufferWithUser(exp, {}, log, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, ElementWiseDifferentShape) {
  auto builder = HloComputation::Builder(TestName());

  Shape in_shape = ShapeUtil::MakeShape(F32, {8});
  Shape out_shape = ShapeUtil::MakeShape(PRED, {8});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, in_shape, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, in_shape, "param1"));
  auto result = builder.AddInstruction(
      HloInstruction::CreateBinary(out_shape, HloOpcode::kEq, param0, param1));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_FALSE(points_to_analysis_->CanShareOperandBufferWithUser(param0, {},
                                                                  result, {}));
  EXPECT_FALSE(points_to_analysis_->CanShareOperandBufferWithUser(param1, {},
                                                                  result, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, CopyShares) {
  auto builder = HloComputation::Builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {8});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, exp));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_TRUE(
      points_to_analysis_->CanShareOperandBufferWithUser(param, {}, exp, {}));
  EXPECT_TRUE(
      points_to_analysis_->CanShareOperandBufferWithUser(exp, {}, copy, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, FusedDynamicUpdateSlice) {
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32>({2})));
  auto update = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));
  auto dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          data_shape, gte1, update, starts));
  builder.AddInstruction(
      HloInstruction::CreateTuple({gte0, dynamic_update_slice}));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {dynamic_update_slice, starts, update, gte1},
      HloInstruction::FusionKind::kLoop);
  RunAnalysis();

  // The fusion instruction can share with tuple element 1.
  EXPECT_FALSE(points_to_analysis_->CanShareOperandBufferWithUser(tuple, {0},
                                                                  fusion, {}));
  EXPECT_TRUE(points_to_analysis_->CanShareOperandBufferWithUser(tuple, {1},
                                                                 fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, DynamicUpdateSliceCanShare) {
  auto builder = HloComputation::Builder(TestName());

  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  Shape update_shape = ShapeUtil::MakeShape(F32, {4});
  Shape starts_shape = ShapeUtil::MakeShape(S32, {1});
  auto data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape, "data"));
  auto update = builder.AddInstruction(
      HloInstruction::CreateParameter(1, update_shape, "update"));
  auto starts = builder.AddInstruction(
      HloInstruction::CreateParameter(2, starts_shape, "starts"));
  auto dus = builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      data_shape, data, update, starts));

  BuildModuleAndRunAnalysis(builder.Build());

  // The DynamicUpdateSlice instruction can share with the data operand, but not
  // with update or starts.
  EXPECT_TRUE(
      points_to_analysis_->CanShareOperandBufferWithUser(data, {}, dus, {}));
  EXPECT_FALSE(
      points_to_analysis_->CanShareOperandBufferWithUser(update, {}, dus, {}));
  EXPECT_FALSE(
      points_to_analysis_->CanShareOperandBufferWithUser(starts, {}, dus, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, SortCanShare) {
  auto builder = HloComputation::Builder(TestName());

  Shape keys_shape = ShapeUtil::MakeShape(F32, {8});
  auto keys = builder.AddInstruction(
      HloInstruction::CreateParameter(0, keys_shape, "keys"));
  auto sort =
      builder.AddInstruction(HloInstruction::CreateSort(keys_shape, 0, keys));

  BuildModuleAndRunAnalysis(builder.Build());

  EXPECT_TRUE(
      points_to_analysis_->CanShareOperandBufferWithUser(keys, {}, sort, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, SortCanShareWithTupleUser) {
  auto builder = HloComputation::Builder(TestName());

  Shape keys_shape = ShapeUtil::MakeShape(F32, {8});
  Shape values_shape = ShapeUtil::MakeShape(F32, {8});
  auto keys = builder.AddInstruction(
      HloInstruction::CreateParameter(0, keys_shape, "keys"));
  auto values = builder.AddInstruction(
      HloInstruction::CreateParameter(1, values_shape, "values"));
  auto sort = builder.AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeTupleShape({keys_shape, values_shape}), 0, keys, values));

  BuildModuleAndRunAnalysis(builder.Build());

  // The buffer for the keys can be shared with the first tuple entry.
  EXPECT_TRUE(
      points_to_analysis_->CanShareOperandBufferWithUser(keys, {}, sort, {0}));
  // The buffer for the values can be shared with the second tuple entry.
  EXPECT_TRUE(points_to_analysis_->CanShareOperandBufferWithUser(values, {},
                                                                 sort, {1}));
  // Verify that the buffers are not shared with the "wrong" tuple entry.
  EXPECT_FALSE(
      points_to_analysis_->CanShareOperandBufferWithUser(keys, {}, sort, {1}));
  EXPECT_FALSE(points_to_analysis_->CanShareOperandBufferWithUser(values, {},
                                                                  sort, {0}));
}

TEST_F(CanShareOperandBufferWithUserTest, FusedDotAdd) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto a = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 0.0}, {0.0, 1.0}})));
  auto b = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{2.0, 2.0}, {2.0, 2.0}})));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(data_shape, a, b, dot_dnums));

  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto add_operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {1}));

  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      data_shape, HloOpcode::kAdd, dot, add_operand));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {add, dot}, HloInstruction::FusionKind::kOutput);
  RunAnalysis();

  // Output fused dot add should be able to share buffer with 'add_operand'.
  EXPECT_TRUE(points_to_analysis_->CanShareOperandBufferWithUser(
      add_operand, {}, fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, OutputFusionCantAliasOperandBuffer) {
  auto builder = HloComputation::Builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});

  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {1}));

  auto reverse = builder.AddInstruction(
      HloInstruction::CreateReverse(data_shape, operand, {0, 1}));

  auto two = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{2.0, 2.0}, {2.0, 2.0}})));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(data_shape, HloOpcode::kAdd, reverse, two));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {add, two, reverse}, HloInstruction::FusionKind::kOutput);
  RunAnalysis();

  // Output fused operand->reverse->add cannot alias operand buffer 'operand'.
  EXPECT_FALSE(points_to_analysis_->CanShareOperandBufferWithUser(operand, {},
                                                                  fusion, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, WhileCanShare) {
  Shape data_shape = ShapeUtil::MakeShape(F32, {8});

  auto make_cond = [&data_shape]() {
    auto builder = HloComputation::Builder(TestName() + ".Cond");
    auto data = builder.AddInstruction(
        HloInstruction::CreateParameter(0, data_shape, "data"));
    builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kEq, data, data));
    return builder.Build();
  };

  auto make_body = [&data_shape]() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    auto data = builder.AddInstruction(
        HloInstruction::CreateParameter(0, data_shape, "data"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(data_shape, HloOpcode::kAdd, data, data));
    return builder.Build();
  };

  module_ = CreateNewModule();
  HloComputation* cond_computation =
      module_->AddEmbeddedComputation(make_cond());
  HloComputation* body_computation =
      module_->AddEmbeddedComputation(make_body());

  auto builder = HloComputation::Builder(TestName());
  auto data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape, "data"));
  auto whil = builder.AddInstruction(HloInstruction::CreateWhile(
      data_shape, cond_computation, body_computation, data));
  computation_ = module_->AddEntryComputation(builder.Build());

  RunAnalysis();

  // The While instruction can share with the data operand.
  EXPECT_TRUE(
      points_to_analysis_->CanShareOperandBufferWithUser(data, {}, whil, {}));
}

// Tests that Call can alias operand buffer if the only use of the operand
// in the called computation is an elementwise instruction.
TEST_F(CanShareOperandBufferWithUserTest, CallToComputationWithFusionRoot) {
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  // Build sub-computation with fusion root.
  auto sub_builder = HloComputation::Builder(TestName() + "_sub");
  auto sub_param = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "sub_param"));
  auto one = sub_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto ones = sub_builder.AddInstruction(
      HloInstruction::CreateBroadcast(shape, one, {1}));
  auto add = sub_builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, sub_param, ones));

  module_ = CreateNewModule();
  auto sub_computation = module_->AddEmbeddedComputation(sub_builder.Build());
  sub_computation->CreateFusionInstruction({add, ones},
                                           HloInstruction::FusionKind::kLoop);

  // Build entry-computation with kCall which calls 'sub_computation'.
  auto builder = HloComputation::Builder(TestName());

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto reverse =
      builder.AddInstruction(HloInstruction::CreateReverse(shape, param, {0}));
  auto call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {reverse}, sub_computation));
  computation_ = module_->AddEntryComputation(builder.Build());

  RunAnalysis();

  EXPECT_TRUE(points_to_analysis_->CanShareOperandBufferWithUser(reverse, {},
                                                                 call, {}));
}

TEST_F(CanShareOperandBufferWithUserTest, LoopFusionWithElementwiseOperand) {
  Shape full_shape = ShapeUtil::MakeShape(F32, {16, 32});
  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {16});

  auto builder = HloComputation::Builder(TestName() + "_fusion");
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, full_shape, "full"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, broadcast_shape, "small"));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(full_shape, param1, {0}));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      full_shape, HloOpcode::kAdd, param0, broadcast));

  BuildModule(builder.Build());
  auto fusion = computation_->CreateFusionInstruction(
      {add, broadcast}, HloInstruction::FusionKind::kLoop);
  RunAnalysis();

  EXPECT_TRUE(points_to_analysis_->CanShareOperandBufferWithUser(param0, {},
                                                                 fusion, {}));
  EXPECT_FALSE(points_to_analysis_->CanShareOperandBufferWithUser(param1, {},
                                                                  fusion, {}));
}

}  // namespace
}  // namespace xla
