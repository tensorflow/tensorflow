/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/analysis/shape_tracker.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

void ExpectStepsEqual(const std::vector<ShapeTracker::Step>& actual,
                      const std::vector<ShapeTracker::Step>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i].type, expected[i].type)
        << "Step mismatch at index " << i;
    EXPECT_EQ(actual[i].dimensions, expected[i].dimensions)
        << "Step mismatch at index " << i;
  }
}

using BufferView = ShapeTracker::BufferView;

}  // namespace

TEST(ShapeTrackerTest, AppendTranspose) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{4, 3, 2}));
}

TEST(ShapeTrackerTest, AppendTransposeWithDegenerate) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 1, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{4, 1, 2}));
}

TEST(ShapeTrackerTest, StackTransposes) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({1, 0, 2}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{4, 2, 3}));
}

TEST(ShapeTrackerTest, AppendReshapeFlatten) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendReshape({24}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{24}));
}

TEST(ShapeTrackerTest, AppendReshapeSplit) {
  Shape shape = ShapeUtil::MakeShape(F32, {6});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendReshape({2, 3}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{2, 3}));
}

TEST(ShapeTrackerTest, AppendReshapeMerge) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendReshape({6}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{6}));
}

TEST(ShapeTrackerTest, AppendReshapeAllOnes) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendReshape({1, 1}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{1, 1}));
}

TEST(ShapeTrackerTest, AppendReshapeProductMismatch) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  absl::Status status = tracker.AppendReshape({10, 2});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, GetInvertedSimple) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());

  auto inverted_or = tracker.GetInverted();
  ASSERT_TRUE(inverted_or.ok());
  ShapeTracker inverted = std::move(inverted_or).value();

  EXPECT_EQ(inverted.input_shape().dimensions(), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(inverted.output_shape().dimensions(), (std::vector<int64_t>{2, 3}));
}

TEST(ShapeTrackerTest, AppendBitcastSimple) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {6});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {2, 3});

  ShapeTracker tracker(src_shape);
  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{2, 3}));
}

TEST(ShapeTrackerTest, AppendBitcastWithLayout) {
  Shape src_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {1, 0});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {6});

  ShapeTracker tracker(src_shape);
  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{6}));
}

TEST(ShapeTrackerTest, AppendBitcastFail) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {3, 2});

  ShapeTracker tracker(src_shape);
  EXPECT_FALSE(tracker.AppendBitcast(src_shape, dst_shape).ok());
}

TEST(ShapeTrackerTest, AppendBitcastTricky) {
  Shape src_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 13, 33, 1, 35},
                                                        {4, 0, 2, 1, 3});
  Shape dst_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {39, 1, 11, 10, 7},
                                                        {4, 3, 2, 0, 1});

  ShapeTracker tracker(src_shape);
  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{39, 1, 11, 10, 7}));
}

TEST(ShapeTrackerTest, ConcatenateFrom) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker1(shape1);
  ASSERT_TRUE(tracker1.AppendTranspose({1, 0}).ok());

  Shape shape2 = ShapeUtil::MakeShape(F32, {3, 2});
  ShapeTracker tracker2(shape2);
  ASSERT_TRUE(tracker2.AppendReshape({6}).ok());

  ASSERT_TRUE(tracker1.ConcatenateFrom(tracker2).ok());

  EXPECT_EQ(tracker1.output_shape().dimensions(), (std::vector<int64_t>{6}));

  std::vector<ShapeTracker::Step> steps = tracker1.GetSteps();
  ASSERT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{6}));
}

TEST(ShapeTrackerTest, ConcatenateFromIncompatible) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker1(shape1);
  ASSERT_TRUE(tracker1.AppendTranspose({1, 0}).ok());

  Shape shape2 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker2(shape2);

  absl::Status status = tracker1.ConcatenateFrom(tracker2);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, IdentityTransformationProducesNoSteps) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  EXPECT_TRUE(tracker.GetSteps().empty());
  EXPECT_EQ(tracker.DebugString(), "[2,3]");
}

TEST(ShapeTrackerTest, GetStepsSingleTranspose) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1, 0}));
}

TEST(ShapeTrackerTest, GetInvertedDoubleIdentity) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({6, 4}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());

  auto inverted_or = tracker.GetInverted();
  ASSERT_TRUE(inverted_or.ok());
  ShapeTracker inverted = std::move(inverted_or).value();

  auto double_inverted_or = inverted.GetInverted();
  ASSERT_TRUE(double_inverted_or.ok());
  ShapeTracker double_inverted = std::move(double_inverted_or).value();

  EXPECT_EQ(double_inverted.input_shape(), tracker.input_shape());
  EXPECT_EQ(double_inverted.output_shape(), tracker.output_shape());

  EXPECT_EQ(double_inverted.DebugString(), tracker.DebugString());
  ExpectStepsEqual(double_inverted.GetSteps(), tracker.GetSteps());
}

TEST(ShapeTrackerTest, TrickyCaseDoubleInversion) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 13, 33, 1, 35});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendReshape({26, 35, 33}).ok());
  ASSERT_TRUE(tracker.AppendReshape({13, 2, 7, 5, 11, 3}).ok());
  ASSERT_TRUE(tracker.AppendReshape({13, 14, 55, 3}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({3, 0, 2, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({39, 1, 11, 10, 7}).ok());

  auto inverted_or = tracker.GetInverted();
  ASSERT_TRUE(inverted_or.ok());
  ShapeTracker inverted = std::move(inverted_or).value();

  auto double_inverted_or = inverted.GetInverted();
  ASSERT_TRUE(double_inverted_or.ok());
  ShapeTracker double_inverted = std::move(double_inverted_or).value();

  EXPECT_EQ(double_inverted.input_shape(), tracker.input_shape());
  EXPECT_EQ(double_inverted.output_shape(), tracker.output_shape());

  EXPECT_EQ(double_inverted.DebugString(), tracker.DebugString());
  ExpectStepsEqual(double_inverted.GetSteps(), tracker.GetSteps());
}

TEST(ShapeTrackerTest, ThreeProjectionsDoubleInversion) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({3, 8}).ok());  // Forces copy
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());
  ASSERT_TRUE(tracker.AppendReshape({3, 8}).ok());  // Forces copy

  auto inverted_or = tracker.GetInverted();
  ASSERT_TRUE(inverted_or.ok());
  ShapeTracker inverted = std::move(inverted_or).value();

  auto double_inverted_or = inverted.GetInverted();
  ASSERT_TRUE(double_inverted_or.ok());
  ShapeTracker double_inverted = std::move(double_inverted_or).value();

  EXPECT_EQ(double_inverted.input_shape(), tracker.input_shape());
  EXPECT_EQ(double_inverted.output_shape(), tracker.output_shape());

  EXPECT_EQ(double_inverted.DebugString(), tracker.DebugString());
  ExpectStepsEqual(double_inverted.GetSteps(), tracker.GetSteps());
}

TEST(ShapeTrackerTest, FoldRedundantProjections) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({3, 8}).ok());  // Forces copy
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());
  ASSERT_TRUE(tracker.AppendReshape({3, 8}).ok());  // Forces copy

  auto inverted_or = tracker.GetInverted();
  ASSERT_TRUE(inverted_or.ok());
  ShapeTracker inverted = std::move(inverted_or).value();

  ASSERT_TRUE(tracker.ConcatenateFrom(inverted).ok());

  EXPECT_EQ(tracker.output_shape(), tracker.input_shape());
  EXPECT_TRUE(tracker.GetSteps().empty());
}

TEST(ShapeTrackerTest, GetStepsAndDebugString) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({0, 2, 1}).ok());

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 3);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(steps[2].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[2].dimensions, (std::vector<int64_t>{1, 3, 2}));

  EXPECT_EQ(tracker.DebugString(), "[1,2,3] -> T[3,1,2] -> R[1,3,2]");
}

TEST(ShapeTrackerTest, GetStepsFallback) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({8, 3}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{8, 3}));

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 3);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{6, 4}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(steps[2].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[2].dimensions, (std::vector<int64_t>{8, 3}));

  EXPECT_EQ(tracker.DebugString(), "[2,3,4] -> T[4,2,3] -> R[8,3]");
}

TEST(ShapeTrackerTest, PrependTranspose) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.PrependTranspose({1, 0, 2}).ok());

  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{3, 2, 4}));
  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1, 0, 2}));
}

TEST(ShapeTrackerTest, PrependTransposeNonSymmetric) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.PrependTranspose({1, 2, 0}).ok());
  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{4, 2, 3}));

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 3);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{4, 6}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(steps[2].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[2].dimensions, (std::vector<int64_t>{2, 3, 4}));

  EXPECT_EQ(tracker.DebugString(), "[4,2,3] -> T[2,3,4]");
}

TEST(ShapeTrackerTest, ScalarShapeBitcast) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {});
  Shape dst_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 1}, {1, 0});
  ShapeTracker tracker(src_shape);

  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{1, 1}));

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1, 1}));
}

TEST(ShapeTrackerTest, PrependReshape) {
  Shape shape = ShapeUtil::MakeShape(F32, {6});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.PrependReshape({2, 3}).ok());

  EXPECT_EQ(tracker.input_shape().dimensions(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{6}));

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{6}));
}

TEST(ShapeTrackerTest, PrependBitcast) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {6});

  ShapeTracker tracker(dst_shape);

  ASSERT_TRUE(tracker.PrependBitcast(src_shape, dst_shape).ok());

  EXPECT_EQ(tracker.input_shape().dimensions(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{6}));
}

TEST(ShapeTrackerTest, AppendInstructionTranspose) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  auto param = HloInstruction::CreateParameter(0, shape, "param");
  auto transpose = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), param.get(), {2, 1, 0});

  ASSERT_TRUE(tracker.AppendInstruction(transpose.get()).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{4, 3, 2}));
}

TEST(ShapeTrackerTest, AppendInstructionReshape) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  auto param = HloInstruction::CreateParameter(0, shape, "param");
  auto reshape = HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {24}),
                                               param.get());

  ASSERT_TRUE(tracker.AppendInstruction(reshape.get()).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{24}));
}

TEST(ShapeTrackerTest, AppendInstructionBitcast) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {6});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {2, 3});

  ShapeTracker tracker(src_shape);

  auto param = HloInstruction::CreateParameter(0, src_shape, "param");
  auto bitcast = HloInstruction::CreateBitcast(dst_shape, param.get());

  ASSERT_TRUE(tracker.AppendInstruction(bitcast.get()).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{2, 3}));
}

TEST(ShapeTrackerTest, AppendInstructionUnsupported) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  auto param = HloInstruction::CreateParameter(0, shape, "param");
  auto negate =
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param.get());

  absl::Status status = tracker.AppendInstruction(negate.get());
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, AppendInstructionIncompatibleShape) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  Shape wrong_shape = ShapeUtil::MakeShape(F32, {3, 2, 4});
  auto param = HloInstruction::CreateParameter(0, wrong_shape, "param");
  auto transpose = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 2, 3}), param.get(), {2, 1, 0});

  absl::Status status = tracker.AppendInstruction(transpose.get());
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, PrependInstructionTranspose) {
  Shape shape = ShapeUtil::MakeShape(F32, {4, 3, 2});
  ShapeTracker tracker(shape);

  Shape param_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto param = HloInstruction::CreateParameter(0, param_shape, "param");
  auto transpose =
      HloInstruction::CreateTranspose(shape, param.get(), {2, 1, 0});

  ASSERT_TRUE(tracker.PrependInstruction(transpose.get()).ok());

  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));
}

TEST(ShapeTrackerTest, PrependInstructionReshape) {
  Shape shape = ShapeUtil::MakeShape(F32, {24});
  ShapeTracker tracker(shape);

  Shape param_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto param = HloInstruction::CreateParameter(0, param_shape, "param");
  auto reshape = HloInstruction::CreateReshape(shape, param.get());

  ASSERT_TRUE(tracker.PrependInstruction(reshape.get()).ok());

  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));
}

TEST(ShapeTrackerTest, PrependInstructionBitcast) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {6});

  ShapeTracker tracker(dst_shape);

  auto param = HloInstruction::CreateParameter(0, src_shape, "param");
  auto bitcast = HloInstruction::CreateBitcast(dst_shape, param.get());

  ASSERT_TRUE(tracker.PrependInstruction(bitcast.get()).ok());

  EXPECT_EQ(tracker.input_shape().dimensions(), (std::vector<int64_t>{2, 3}));
}

TEST(ShapeTrackerTest, PrependInstructionUnsupported) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  auto param = HloInstruction::CreateParameter(0, shape, "param");
  auto negate =
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param.get());

  absl::Status status = tracker.PrependInstruction(negate.get());
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, PrependInstructionIncompatibleShape) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  Shape wrong_shape = ShapeUtil::MakeShape(F32, {3, 2, 4});
  auto param = HloInstruction::CreateParameter(0, wrong_shape, "param");
  auto transpose = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 2, 3}), param.get(), {2, 1, 0});

  absl::Status status = tracker.PrependInstruction(transpose.get());
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}
TEST(ShapeTrackerTest, FromProducerConsumerTransposeReshape) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto param = HloInstruction::CreateParameter(0, shape, "param");

  auto transpose = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), param.get(), {2, 1, 0});

  auto reshape = HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {24}),
                                               transpose.get());

  auto tracker_or =
      ShapeTracker::FromProducerConsumer(transpose.get(), reshape.get());
  ASSERT_TRUE(tracker_or.ok());
  ShapeTracker tracker = std::move(tracker_or).value();

  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{4, 3, 2}));
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{24}));
}

TEST(ShapeTrackerTest, FromProducerConsumerReshapeTranspose) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});

  auto param = HloInstruction::CreateParameter(0, shape, "param");
  auto reshape = HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 4}), param.get());
  auto transpose = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 6}), reshape.get(), {1, 0});

  auto tracker_or =
      ShapeTracker::FromProducerConsumer(reshape.get(), transpose.get());
  ASSERT_TRUE(tracker_or.ok());
  ShapeTracker tracker = std::move(tracker_or).value();

  EXPECT_EQ(tracker.input_shape().dimensions(), (std::vector<int64_t>{6, 4}));
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{4, 6}));
}

TEST(ShapeTrackerTest, FromProducerConsumerSingle) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto param = HloInstruction::CreateParameter(0, shape, "param");

  auto transpose = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), param.get(), {2, 1, 0});

  auto tracker_or =
      ShapeTracker::FromProducerConsumer(transpose.get(), transpose.get());
  ASSERT_TRUE(tracker_or.ok());
  ShapeTracker tracker = std::move(tracker_or).value();

  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{4, 3, 2}));
  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{4, 3, 2}));
}

TEST(ShapeTrackerTest, FromProducerConsumerNotAncestor) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto param = HloInstruction::CreateParameter(0, shape, "param");

  auto transpose1 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), param.get(), {2, 1, 0});

  auto transpose2 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), param.get(), {2, 1, 0});

  auto tracker_or =
      ShapeTracker::FromProducerConsumer(transpose1.get(), transpose2.get());
  EXPECT_FALSE(tracker_or.ok());
  EXPECT_EQ(tracker_or.status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(ShapeTrackerTest, FromProducerConsumerMultiOperand) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto param = HloInstruction::CreateParameter(0, shape, "param");

  auto transpose = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), param.get(), {2, 1, 0});

  auto add = HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {4, 3, 2}),
                                          HloOpcode::kAdd, transpose.get(),
                                          transpose.get());

  auto tracker_or =
      ShapeTracker::FromProducerConsumer(transpose.get(), add.get());
  EXPECT_FALSE(tracker_or.ok());
  EXPECT_EQ(tracker_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, AppendBitcastSourcePhysicalLogicalDiffer) {
  // Logical shape {2, 3}, but layout has dim 0 as minor, dim 1 as major.
  Shape src_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {0, 1});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {6});

  ShapeTracker tracker(src_shape);
  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{6}));

  // Verify that the tracker correctly simplified this to a Transpose followed
  // by a Reshape.
  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{6}));
}

TEST(ShapeTrackerTest, AppendBitcastNonDefaultLayoutCorrectness) {
  Shape src_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3, 4}, {1, 2, 0});
  Shape dst_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 12}, {1, 0});

  ShapeTracker tracker(src_shape);
  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());

  // Check that the output shape matches.
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{2, 12}));

  // Check that steps represent the correct logical transposition and reshape.
  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{0, 2, 1}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{2, 12}));
}

TEST(ShapeTrackerTest, SimplifyConsecutiveReshapes) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 512, 5376});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());
  ASSERT_TRUE(tracker.AppendReshape({1, 512, 5376}).ok());

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();

  // The intermediate reshape to [512, 5376] is skipped, and we go directly
  // from [5376, 512] to [1, 512, 5376] via a single reshape.
  ASSERT_EQ(steps.size(), 3);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{512, 5376}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(steps[2].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[2].dimensions, (std::vector<int64_t>{1, 512, 5376}));
}

TEST(ShapeTrackerTest, AppendBitcastOneSixToSix) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {1, 6});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {6});

  ShapeTracker tracker(src_shape);
  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{6}));

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{6}));
}

TEST(ShapeTrackerTest, AppendBitcastSixToOneSix) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {6});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {1, 6});

  ShapeTracker tracker(src_shape);
  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{1, 6}));

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1, 6}));
}

TEST(ShapeTrackerTest, OneElementReshapeTransposeBitcastChain) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1});
  ShapeTracker tracker(shape);

  // Append operations
  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());
  ASSERT_TRUE(tracker.AppendReshape({1, 1}).ok());
  ASSERT_TRUE(tracker
                  .AppendBitcast(ShapeUtil::MakeShape(F32, {1, 1}),
                                 ShapeUtil::MakeShape(F32, {1}))
                  .ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{1}));
  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1}));

  // Prepend operations
  ShapeTracker tracker2(ShapeUtil::MakeShape(F32, {1}));
  ASSERT_TRUE(tracker2.PrependTranspose({0}).ok());
  ASSERT_TRUE(tracker2.PrependReshape({1, 1}).ok());
  ASSERT_TRUE(tracker2
                  .PrependBitcast(ShapeUtil::MakeShape(F32, {1, 1, 1}),
                                  ShapeUtil::MakeShape(F32, {1, 1}))
                  .ok());

  std::vector<ShapeTracker::Step> steps2 = tracker2.GetSteps();
  ASSERT_EQ(steps2.size(), 1);
  EXPECT_EQ(steps2[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps2[0].dimensions, (std::vector<int64_t>{1}));
}

TEST(ShapeTrackerTest, OptimizeStepsDebugString) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({8, 3}).ok());

  EXPECT_EQ(tracker.DebugString(), "[2,3,4] -> T[4,2,3] -> R[8,3]");
  EXPECT_EQ(tracker.DebugString(true), "[2,3,4] -> T[4,2,3] -> R[8,3]");
  EXPECT_EQ(tracker.DebugString(false),
            "[2,3,4] -> R[6,4] -> T[4,6] -> R[8,3]");

  ShapeTracker tracker2(ShapeUtil::MakeShape(F32, {1, 2, 3}));
  ASSERT_TRUE(tracker2.AppendTranspose({0, 2, 1}).ok());
  EXPECT_EQ(tracker2.DebugString(true), "[1,2,3] -> T[3,1,2] -> R[1,3,2]");
  EXPECT_EQ(tracker2.DebugString(false),
            "[1,2,3] -> R[2,3] -> T[3,2] -> R[1,3,2]");
}

TEST(ShapeTrackerTest, OptimizeStepsToInstructionChain) {
  auto module = std::make_unique<HloModule>("module", HloModuleConfig());
  HloComputation::Builder builder("comp");

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));

  module->AddEntryComputation(builder.Build());

  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({8, 3}).ok());

  // 1. Test with avoid_combining_reshapes = false (glued legacy path)
  {
    auto chain_or = tracker.ToInstructionChain(param, false);
    ASSERT_TRUE(chain_or.ok());
    HloInstruction* root = chain_or.value();

    // Should produce: Root -> Reshape -> Transpose -> Reshape -> parameter
    EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
    const HloInstruction* next1 = root->operand(0);
    EXPECT_EQ(next1->opcode(), HloOpcode::kTranspose);
    const HloInstruction* next2 = next1->operand(0);
    EXPECT_EQ(next2->opcode(), HloOpcode::kReshape);
    EXPECT_EQ(next2->operand(0), param);
  }

  // 2. Test with avoid_combining_reshapes = true (optimized decombined path)
  {
    auto chain_or = tracker.ToInstructionChain(param, true);
    ASSERT_TRUE(chain_or.ok());
    HloInstruction* root = chain_or.value();

    // Should produce: Root -> Reshape -> Transpose -> parameter (direct
    // transpose on parameter!)
    EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
    const HloInstruction* next1 = root->operand(0);
    EXPECT_EQ(next1->opcode(), HloOpcode::kTranspose);
    EXPECT_EQ(next1->operand(0), param);
  }
}

TEST(ShapeTrackerTest, OptimizeStepsDegenerateDimensionsCorrectness) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 1, 3});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());

  EXPECT_EQ(tracker.DebugString(false),
            "[2,1,3] -> R[2,3] -> T[3,2] -> R[3,1,2]");
  EXPECT_EQ(tracker.DebugString(true), "[2,1,3] -> T[1,3,2] -> R[3,1,2]");
}

TEST(ShapeTrackerTest, OptimizeStepsScalarReshape) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 1});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendReshape({}).ok());

  EXPECT_EQ(tracker.DebugString(true), "[1,1] -> R[]");
  EXPECT_EQ(tracker.DebugString(false), "[1,1] -> R[]");
}
TEST(ShapeTrackerTest, OptimizeStepsOOBWithTrailingDegenerates) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendReshape({6, 1, 1}).ok());
  EXPECT_NO_FATAL_FAILURE(tracker.DebugString(true));
}

namespace {

TEST(BufferViewTest, FromStridesAndExtentsSuccess1D) {
  auto view_or = BufferView::FromStridesAndExtents({4}, {5});
  ASSERT_TRUE(view_or.ok());
  EXPECT_EQ(view_or->strides(), (std::vector<int64_t>{4}));
  EXPECT_EQ(view_or->extents(), (std::vector<int64_t>{5}));
}

TEST(BufferViewTest, FromStridesAndExtentsSuccessMultiDim) {
  auto view_or = BufferView::FromStridesAndExtents({12, 4, 1}, {2, 3, 4});
  ASSERT_TRUE(view_or.ok());
  EXPECT_EQ(view_or->strides(), (std::vector<int64_t>{12, 4, 1}));
  EXPECT_EQ(view_or->extents(), (std::vector<int64_t>{2, 3, 4}));
}

TEST(BufferViewTest, FromStridesAndExtentsSizeMismatch) {
  auto view_or = BufferView::FromStridesAndExtents({4, 1}, {5});
  EXPECT_FALSE(view_or.ok());
  EXPECT_EQ(view_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(BufferViewTest, FromStridesAndExtentsInvalidStride) {
  auto view_or = BufferView::FromStridesAndExtents({0}, {5});
  EXPECT_FALSE(view_or.ok());
  EXPECT_EQ(view_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(BufferViewTest, FromStridesAndExtentsInvalidExtent) {
  auto view_or = BufferView::FromStridesAndExtents({4}, {-1});
  EXPECT_FALSE(view_or.ok());
  EXPECT_EQ(view_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(BufferViewTest, FromStridesAndExtentsOverlap) {
  auto view_or = BufferView::FromStridesAndExtents({2, 3}, {2, 2});
  EXPECT_FALSE(view_or.ok());
  EXPECT_EQ(view_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(BufferViewTest, FromStridesAndExtentsWithGaps) {
  auto view_or = BufferView::FromStridesAndExtents({2, 5}, {2, 3});
  ASSERT_TRUE(view_or.ok());
  EXPECT_EQ(view_or->strides(), (std::vector<int64_t>{2, 5}));
  EXPECT_EQ(view_or->extents(), (std::vector<int64_t>{2, 3}));
}

TEST(BufferViewTest, FromShapeScalar) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  BufferView view = BufferView::FromShape(shape);
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{1}));
  EXPECT_EQ(view.extents(), (std::vector<int64_t>{1}));
}

TEST(BufferViewTest, FromShape1D) {
  Shape shape = ShapeUtil::MakeShape(F32, {10});
  BufferView view = BufferView::FromShape(shape);
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{1}));
  EXPECT_EQ(view.extents(), (std::vector<int64_t>{10}));
}

TEST(BufferViewTest, ElementsIn) {
  // Scalar
  {
    Shape shape = ShapeUtil::MakeShape(F32, {});
    BufferView view = BufferView::FromShape(shape);
    EXPECT_EQ(view.ElementsIn(), 1);
  }
  // 1D
  {
    Shape shape = ShapeUtil::MakeShape(F32, {10});
    BufferView view = BufferView::FromShape(shape);
    EXPECT_EQ(view.ElementsIn(), 10);
  }
  // Multi-dim
  {
    Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
    BufferView view = BufferView::FromShape(shape);
    EXPECT_EQ(view.ElementsIn(), 24);
  }
  // Empty view
  {
    auto view_or = BufferView::FromStridesAndExtents({}, {});
    ASSERT_TRUE(view_or.ok());
    EXPECT_EQ(view_or->ElementsIn(), 0);
  }
}

TEST(BufferViewTest, FromShapeMultiDim) {
  // Shape [2, 3, 4]
  // Expected strides: [12, 4, 1]
  // Expected extents: [2, 3, 4]
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  BufferView view = BufferView::FromShape(shape);
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{12, 4, 1}));
  EXPECT_EQ(view.extents(), (std::vector<int64_t>{2, 3, 4}));
}

TEST(BufferViewTest, PackAlreadyPacked) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({3, 1}, {2, 3}));
  BufferView expected = view;
  view.Pack();
  EXPECT_EQ(view, expected);
}

TEST(BufferViewTest, PackWithGaps) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({8, 2}, {2, 3}));
  view.Pack();
  // Should compress to strides [3, 1]
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{3, 1}));
  EXPECT_EQ(view.extents(), (std::vector<int64_t>{2, 3}));
}

TEST(BufferViewTest, TryIntersectWithStrideExtentCompatibleFull) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({4}, {5}));
  auto result = view.TryIntersectWith(4, 5);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->strides(), (std::vector<int64_t>{4}));
  EXPECT_EQ(result->extents(), (std::vector<int64_t>{5}));
}

TEST(BufferViewTest, TryIntersectWithStrideExtentCompatiblePartialSameStride) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({4}, {5}));
  auto result = view.TryIntersectWith(4, 3);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->strides(), (std::vector<int64_t>{4}));
  EXPECT_EQ(result->extents(), (std::vector<int64_t>{3}));
}

TEST(BufferViewTest,
     TryIntersectWithStrideExtentCompatiblePartialLargerStride) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({4}, {5}));
  auto result = view.TryIntersectWith(8, 2);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->strides(), (std::vector<int64_t>{8}));
  EXPECT_EQ(result->extents(), (std::vector<int64_t>{2}));
}

TEST(BufferViewTest, TryIntersectWithStrideExtentIncompatible) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({4}, {5}));
  auto result = view.TryIntersectWith(3, 3);
  EXPECT_FALSE(result.has_value());
}

TEST(BufferViewTest, TryIntersectWithNonAlignedLimit) {
  // Strictly speaking, the intersection (if we describe it as a set of
  // addressable elements) can be expressed as [8], [3] in this case. But that
  // would result in non-aligned strides, we don't allow this.
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({8}, {3}));
  auto result = view.TryIntersectWith(4, 5);
  EXPECT_FALSE(result.has_value());
}

TEST(BufferViewTest, TryIntersectWithOtherCompatible) {
  ASSERT_OK_AND_ASSIGN(BufferView view1,
                       BufferView::FromStridesAndExtents({4}, {5}));
  ASSERT_OK_AND_ASSIGN(BufferView view2,
                       BufferView::FromStridesAndExtents({8}, {2}));
  auto result = view1.TryIntersectWith(view2);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, view2);
}

TEST(BufferViewTest, TryIntersectWithOtherIncompatible) {
  ASSERT_OK_AND_ASSIGN(BufferView view1,
                       BufferView::FromStridesAndExtents({4}, {5}));
  ASSERT_OK_AND_ASSIGN(BufferView view2,
                       BufferView::FromStridesAndExtents({3}, {3}));
  auto result = view1.TryIntersectWith(view2);
  EXPECT_FALSE(result.has_value());
}

TEST(BufferViewTest, TryIntersectWithOtherUpperSliceLEOutS) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({2}, {3}));
  auto result = view.TryIntersectWith(12, 2);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->strides().empty());
  EXPECT_TRUE(result->extents().empty());
}

TEST(BufferViewTest, MergeAdjacentDimensionsContiguous) {
  ASSERT_OK_AND_ASSIGN(BufferView view, BufferView::FromStridesAndExtents(
                                            {12, 4, 1}, {2, 3, 4}));
  view.MergeAdjacentDimensions();
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{1}));
  EXPECT_EQ(view.extents(), (std::vector<int64_t>{24}));
}

TEST(BufferViewTest, MergeAdjacentDimensionsNonContiguous) {
  ASSERT_OK_AND_ASSIGN(BufferView view, BufferView::FromStridesAndExtents(
                                            {16, 4, 1}, {2, 3, 4}));
  view.MergeAdjacentDimensions();
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{16, 1}));
  EXPECT_EQ(view.extents(), (std::vector<int64_t>{2, 12}));
}

TEST(BufferViewTest, MergeAdjacentDimensionsMultiple) {
  ASSERT_OK_AND_ASSIGN(BufferView view, BufferView::FromStridesAndExtents(
                                            {48, 16, 4, 1}, {2, 3, 4, 4}));
  view.MergeAdjacentDimensions();
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{1}));
  EXPECT_EQ(view.extents(), (std::vector<int64_t>{96}));
}

TEST(BufferViewTest, MergeAdjacentDimensionsPartial) {
  ASSERT_OK_AND_ASSIGN(BufferView view, BufferView::FromStridesAndExtents(
                                            {48, 12, 4}, {2, 2, 3}));
  view.MergeAdjacentDimensions();
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{48, 4}));
  EXPECT_EQ(view.extents(), (std::vector<int64_t>{2, 6}));
}

TEST(BufferViewTest, TryUnflattenFlatTo3D) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({1}, {24}));
  auto result = view.TryUnflatten({2, 3, 4});
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), 3);

  auto expect_eq_view = [&](const BufferView& actual,
                            absl::Span<const int64_t> strides,
                            absl::Span<const int64_t> extents) {
    ASSERT_OK_AND_ASSIGN(auto expected,
                         BufferView::FromStridesAndExtents(strides, extents));
    EXPECT_EQ(actual, expected);
  };
  expect_eq_view((*result)[0], {12}, {2});
  expect_eq_view((*result)[1], {4}, {3});
  expect_eq_view((*result)[2], {1}, {4});
}

TEST(BufferViewTest, TryUnflattenNonContiguousCompatible) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({16, 1}, {2, 12}));
  auto result = view.TryUnflatten({2, 3, 4});
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), 3);

  auto expect_eq_view = [&](const BufferView& actual,
                            absl::Span<const int64_t> strides,
                            absl::Span<const int64_t> extents) {
    ASSERT_OK_AND_ASSIGN(auto expected,
                         BufferView::FromStridesAndExtents(strides, extents));
    EXPECT_EQ(actual, expected);
  };
  expect_eq_view((*result)[0], {16}, {2});
  expect_eq_view((*result)[1], {4}, {3});
  expect_eq_view((*result)[2], {1}, {4});
}

TEST(BufferViewTest, TryUnflattenIncompatible) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({1}, {24}));
  auto result = view.TryUnflatten({5, 5});
  EXPECT_FALSE(result.has_value());
}

TEST(BufferViewTest, AsTransformationIdentity) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({3, 1}, {2, 3}));
  auto trans = view.AsTransformation();
  EXPECT_EQ(trans.input_reshape, (llvm::SmallVector<int64_t, 6>{2, 3}));
  EXPECT_EQ(trans.transpose, (llvm::SmallVector<int64_t, 6>{0, 1}));
}

TEST(BufferViewTest, AsTransformationPermuted) {
  ASSERT_OK_AND_ASSIGN(BufferView view,
                       BufferView::FromStridesAndExtents({1, 3}, {3, 2}));
  auto trans = view.AsTransformation();
  EXPECT_EQ(trans.input_reshape, (llvm::SmallVector<int64_t, 6>{2, 3}));
  EXPECT_EQ(trans.transpose, (llvm::SmallVector<int64_t, 6>{1, 0}));
}

TEST(ShapeTrackerTest, Narrow1DKeep) {
  Shape shape = ShapeUtil::MakeShape(F32, {10});
  ShapeTracker tracker(shape);

  auto narrowed_or = tracker.Narrow({0});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{10}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{10}));
}

TEST(ShapeTrackerTest, Narrow1DDrop) {
  Shape shape = ShapeUtil::MakeShape(F32, {10});
  ShapeTracker tracker(shape);

  auto narrowed_or = tracker.Narrow({});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{}));
}

TEST(ShapeTrackerTest, NarrowMultiDimKeepSubset) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  auto narrowed_or = tracker.Narrow({0, 2});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(narrowed.DebugString(true), "[2,4]");
}

TEST(ShapeTrackerTest, NarrowTransposedShape) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());

  auto narrowed_or = tracker.Narrow({0, 2});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{4, 2}));
  EXPECT_EQ(narrowed.DebugString(true), "[2,4] -> T[4,2]");
}

TEST(ShapeTrackerTest, NarrowToScalar) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  auto narrowed_or = tracker.Narrow({});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{}));
}

TEST(ShapeTrackerTest, NarrowDegenerateCollapsesOutputButPreservesInputRank) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 1, 4});
  ShapeTracker tracker(shape);

  // Narrow keeping the degenerate dimension 1.
  // Kept elements is 1, so it triggers the kept_elements == 1 logic.
  auto narrowed_or = tracker.Narrow({1});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  // Input shape must preserve rank 1 (corresponding to dims_to_keep).
  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{1}));
  // Output shape must collapse to scalar (rank 0).
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{}));
}

TEST(ShapeTrackerTest, NarrowTransposedShapeWithCompaction) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  // This transpose results in strides [1, 12, 4] which compacts dim 1 & 2.
  // The compacted BufferView has rank 2: strides [1, 4], extents [4, 6].
  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());

  // We keep dims 2 and 0 (sizes 4 and 2), dropping dim 1.
  // If SliceProjections is implemented robustly with intersection,
  // it handles the rank-change/compaction gracefully.
  auto narrowed_or = tracker.Narrow({0, 2});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{4, 2}));
  EXPECT_EQ(narrowed.DebugString(true), "[2,4] -> T[4,2]");
}

TEST(ShapeTrackerTest, NarrowValidationErrors) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  EXPECT_FALSE(tracker.Narrow({3}).ok());
  EXPECT_FALSE(tracker.Narrow({-1}).ok());
  EXPECT_FALSE(tracker.Narrow({0, 0}).ok());
  EXPECT_FALSE(tracker.Narrow({2, 0, 2}).ok());
}

TEST(ShapeTrackerTest, NarrowUnsorted) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  // Narrow keeping dims 2 and 0 (sizes 4 and 2), in that order.
  // Input to narrowed tracker should be [4, 2].
  // Output of narrowed tracker should be [2, 4] because original was identity
  // and we implicitly start with transpose [4, 2] -> [2, 4].
  auto narrowed_or = tracker.Narrow({2, 0});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{4, 2}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(narrowed.DebugString(true), "[4,2] -> T[2,4]");
}

TEST(ShapeTrackerTest, NarrowWithMultipleProjections) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());  // output [4, 2, 3]
  ASSERT_TRUE(
      tracker.AppendReshape({3, 8}).ok());  // Forces copy, output [3, 8]

  auto narrowed_or = tracker.Narrow({0, 2});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{4, 2}));
  EXPECT_EQ(narrowed.DebugString(true), "[2,4] -> T[4,2]");
}

TEST(ShapeTrackerTest, NarrowFoldsProjectionsToReshape) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  // Append transpose that compacts dims 1 and 2.
  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());  // output [4, 2, 3]
  // Append reshape that forces a copy.
  ASSERT_TRUE(tracker.AppendReshape({3, 8}).ok());  // output [3, 8]

  // Verify we have multiple steps.
  EXPECT_GT(tracker.GetSteps().size(), 1);

  // Narrow keeping only input dimensions 0 and 1.
  auto narrowed_or = tracker.Narrow({0, 1});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  // The resulting tracker should have folded the projections and simplified to
  // 1 reshape step.
  EXPECT_EQ(narrowed.GetSteps().size(), 1);
  EXPECT_EQ(narrowed.DebugString(true), "[2,3] -> R[6]");
}

TEST(ShapeTrackerTest, NarrowFoldsProjectionsToTranspose) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  // Append transpose that compacts dims 1 and 2.
  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());  // output [4, 2, 3]
  // Append reshape that forces a copy.
  ASSERT_TRUE(tracker.AppendReshape({3, 8}).ok());  // output [3, 8]

  // Verify we have multiple steps.
  EXPECT_GT(tracker.GetSteps().size(), 1);

  // Narrow keeping only input dimensions 0 and 2.
  auto narrowed_or = tracker.Narrow({0, 2});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();

  // The resulting narrowed tracker should have folded the projections.
  // If folding occurred, it should be represented by a single transpose step.
  std::vector<ShapeTracker::Step> steps = narrowed.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1, 0}));
}

TEST(ShapeTrackerTest, RemappingInSliceProjectionsWorksForTransposedInput) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 10});
  ShapeTracker tracker(shape);

  // P1 will be unflattened and permuted to strides {1, 2}, extents {2, 5}.
  ASSERT_TRUE(tracker.AppendReshape({5, 2}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());

  // P2 will be unflattened and permuted to strides {1, 5}, extents {5, 2}.
  ASSERT_TRUE(tracker.AppendReshape({2, 5}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());

  // P1 intersection succeeds and remapped_slice is correctly formed.
  // P2 intersection succeeds on the remapped slice.
  auto narrowed_or = tracker.Narrow({1});
  ASSERT_TRUE(narrowed_or.ok());
  ShapeTracker narrowed = std::move(narrowed_or).value();
  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{10}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{10}));
}

TEST(ShapeTrackerTest, NarrowIncompatibleSlice) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 5, 3});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendReshape({5, 6}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());

  auto narrowed_or = tracker.Narrow({1});
  EXPECT_FALSE(narrowed_or.ok());
  EXPECT_EQ(narrowed_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(narrowed_or.status().message(),
              ::testing::HasSubstr("Slice is incompatible with projection"));
}

TEST(ShapeTrackerTest, NarrowSkipFirstDimensionInFind) {
  // We want to test the case where the lambda inside absl::c_find_if evaluates
  // to false for the first projection dimension and true for the second.
  // This happens when the slice belongs to the inner dimension.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 10});
  ShapeTracker tracker(shape);

  // Slice that intersects only with the inner dimension.
  // The first check in c_find_if will be `s % proj_stride == 0`
  // -> `1 % 10 == 0`, which is false.
  // The second check will be `1 % 1 == 0`, which is true.
  auto narrowed_or = tracker.Narrow({1});
  ASSERT_TRUE(narrowed_or.ok());

  ShapeTracker narrowed = narrowed_or.value();
  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{10}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{10}));
}

TEST(ShapeTrackerTest, ZipSimpleContiguous) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker1(shape1);
  ASSERT_TRUE(tracker1.AppendTranspose({1, 0}).ok());

  Shape shape2 = ShapeUtil::MakeShape(F32, {4});
  ShapeTracker tracker2(shape2);

  auto zipped_or = ShapeTracker::Zip({tracker1, tracker2});
  ASSERT_TRUE(zipped_or.ok());
  ShapeTracker zipped = std::move(zipped_or).value();

  EXPECT_EQ(zipped.input_shape().dimensions(), (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(zipped.output_shape().dimensions(),
            (std::vector<int64_t>{3, 2, 4}));
  EXPECT_EQ(zipped.DebugString(/*avoid_combining_reshapes=*/false),
            "[2,3,4] -> T[3,2,4]");
}

TEST(ShapeTrackerTest, ZipWithPadding) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker1(shape1);
  // Force two projections:
  ASSERT_TRUE(tracker1.AppendTranspose({2, 0, 1}).ok());  // output [4, 2, 3]
  ASSERT_TRUE(
      tracker1.AppendReshape({3, 8}).ok());  // Forces copy, output [3, 8]

  Shape shape2 = ShapeUtil::MakeShape(F32, {5});
  ShapeTracker tracker2(shape2);

  auto zipped_or = ShapeTracker::Zip({tracker1, tracker2});
  ASSERT_TRUE(zipped_or.ok());
  ShapeTracker zipped = std::move(zipped_or).value();

  EXPECT_EQ(zipped.input_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4, 5}));
  EXPECT_EQ(zipped.output_shape().dimensions(),
            (std::vector<int64_t>{3, 8, 5}));

  // We should have at least 2 steps representing the joint operations:
  EXPECT_EQ(zipped.DebugString(/*avoid_combining_reshapes=*/false),
            "[2,3,4,5] -> R[6,4,5] -> T[4,6,5] -> R[3,8,5]");
}

TEST(ShapeTrackerTest, ZipWithOneElementShape) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {1});
  ShapeTracker tracker1(shape1);

  Shape shape2 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker2(shape2);
  ASSERT_TRUE(tracker2.AppendTranspose({1, 0}).ok());

  auto zipped_or = ShapeTracker::Zip({tracker1, tracker2});
  ASSERT_TRUE(zipped_or.ok());
  ShapeTracker zipped = std::move(zipped_or).value();

  EXPECT_EQ(zipped.input_shape().dimensions(), (std::vector<int64_t>{1, 2, 3}));
  EXPECT_EQ(zipped.output_shape().dimensions(),
            (std::vector<int64_t>{1, 3, 2}));

  std::vector<ShapeTracker::Step> steps = zipped.GetSteps();
  ASSERT_EQ(steps.size(), 3);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(steps[2].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[2].dimensions, (std::vector<int64_t>{1, 3, 2}));

  EXPECT_EQ(zipped.DebugString(), "[1,2,3] -> T[3,1,2] -> R[1,3,2]");
}

TEST(ShapeTrackerTest, ZipValidationErrors) {
  Shape shape_f32 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker_f32(shape_f32);

  Shape shape_s32 = ShapeUtil::MakeShape(S32, {4});
  ShapeTracker tracker_s32(shape_s32);

  // Empty list:
  EXPECT_FALSE(ShapeTracker::Zip({}).ok());

  // Element type mismatch:
  EXPECT_FALSE(ShapeTracker::Zip({tracker_f32, tracker_s32}).ok());
}

TEST(ShapeTrackerTest, ZipTotalElementsIsOne) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {1});
  ShapeTracker tracker1(shape1);

  Shape shape2 = ShapeUtil::MakeShape(F32, {1, 1});
  ShapeTracker tracker2(shape2);

  auto zipped_or = ShapeTracker::Zip({tracker1, tracker2});
  ASSERT_TRUE(zipped_or.ok());
  ShapeTracker zipped = std::move(zipped_or).value();

  // Since total_elements == 1, it returns early.
  EXPECT_EQ(zipped.input_shape().dimensions(), (std::vector<int64_t>{1, 1, 1}));
  EXPECT_EQ(zipped.output_shape().dimensions(),
            (std::vector<int64_t>{1, 1, 1}));
}

TEST(ShapeTrackerTest, ZipPadProjections) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker1(shape1);
  ASSERT_TRUE(tracker1.AppendTranspose({1, 0}).ok());
  // AppendReshape that forces a new projection.
  ASSERT_TRUE(tracker1.AppendReshape({6}).ok());

  Shape shape2 = ShapeUtil::MakeShape(F32, {4});
  ShapeTracker tracker2(shape2);
  // tracker2 has 1 projection and >1 elements.
  // max_projections will be 2 (from tracker1).
  // tracker2 will be padded with a noop projection.

  auto zipped_or = ShapeTracker::Zip({tracker1, tracker2});
  ASSERT_TRUE(zipped_or.ok());
  ShapeTracker zipped = std::move(zipped_or).value();

  EXPECT_EQ(zipped.input_shape().dimensions(), (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(zipped.output_shape().dimensions(), (std::vector<int64_t>{6, 4}));
}

}  // namespace
}  // namespace xla
