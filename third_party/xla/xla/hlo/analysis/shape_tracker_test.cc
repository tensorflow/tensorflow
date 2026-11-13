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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
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
    EXPECT_EQ(actual[i].broadcast_dimensions, expected[i].broadcast_dimensions)
        << "Step broadcast_dimensions mismatch at index " << i;
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

TEST(ShapeTrackerTest, GetInvertedPreservesInputLayout) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3, 4}, {0, 1, 2});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({4, 6}).ok());

  auto inverted_or = tracker.GetInverted();
  ASSERT_TRUE(inverted_or.ok());
  EXPECT_EQ(inverted_or->output_shape(), shape);
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

TEST(ShapeTrackerTest, AppendBitcastNonFactorizable) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {3, 2});

  ShapeTracker tracker(src_shape);
  ASSERT_TRUE(tracker.AppendBitcast(src_shape, dst_shape).ok());

  ShapeTracker expected(src_shape);
  ASSERT_TRUE(expected.AppendReshape({3, 2}).ok());
  EXPECT_EQ(tracker, expected);

  // Changing the layout between {1,0} and {0,1} reshapes the physical buffer
  // from [2,3] to [3,2] and transposes it back to [2,3], so individual
  // dimensions cannot be mapped or narrowed across the bitcast.
  Shape dst_col_major =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {0, 1});
  ShapeTracker col_major_tracker(src_shape);
  ASSERT_TRUE(col_major_tracker.AppendBitcast(src_shape, dst_col_major).ok());
  EXPECT_EQ(col_major_tracker.DebugString(), "[2,3] -> R[3,2] -> T[2,3]");
  EXPECT_EQ(col_major_tracker.MapInputDimensionsToOutputUnordered({0}),
            std::nullopt);
  EXPECT_THAT(
      col_major_tracker.Narrow({0}),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Slice is incompatible with projection")));
}

TEST(ShapeTrackerTest, AppendBitcastIncompatible) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(src_shape);

  EXPECT_THAT(
      tracker.AppendBitcast(src_shape, ShapeUtil::MakeShape(F32, {2, 4})),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Bitcast must preserve the total number of elements")));
  EXPECT_THAT(
      tracker.AppendBitcast(ShapeUtil::MakeShape(F32, {3, 2}),
                            ShapeUtil::MakeShape(F32, {2, 3})),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Bitcast operand shape does not match current output shape")));
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

TEST(ShapeTrackerTest, GetStepsSingleTransposeMergingDimensions) {
  Shape shape = ShapeUtil::MakeShape(F32, {4, 8, 16, 32});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({0, 2, 3, 1}).ok());

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{0, 2, 3, 1}));
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
  ASSERT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{2, 0, 1}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{1, 3, 2}));

  EXPECT_EQ(tracker.DebugString(), "[1,2,3] -> T[3,1,2] -> R[1,3,2]");
}

TEST(ShapeTrackerTest, GetStepsFallback) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());
  ASSERT_TRUE(tracker.AppendReshape({8, 3}).ok());

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{8, 3}));

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{2, 0, 1}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{8, 3}));

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
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1, 2, 0}));

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

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps(false);

  // The intermediate reshape to [512, 5376] is skipped, and we go directly
  // from [5376, 512] to [1, 512, 5376] via a single reshape.
  ASSERT_EQ(steps.size(), 3);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{512, 5376}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(steps[2].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[2].dimensions, (std::vector<int64_t>{1, 512, 5376}));

  std::vector<ShapeTracker::Step> opt_steps = tracker.GetSteps();
  ASSERT_EQ(opt_steps.size(), 2);
  EXPECT_EQ(opt_steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(opt_steps[0].dimensions, (std::vector<int64_t>{2, 0, 1}));
  EXPECT_EQ(opt_steps[1].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(opt_steps[1].dimensions, (std::vector<int64_t>{1, 512, 5376}));
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
  ASSERT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kTranspose);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{2, 0, 1}));
  EXPECT_EQ(steps[1].type, ShapeTracker::Step::Type::kReshape);
  EXPECT_EQ(steps[1].dimensions, (std::vector<int64_t>{1, 3, 2}));

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

TEST(ShapeTrackerTest, FromSiblingsBasic) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto lca = HloInstruction::CreateParameter(0, shape, "lca");

  auto t1 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), lca.get(), {2, 1, 0});
  auto r1 = HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {12, 2}),
                                          t1.get());

  auto t2 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), lca.get(), {1, 0, 2});
  auto r2 = HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {6, 4}),
                                          t2.get());

  auto tracker_or = ShapeTracker::FromSiblings(r1.get(), r2.get());
  ASSERT_TRUE(tracker_or.ok());
  ShapeTracker tracker = std::move(tracker_or).value();

  EXPECT_EQ(tracker.input_shape().dimensions(), (std::vector<int64_t>{12, 2}));
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{6, 4}));
}

TEST(ShapeTrackerTest, FromSiblingsAncestorAsSource) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto lca = HloInstruction::CreateParameter(0, shape, "lca");

  auto t2 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), lca.get(), {1, 0, 2});
  auto r2 = HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {6, 4}),
                                          t2.get());

  auto tracker_or = ShapeTracker::FromSiblings(lca.get(), r2.get());
  ASSERT_TRUE(tracker_or.ok());
  ShapeTracker tracker = std::move(tracker_or).value();

  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{6, 4}));
}

TEST(ShapeTrackerTest, FromSiblingsAncestorAsDestination) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto lca = HloInstruction::CreateParameter(0, shape, "lca");

  auto t1 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), lca.get(), {2, 1, 0});
  auto r1 = HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {12, 2}),
                                          t1.get());

  auto tracker_or = ShapeTracker::FromSiblings(r1.get(), lca.get());
  ASSERT_TRUE(tracker_or.ok());
  ShapeTracker tracker = std::move(tracker_or).value();

  EXPECT_EQ(tracker.input_shape().dimensions(), (std::vector<int64_t>{12, 2}));
  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));
}

TEST(ShapeTrackerTest, FromSiblingsIdenticalNodes) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto lca = HloInstruction::CreateParameter(0, shape, "lca");

  auto tracker_or = ShapeTracker::FromSiblings(lca.get(), lca.get());
  ASSERT_TRUE(tracker_or.ok());
  ShapeTracker tracker = std::move(tracker_or).value();

  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));
}

TEST(ShapeTrackerTest, FromSiblingsNoCommonAncestor) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto param1 = HloInstruction::CreateParameter(0, shape, "param1");
  auto param2 = HloInstruction::CreateParameter(1, shape, "param2");

  auto t1 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), param1.get(), {2, 1, 0});
  auto t2 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), param2.get(), {1, 0, 2});

  auto tracker_or = ShapeTracker::FromSiblings(t1.get(), t2.get());
  EXPECT_FALSE(tracker_or.ok());
  EXPECT_EQ(tracker_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, FromSiblingsInvalidIntermediateNode) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto lca = HloInstruction::CreateParameter(0, shape, "lca");

  auto t1 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), lca.get(), {2, 1, 0});

  auto dummy = HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {4, 3, 2}), "dummy");
  auto add =
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {4, 3, 2}),
                                   HloOpcode::kAdd, t1.get(), dummy.get());

  auto r1 = HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {12, 2}),
                                          add.get());

  auto t2 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), lca.get(), {1, 0, 2});

  auto tracker_or = ShapeTracker::FromSiblings(r1.get(), t2.get());
  EXPECT_FALSE(tracker_or.ok());
  EXPECT_EQ(tracker_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, FromSiblingsUnsupportedIntermediateNode) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  auto lca = HloInstruction::CreateParameter(0, shape, "lca");

  auto negate =
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, lca.get());
  auto r1 = HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {24}),
                                          negate.get());

  auto t2 = HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), lca.get(), {1, 0, 2});

  auto tracker_or = ShapeTracker::FromSiblings(r1.get(), t2.get());
  EXPECT_FALSE(tracker_or.ok());
  EXPECT_EQ(tracker_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedIdentity) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{0});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}),
            std::vector<int64_t>{1});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1}),
            (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({}),
            std::vector<int64_t>{});
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedTranspose) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({1, 0}).ok());  // output [3, 2]

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{1});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}),
            std::vector<int64_t>{0});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1}),
            (std::vector<int64_t>{0, 1}));
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedUnsortedInput) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());  // output [5, 2, 3]

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1, 0}),
            (std::vector<int64_t>{1, 2}));
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedReshapeSplit) {
  Shape shape = ShapeUtil::MakeShape(F32, {6});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendReshape({2, 3}).ok());

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({}),
            std::vector<int64_t>{});
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedReshapeGlue) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendReshape({6}).ok());

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1}),
            std::vector<int64_t>{0});
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedDegenerate) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 1, 3});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());  // output [3, 1, 2]

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{2});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}),
            std::vector<int64_t>{});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1}),
            std::vector<int64_t>{2});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 2}),
            (std::vector<int64_t>{0, 2}));
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedTrickyCase) {
  Shape shape = ShapeUtil::MakeShape(F32, {3, 14, 5});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({2, 1, 0}).ok());  // [5, 14, 3]
  ASSERT_TRUE(tracker.AppendReshape({2, 15, 7}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({2, 0, 1}).ok());  // [7, 2, 15]

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({2}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1, 2}),
            (std::vector<int64_t>{0, 1, 2}));
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedGluedSwapBack) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 5, 7});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendReshape({2, 3, 7, 5}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({1, 0, 3, 2}).ok());  // [3, 2, 5, 7]
  ASSERT_TRUE(tracker.AppendReshape({3, 2, 7, 5}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({1, 0, 3, 2}).ok());  // [2, 3, 5, 7]

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1}),
            (std::vector<int64_t>{0, 1}));
}

TEST(ShapeTrackerTest,
     MapInputDimensionsToOutputUnorderedScalarOrDegenerateInput) {
  Shape shape = ShapeUtil::MakeShape(F32, {1});
  ShapeTracker tracker(shape);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({}),
            std::vector<int64_t>{});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{});

  Shape shape2 = ShapeUtil::MakeShape(F32, {1, 1});
  ShapeTracker tracker2(shape2);
  EXPECT_EQ(tracker2.MapInputDimensionsToOutputUnordered({0, 1}),
            std::vector<int64_t>{});
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedIncompatibleReshape) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({1, 0, 2}).ok());  // [3, 2, 4]
  ASSERT_TRUE(tracker.AppendReshape({2, 12}).ok());

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({2}), std::nullopt);
}

TEST(ShapeTrackerTest, MapInputDimensionsToOutputUnorderedFractionalReshape) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendReshape({3, 2}).ok());

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1}),
            (std::vector<int64_t>{0, 1}));
}

TEST(ShapeTrackerTest, ZipInvertCommutativity) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker t1(shape1);
  ASSERT_TRUE(t1.AppendTranspose({1, 0}).ok());

  Shape shape2 = ShapeUtil::MakeShape(F32, {4, 5});
  ShapeTracker t2(shape2);
  ASSERT_TRUE(t2.AppendTranspose({1, 0}).ok());

  auto zipped = ShapeTracker::Zip({t1, t2});
  ASSERT_TRUE(zipped.ok());
  auto zipped_val = std::move(zipped).value();
  auto zipped_inverted = zipped_val.GetInverted();
  ASSERT_TRUE(zipped_inverted.ok());
  auto zipped_inverted_val = std::move(zipped_inverted).value();

  auto t1_inv = t1.GetInverted();
  ASSERT_TRUE(t1_inv.ok());
  auto t2_inv = t2.GetInverted();
  ASSERT_TRUE(t2_inv.ok());
  auto inverted_zipped = ShapeTracker::Zip({t1_inv.value(), t2_inv.value()});
  ASSERT_TRUE(inverted_zipped.ok());
  auto inverted_zipped_val = std::move(inverted_zipped).value();

  EXPECT_EQ(zipped_inverted_val.input_shape(),
            inverted_zipped_val.input_shape());
  EXPECT_EQ(zipped_inverted_val.output_shape(),
            inverted_zipped_val.output_shape());
  EXPECT_EQ(zipped_inverted_val.DebugString(),
            inverted_zipped_val.DebugString());
  ExpectStepsEqual(zipped_inverted_val.GetSteps(),
                   inverted_zipped_val.GetSteps());
}

TEST(ShapeTrackerTest, ZipInvertCommutativityWithReshape) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker t1(shape1);
  ASSERT_TRUE(t1.AppendReshape({6}).ok());

  Shape shape2 = ShapeUtil::MakeShape(F32, {4});
  ShapeTracker t2(shape2);

  auto zipped = ShapeTracker::Zip({t1, t2});
  ASSERT_TRUE(zipped.ok());
  auto zipped_val = std::move(zipped).value();
  auto zipped_inverted = zipped_val.GetInverted();
  ASSERT_TRUE(zipped_inverted.ok());
  auto zipped_inverted_val = std::move(zipped_inverted).value();

  auto t1_inv = t1.GetInverted();
  ASSERT_TRUE(t1_inv.ok());
  auto t2_inv = t2.GetInverted();
  ASSERT_TRUE(t2_inv.ok());
  auto inverted_zipped = ShapeTracker::Zip({t1_inv.value(), t2_inv.value()});
  ASSERT_TRUE(inverted_zipped.ok());
  auto inverted_zipped_val = std::move(inverted_zipped).value();

  EXPECT_EQ(zipped_inverted_val.input_shape(),
            inverted_zipped_val.input_shape());
  EXPECT_EQ(zipped_inverted_val.output_shape(),
            inverted_zipped_val.output_shape());
  EXPECT_EQ(zipped_inverted_val.DebugString(),
            inverted_zipped_val.DebugString());
  ExpectStepsEqual(zipped_inverted_val.GetSteps(),
                   inverted_zipped_val.GetSteps());
}

TEST(ShapeTrackerMapsToOneStrideTest, NoChangeIsContiguous) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  EXPECT_TRUE(tracker.MapsToOneStride({0, 1, 2}));
  EXPECT_TRUE(tracker.MapsToOneStride({1, 2}));
  EXPECT_TRUE(tracker.MapsToOneStride({0}));
}

TEST(ShapeTrackerMapsToOneStrideTest, RealSwapIsNonContiguous) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({0, 2, 1}).ok());
  EXPECT_FALSE(tracker.MapsToOneStride({1, 2}));
  EXPECT_TRUE(tracker.MapsToOneStride({0}));
}

TEST(ShapeTrackerMapsToOneStrideTest, IdentityTransposeIsContiguous) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({0, 1, 2}).ok());
  EXPECT_TRUE(tracker.MapsToOneStride({1, 2}));
}

TEST(ShapeTrackerMapsToOneStrideTest, Size1TransposeIsContiguous) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 1, 4});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({0, 2, 1}).ok());
  EXPECT_TRUE(tracker.MapsToOneStride({1, 2}));
}

TEST(ShapeTrackerMapsToOneStrideTest, SwapWithReshapeIsNonContiguous) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendReshape({2, 2, 2}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({0, 2, 1}).ok());
  EXPECT_FALSE(tracker.MapsToOneStride({1}));
}

TEST(ShapeTrackerMapsToOneStrideTest, NoOpTransposeWithReshapeIsContiguous) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendReshape({2, 2, 1}).ok());
  ASSERT_TRUE(tracker.AppendTranspose({0, 2, 1}).ok());
  EXPECT_TRUE(tracker.MapsToOneStride({1}));
}

TEST(ShapeTrackerMapsToOneStrideTest, ContiguousTransposeReshapeIsContiguous) {
  Shape shape = ShapeUtil::MakeShape(F32, {4, 2, 3, 2});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({3, 1, 2, 0}).ok());
  ASSERT_TRUE(tracker.AppendReshape({2, 6, 4}).ok());
  EXPECT_TRUE(tracker.MapsToOneStride({1, 2}));
}

TEST(ShapeTrackerMapsToOneStrideTest, AllowSwapsSwappedIsContiguous) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  ASSERT_TRUE(tracker.AppendTranspose({0, 2, 1}).ok());
  EXPECT_FALSE(tracker.MapsToOneStride({1, 2}, /*allow_swaps=*/false));
  EXPECT_TRUE(tracker.MapsToOneStride({1, 2}, /*allow_swaps=*/true));
}

TEST(ShapeTrackerBroadcastReduceTest, AppendBroadcastSimple) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendBroadcast({4, 2, 5, 3}, {1, 3}));

  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{4, 2, 5, 3}));
  EXPECT_EQ(tracker.DebugString(true), "[2,3] -> B[4,2,5,3]");
  EXPECT_EQ(tracker.DebugString(false), "[2,3] -> B[4,2,5,3]");

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kBroadcast);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{4, 2, 5, 3}));
  EXPECT_EQ(steps[0].broadcast_dimensions, (std::vector<int64_t>{1, 3}));
}

TEST(ShapeTrackerBroadcastReduceTest, AppendBroadcastScalar) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendBroadcast({2, 3}, {}));

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tracker.DebugString(true), "[] -> B[6] -> R[2,3]");
}

TEST(ShapeTrackerBroadcastReduceTest, AppendBroadcastValidationErrors) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  // Non-positive dimension.
  EXPECT_THAT(tracker.AppendBroadcast({0, 2, 3}, {1, 2}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  // broadcast_dimensions size mismatch.
  EXPECT_THAT(tracker.AppendBroadcast({4, 2, 3}, {1}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  // Out of bounds index.
  EXPECT_THAT(tracker.AppendBroadcast({4, 2, 3}, {1, 3}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(tracker.AppendBroadcast({4, 2, 3}, {-1, 2}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  // Duplicate broadcast_dimensions.
  EXPECT_THAT(tracker.AppendBroadcast({2, 2, 3}, {0, 0}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  // Extent mismatch.
  EXPECT_THAT(tracker.AppendBroadcast({4, 5, 3}, {1, 2}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ShapeTrackerBroadcastReduceTest, PrependBroadcastSimpleAndUnsorted) {
  Shape shape = ShapeUtil::MakeShape(F32, {4, 2, 5, 3});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.PrependBroadcast({1, 3}));

  EXPECT_EQ(tracker.input_shape().dimensions(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tracker.output_shape().dimensions(),
            (std::vector<int64_t>{4, 2, 5, 3}));
  EXPECT_EQ(tracker.DebugString(true), "[2,3] -> B[4,2,5,3]");

  // Unsorted broadcast_dimensions: operand dim 0 -> output dim 3 (size 3),
  // operand dim 1 -> output dim 1 (size 2).
  ShapeTracker tracker_unsorted(shape);
  ASSERT_OK(tracker_unsorted.PrependBroadcast({3, 1}));
  EXPECT_EQ(tracker_unsorted.input_shape().dimensions(),
            (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(tracker_unsorted.output_shape().dimensions(),
            (std::vector<int64_t>{4, 2, 5, 3}));
  EXPECT_EQ(tracker_unsorted.DebugString(true),
            "[3,2] -> T[2,3] -> B[4,2,5,3]");
}

TEST(ShapeTrackerBroadcastReduceTest, PrependBroadcastValidationErrors) {
  Shape shape = ShapeUtil::MakeShape(F32, {4, 2, 3});
  ShapeTracker tracker(shape);

  EXPECT_THAT(tracker.PrependBroadcast({-1}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(tracker.PrependBroadcast({3}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(tracker.PrependBroadcast({1, 1}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ShapeTrackerBroadcastReduceTest, AppendAndPrependInstructionBroadcast) {
  Shape src_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {4, 2, 3});
  auto param = HloInstruction::CreateParameter(0, src_shape, "param");
  auto bcast = HloInstruction::CreateBroadcast(dst_shape, param.get(), {1, 2});

  ShapeTracker tracker_append(src_shape);
  ASSERT_OK(tracker_append.AppendInstruction(bcast.get()));
  EXPECT_EQ(tracker_append.output_shape().dimensions(),
            (std::vector<int64_t>{4, 2, 3}));
  EXPECT_EQ(tracker_append.DebugString(true), "[2,3] -> B[4,2,3]");

  ShapeTracker tracker_prepend(dst_shape);
  ASSERT_OK(tracker_prepend.PrependInstruction(bcast.get()));
  EXPECT_EQ(tracker_prepend.input_shape().dimensions(),
            (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tracker_prepend.DebugString(true), "[2,3] -> B[4,2,3]");

  EXPECT_EQ(tracker_append, tracker_prepend);
}

TEST(ShapeTrackerBroadcastReduceTest, AppendReduceSimple) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendReduce({1}));

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(tracker.DebugString(true), "[2,3,4] -> Red[2,4]");

  std::vector<ShapeTracker::Step> steps = tracker.GetSteps();
  ASSERT_EQ(steps.size(), 1);
  EXPECT_EQ(steps[0].type, ShapeTracker::Step::Type::kReduce);
  EXPECT_EQ(steps[0].dimensions, (std::vector<int64_t>{1}));
}

TEST(ShapeTrackerBroadcastReduceTest, AppendReduceToScalar) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendReduce({0, 1}));

  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{}));
  EXPECT_EQ(tracker.DebugString(true), "[2,3] -> Red[]");
  EXPECT_EQ(tracker.DebugString(false), "[2,3] -> R[6] -> Red[]");
}

TEST(ShapeTrackerBroadcastReduceTest, AppendReduceAdjacentMerging) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4, 5});
  ShapeTracker tracker1(shape);
  ASSERT_OK(tracker1.AppendReduce({1}));  // [2, 4, 5]
  ASSERT_OK(tracker1.AppendReduce({1}));  // [2, 5]

  ShapeTracker tracker2(shape);
  ASSERT_OK(tracker2.AppendReduce({1, 2}));  // [2, 5]

  EXPECT_EQ(tracker1, tracker2);
  EXPECT_EQ(tracker1.DebugString(true), "[2,3,4,5] -> Red[2,5]");
}

TEST(ShapeTrackerBroadcastReduceTest, AppendReduceValidationErrors) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);

  EXPECT_THAT(tracker.AppendReduce({-1}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(tracker.AppendReduce({3}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(tracker.AppendReduce({1, 1}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ShapeTrackerBroadcastReduceTest, PrependReduceSimpleAndValidation) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.PrependReduce({2, 3, 4}, {1}));
  EXPECT_EQ(tracker.input_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(tracker.output_shape().dimensions(), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(tracker.DebugString(true), "[2,3,4] -> Red[2,4]");

  // Validation errors:
  EXPECT_THAT(tracker.PrependReduce({0, 2, 3, 4}, {0}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(tracker.PrependReduce({5, 2, 3, 4}, {4}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(tracker.PrependReduce({5, 2, 3, 4}, {0, 0}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(tracker.PrependReduce({2, 3, 5}, {1}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ShapeTrackerBroadcastReduceTest, BroadcastThenReduceCancelsToIdentity) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendBroadcast({4, 2, 5, 3}, {1, 3}));
  ASSERT_OK(tracker.AppendReduce({0, 2}));

  EXPECT_EQ(tracker, ShapeTracker(shape));
  EXPECT_TRUE(tracker.GetSteps().empty());
  EXPECT_EQ(tracker.DebugString(), "[2,3]");
}

TEST(ShapeTrackerBroadcastReduceTest, BroadcastReshapePartialReduce) {
  Shape shape = ShapeUtil::MakeShape(F32, {2});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendBroadcast({2, 6}, {0}));
  ASSERT_OK(tracker.AppendReshape({2, 2, 3}));
  ASSERT_OK(tracker.AppendReduce({1}));

  ShapeTracker expected(shape);
  ASSERT_OK(expected.AppendBroadcast({2, 3}, {0}));

  EXPECT_EQ(tracker, expected);
  EXPECT_EQ(tracker.DebugString(true), "[2] -> B[2,3]");
}

TEST(ShapeTrackerBroadcastReduceTest, BroadcastThenTransposeSimplifies) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendBroadcast({4, 2, 3}, {1, 2}));
  ASSERT_OK(tracker.AppendTranspose({1, 0, 2}));

  ShapeTracker expected(shape);
  ASSERT_OK(expected.AppendBroadcast({2, 4, 3}, {0, 2}));

  EXPECT_EQ(tracker, expected);
  EXPECT_EQ(tracker.DebugString(true), "[2,3] -> B[2,4,3]");
}

TEST(ShapeTrackerBroadcastReduceTest, BroadcastThenTransposeKeptDims) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendBroadcast({4, 2, 3}, {1, 2}));
  ASSERT_OK(tracker.AppendTranspose({2, 0, 1}));

  EXPECT_EQ(tracker.DebugString(true), "[2,3] -> T[3,2] -> B[3,4,2]");
}

TEST(ShapeTrackerBroadcastReduceTest, ReduceThenBroadcastDoesNotCancel) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);

  ASSERT_OK(tracker.AppendReduce({1}));
  ASSERT_OK(tracker.AppendBroadcast({2, 3}, {0}));

  EXPECT_NE(tracker, ShapeTracker(shape));
  EXPECT_EQ(tracker.DebugString(true), "[2,3] -> Red[2] -> B[2,3]");
}

TEST(ShapeTrackerBroadcastReduceTest, DoubleInversionIdentity) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4, 5});
  ShapeTracker tracker(shape);

  // Reduce dim 1 (size 3) -> [2, 4, 5]
  ASSERT_OK(tracker.AppendReduce({1}));
  // Transpose {2, 0, 1} -> [5, 2, 4]
  ASSERT_OK(tracker.AppendTranspose({2, 0, 1}));
  // Broadcast to [7, 5, 2, 11, 4] with {1, 2, 4}
  ASSERT_OK(tracker.AppendBroadcast({7, 5, 2, 11, 4}, {1, 2, 4}));
  // Reshape to [35, 22, 4]
  ASSERT_OK(tracker.AppendReshape({35, 22, 4}));

  ASSERT_OK_AND_ASSIGN(ShapeTracker inverted, tracker.GetInverted());
  ASSERT_OK_AND_ASSIGN(ShapeTracker double_inverted, inverted.GetInverted());

  EXPECT_EQ(double_inverted, tracker);
  EXPECT_EQ(double_inverted.DebugString(), tracker.DebugString());
  ExpectStepsEqual(double_inverted.GetSteps(), tracker.GetSteps());
}

TEST(ShapeTrackerBroadcastReduceTest,
     DoubleInversionBoundaryReducedDimensions) {
  // Test reducing major-most (dim 0) and minor-most (dim 2) dimensions.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendReduce({0, 2}));  // [3]
  ASSERT_OK(tracker.AppendBroadcast({5, 3, 7}, {1}));

  ASSERT_OK_AND_ASSIGN(ShapeTracker inverted, tracker.GetInverted());
  ASSERT_OK_AND_ASSIGN(ShapeTracker double_inverted, inverted.GetInverted());

  EXPECT_EQ(double_inverted, tracker);
  ExpectStepsEqual(double_inverted.GetSteps(), tracker.GetSteps());
}

TEST(ShapeTrackerBroadcastReduceTest, ConcatenateBroadcastWithInvertedCancels) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendTranspose({2, 0, 1}));  // [4, 2, 3]
  ASSERT_OK(tracker.AppendBroadcast({5, 4, 7, 2, 3}, {1, 3, 4}));
  ASSERT_OK(tracker.AppendReshape({20, 14, 3}));

  ASSERT_OK_AND_ASSIGN(ShapeTracker inverted, tracker.GetInverted());
  ASSERT_OK(tracker.ConcatenateFrom(inverted));

  EXPECT_EQ(tracker, ShapeTracker(shape));
  EXPECT_TRUE(tracker.GetSteps().empty());
}

TEST(ShapeTrackerBroadcastReduceTest, ToInstructionChainBroadcastAndReduce) {
  auto module = std::make_unique<HloModule>("module", HloModuleConfig());
  HloComputation::Builder builder("comp");

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  module->AddEntryComputation(builder.Build());

  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendBroadcast({2, 3, 5, 4}, {0, 1, 3}));

  // With avoid_combining_reshapes = true, should emit a single kBroadcast!
  {
    ASSERT_OK_AND_ASSIGN(HloInstruction * root,
                         tracker.ToInstructionChain(param, true));
    EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
    EXPECT_EQ(root->shape().dimensions(), (std::vector<int64_t>{2, 3, 5, 4}));
    EXPECT_EQ(root->dimensions(), (std::vector<int64_t>{0, 1, 3}));
    EXPECT_EQ(root->operand(0), param);
  }

  // With avoid_combining_reshapes = false, emits Reshape -> Broadcast ->
  // Reshape.
  {
    ASSERT_OK_AND_ASSIGN(HloInstruction * root,
                         tracker.ToInstructionChain(param, false));
    EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
    EXPECT_EQ(root->shape().dimensions(), (std::vector<int64_t>{2, 3, 5, 4}));
    const HloInstruction* bcast = root->operand(0);
    EXPECT_EQ(bcast->opcode(), HloOpcode::kBroadcast);
    EXPECT_EQ(bcast->shape().dimensions(), (std::vector<int64_t>{6, 5, 4}));
    EXPECT_EQ(bcast->dimensions(), (std::vector<int64_t>{0, 2}));
    const HloInstruction* reshape_in = bcast->operand(0);
    EXPECT_EQ(reshape_in->opcode(), HloOpcode::kReshape);
    EXPECT_EQ(reshape_in->operand(0), param);
  }

  // If an uncancelled reduction remains, ToInstructionChain returns
  // InvalidArgumentError.
  ShapeTracker reduce_tracker(shape);
  ASSERT_OK(reduce_tracker.AppendReduce({1}));
  EXPECT_THAT(reduce_tracker.ToInstructionChain(param),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ShapeTrackerBroadcastReduceTest, FromSiblingsWithBroadcast) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  auto lca = HloInstruction::CreateParameter(0, shape, "lca");

  auto b1 = HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {4, 2, 3}), lca.get(), {1, 2});
  auto t2 = HloInstruction::CreateTranspose(ShapeUtil::MakeShape(F32, {3, 2}),
                                            lca.get(), {1, 0});

  // From lca to b1: pure broadcast.
  {
    ASSERT_OK_AND_ASSIGN(ShapeTracker tracker,
                         ShapeTracker::FromSiblings(lca.get(), b1.get()));
    EXPECT_EQ(tracker.DebugString(true), "[2,3] -> B[4,2,3]");
  }

  // From b1 to t2: reduces broadcasted dim 0, then transposes [2, 3] -> [3, 2].
  {
    ASSERT_OK_AND_ASSIGN(ShapeTracker tracker,
                         ShapeTracker::FromSiblings(b1.get(), t2.get()));
    EXPECT_EQ(tracker.DebugString(true), "[4,2,3] -> Red[2,3] -> T[3,2]");
    EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}),
              std::vector<int64_t>{1});
    EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({2}),
              std::vector<int64_t>{0});
    EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
              std::vector<int64_t>{});
  }
}

TEST(ShapeTrackerBroadcastReduceTest, NarrowAndMapDimensionsWithBroadcast) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendBroadcast({4, 2, 5, 3}, {1, 3}));

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{1});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}),
            std::vector<int64_t>{3});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1}),
            (std::vector<int64_t>{1, 3}));
  // Dims 1 and 3 in output are separated by non-degenerate broadcast dim 2
  // (size 5), so {0, 1} does not map to a single stride in output.
  EXPECT_FALSE(tracker.MapsToOneStride({0, 1}));
  EXPECT_TRUE(tracker.MapsToOneStride({0}));
  EXPECT_TRUE(tracker.MapsToOneStride({1}));

  ASSERT_OK_AND_ASSIGN(ShapeTracker narrowed, tracker.Narrow({0}));
  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{2}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{2}));

  // If broadcast dim 0 (size 4) is glued with kept dim 1 (size 2) into size
  // 8, then input dim 0 only partially covers output dim 0, so mapping fails.
  ASSERT_OK(tracker.AppendReshape({8, 5, 3}));
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}),
            std::vector<int64_t>{2});
}

TEST(ShapeTrackerBroadcastReduceTest, NarrowAndMapDimensionsWithReduce) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendReduce({1}));

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{0});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({2}),
            std::vector<int64_t>{1});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 2}),
            (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}),
            std::vector<int64_t>{});
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0, 1}),
            std::vector<int64_t>{0});

  // Since dim 1 was reduced, dims {0, 2} are now adjacent in the output!
  EXPECT_TRUE(tracker.MapsToOneStride({0, 2}));
  EXPECT_TRUE(tracker.MapsToOneStride({1}));
  EXPECT_TRUE(tracker.MapsToOneStride({0, 1}));

  ASSERT_OK_AND_ASSIGN(ShapeTracker narrowed_02, tracker.Narrow({0, 2}));
  EXPECT_EQ(narrowed_02.input_shape().dimensions(),
            (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(narrowed_02.output_shape().dimensions(),
            (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(narrowed_02.DebugString(true), "[2,4]");

  ASSERT_OK_AND_ASSIGN(ShapeTracker narrowed_01, tracker.Narrow({0, 1}));
  EXPECT_EQ(narrowed_01.input_shape().dimensions(),
            (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(narrowed_01.output_shape().dimensions(), (std::vector<int64_t>{2}));
  EXPECT_EQ(narrowed_01.DebugString(true), "[2,3] -> Red[2]");

  ASSERT_OK_AND_ASSIGN(ShapeTracker narrowed_1, tracker.Narrow({1}));
  EXPECT_EQ(narrowed_1.input_shape().dimensions(), (std::vector<int64_t>{3}));
  EXPECT_EQ(narrowed_1.output_shape().dimensions(), (std::vector<int64_t>{}));
  EXPECT_EQ(narrowed_1.DebugString(true), "[3] -> Red[]");
}

TEST(ShapeTrackerBroadcastReduceTest,
     SubchannelDequantizationInvertedTrackerMapping) {
  // s: [2, 8, 64] -> T[2, 64, 8] -> B[2, 64, 8, 256] -> R[2, 64, 2048]
  // inverted: [2, 64, 2048] -> R[2, 64, 8, 256] -> Red[2, 64, 8] -> T[2, 8, 64]
  Shape shape = ShapeUtil::MakeShape(F32, {2, 8, 64});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendTranspose({0, 2, 1}));
  ASSERT_OK(tracker.AppendBroadcast({2, 64, 8, 256}, {0, 1, 2}));
  ASSERT_OK(tracker.AppendReshape({2, 64, 2048}));
  ASSERT_OK_AND_ASSIGN(ShapeTracker inverted, tracker.GetInverted());

  EXPECT_EQ(inverted.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{0});
  EXPECT_EQ(inverted.MapInputDimensionsToOutputUnordered({1}),
            std::vector<int64_t>{2});
  EXPECT_EQ(inverted.MapInputDimensionsToOutputUnordered({2}),
            std::vector<int64_t>{1});
  EXPECT_TRUE(inverted.MapsToOneStride({0}));
  EXPECT_TRUE(inverted.MapsToOneStride({1}));
  EXPECT_TRUE(inverted.MapsToOneStride({2}));

  ASSERT_OK_AND_ASSIGN(ShapeTracker narrowed_2, inverted.Narrow({2}));
  EXPECT_EQ(narrowed_2.DebugString(true), "[2048] -> R[8,256] -> Red[8]");
}

TEST(ShapeTrackerBroadcastReduceTest,
     SplitContractingDimensionAcrossBroadcast) {
  // s: [2, 2, 64, 4] -> T[2, 64, 4, 2] -> B[2, 64, 4, 2, 256] -> R[2, 64, 2048]
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2, 64, 4});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendTranspose({0, 2, 3, 1}));
  ASSERT_OK(tracker.AppendBroadcast({2, 64, 4, 2, 256}, {0, 1, 2, 3}));
  ASSERT_OK(tracker.AppendReshape({2, 64, 2048}));
  ASSERT_OK_AND_ASSIGN(ShapeTracker inverted, tracker.GetInverted());

  EXPECT_EQ(inverted.MapInputDimensionsToOutputUnordered({2}),
            (std::vector<int64_t>{1, 3}));
  EXPECT_FALSE(inverted.MapsToOneStride({2}));
}

TEST(ShapeTrackerBroadcastReduceTest, SwappedSubDimensionsAcrossBroadcast) {
  // s: [2, 64, 2, 4] -> T[2, 64, 4, 2] -> B[2, 64, 4, 2, 256] -> R[2, 64, 2048]
  Shape shape = ShapeUtil::MakeShape(F32, {2, 64, 2, 4});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendTranspose({0, 1, 3, 2}));
  ASSERT_OK(tracker.AppendBroadcast({2, 64, 4, 2, 256}, {0, 1, 2, 3}));
  ASSERT_OK(tracker.AppendReshape({2, 64, 2048}));
  ASSERT_OK_AND_ASSIGN(ShapeTracker inverted, tracker.GetInverted());

  EXPECT_EQ(inverted.MapInputDimensionsToOutputUnordered({2}),
            (std::vector<int64_t>{2, 3}));
  EXPECT_FALSE(inverted.MapsToOneStride({2}, /*allow_swaps=*/false));
  EXPECT_TRUE(inverted.MapsToOneStride({2}, /*allow_swaps=*/true));
}

TEST(ShapeTrackerBroadcastReduceTest, IncompatibleReshapeAcrossReduction) {
  Shape shape = ShapeUtil::MakeShape(F32, {6, 4});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendReshape({4, 6}));
  ASSERT_OK(tracker.AppendReduce({0}));

  // Input dim 0 ([4, 24)) straddles the kept ([1, 6)) and reduced ([6, 24))
  // regions incompatibly.
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}), std::nullopt);
  EXPECT_THAT(tracker.Narrow({0}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  // Input dim 1 ([1, 4)) only partially covers the output dimension ([1, 6)).
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}), std::nullopt);
}

TEST(ShapeTrackerBroadcastReduceTest, ZipWithBroadcastAndReduce) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {2});
  ShapeTracker t1(shape1);
  ASSERT_OK(t1.AppendBroadcast({2, 3}, {0}));

  Shape shape2 = ShapeUtil::MakeShape(F32, {4, 5});
  ShapeTracker t2(shape2);
  ASSERT_OK(t2.AppendReduce({0}));

  ASSERT_OK_AND_ASSIGN(ShapeTracker zipped, ShapeTracker::Zip({t1, t2}));
  EXPECT_EQ(zipped.input_shape().dimensions(), (std::vector<int64_t>{2, 4, 5}));
  EXPECT_EQ(zipped.output_shape().dimensions(),
            (std::vector<int64_t>{2, 3, 5}));
  EXPECT_EQ(zipped.DebugString(true), "[2,4,5] -> Red[2,5] -> B[2,3,5]");

  // Verify Zip and Invert commute!
  ASSERT_OK_AND_ASSIGN(ShapeTracker zipped_inv, zipped.GetInverted());
  ASSERT_OK_AND_ASSIGN(ShapeTracker t1_inv, t1.GetInverted());
  ASSERT_OK_AND_ASSIGN(ShapeTracker t2_inv, t2.GetInverted());
  ASSERT_OK_AND_ASSIGN(ShapeTracker inv_zipped,
                       ShapeTracker::Zip({t1_inv, t2_inv}));

  EXPECT_EQ(zipped_inv, inv_zipped);
  EXPECT_EQ(zipped_inv.DebugString(), inv_zipped.DebugString());
  ExpectStepsEqual(zipped_inv.GetSteps(), inv_zipped.GetSteps());
}

TEST(ShapeTrackerBroadcastReduceTest,
     MultiProjectionBroadcastPreservesUnrelatedOuterDim) {
  // Input dim 0 (size 2) is kept across a two-projection chain where
  // projections_[0] broadcasts a middle dimension of size 4 and projections_[1]
  // reshapes/transposes the inner dimensions [4, 3].
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendBroadcast({2, 4, 3}, {0, 2}));
  ASSERT_OK(tracker.AppendReshape({2, 3, 4}));
  ASSERT_OK(tracker.AppendTranspose({0, 2, 1}));

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{0});
  EXPECT_TRUE(tracker.MapsToOneStride({0}));
  ASSERT_OK_AND_ASSIGN(ShapeTracker narrowed, tracker.Narrow({0}));
  EXPECT_EQ(narrowed.input_shape().dimensions(), (std::vector<int64_t>{2}));
  EXPECT_EQ(narrowed.output_shape().dimensions(), (std::vector<int64_t>{2}));
}

TEST(ShapeTrackerBroadcastReduceTest,
     MultiProjectionBroadcastDoesNotMisMapToSameSizeBroadcastDim) {
  // Broadcast dim (output dim 1, size 2) has the same extent as kept input
  // dim 0 (output dim 0, size 2); input dim 0 must map to output dim 0, not 1.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 5, 7});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendTranspose({0, 2, 1}));
  ASSERT_OK(tracker.AppendBroadcast({2, 2, 7, 5}, {0, 2, 3}));
  ASSERT_OK(tracker.AppendReshape({2, 2, 5, 7}));
  ASSERT_OK(tracker.AppendTranspose({0, 1, 3, 2}));

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}),
            std::vector<int64_t>{0});
}

TEST(ShapeTrackerBroadcastReduceTest,
     MultiProjectionBroadcastMixedWithReshapeCannotMap) {
  // Reshape mixes kept input dim with broadcasted dim across two
  // projections: input dim 0 no longer maps to a clean subset of output dims.
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendBroadcast({4, 3}, {0}));
  ASSERT_OK(tracker.AppendReshape({3, 4}));
  ASSERT_OK(tracker.AppendTranspose({1, 0}));

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}), std::nullopt);
  EXPECT_FALSE(tracker.MapsToOneStride({0}));
  EXPECT_THAT(tracker.Narrow({0}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ShapeTrackerBroadcastReduceTest,
     MultiProjectionTransposeMixedWithReshapeCannotMap) {
  // Two-projection transpose-reshape-transpose chain without broadcast where
  // the middle reshape mixes the transposed dimensions.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendTranspose({1, 0}));
  ASSERT_OK(tracker.AppendReshape({2, 3}));
  ASSERT_OK(tracker.AppendTranspose({1, 0}));

  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({0}), std::nullopt);
  EXPECT_EQ(tracker.MapInputDimensionsToOutputUnordered({1}), std::nullopt);
  EXPECT_FALSE(tracker.MapsToOneStride({0}));
  EXPECT_FALSE(tracker.MapsToOneStride({1}));
}

TEST(ShapeTrackerBroadcastReduceTest,
     ToInstructionChainReduceDoesNotMutateComputationOnError) {
  auto module = std::make_unique<HloModule>("module", HloModuleConfig());
  HloComputation::Builder builder("comp");
  Shape shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ShapeTracker reduce_tracker(shape);
  ASSERT_OK(reduce_tracker.AppendReshape({2, 3}));
  ASSERT_OK(reduce_tracker.AppendReduce({0}));

  const int64_t initial_count = computation->instruction_count();
  EXPECT_THAT(reduce_tracker.ToInstructionChain(param),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(computation->instruction_count(), initial_count);
}

TEST(ShapeTrackerBroadcastReduceTest,
     NarrowMultiProjectionMergesAndFoldsWhenDroppedDimSeparatedSegments) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3, 5, 7});
  ShapeTracker tracker(shape);
  ASSERT_OK(tracker.AppendTranspose({1, 0, 2, 3}));  // [3, 2, 5, 7]
  ASSERT_OK(tracker.AppendReshape({2, 3, 5, 7}));    // [2, 3, 5, 7]
  ASSERT_OK(tracker.AppendTranspose({0, 2, 1, 3}));  // [2, 5, 3, 7]

  // Dropping input dim 2 (size 5) makes the second transpose an identity on the
  // remaining dimensions [2, 3, 7], so the narrowed tracker should fold to a
  // single projection matching `[2, 3, 7] -> T[3, 2, 7] -> R[2, 3, 7]`.
  ASSERT_OK_AND_ASSIGN(ShapeTracker narrowed, tracker.Narrow({0, 1, 3}));

  Shape expected_shape = ShapeUtil::MakeShape(F32, {2, 3, 7});
  ShapeTracker expected(expected_shape);
  ASSERT_OK(expected.AppendTranspose({1, 0, 2}));
  ASSERT_OK(expected.AppendReshape({2, 3, 7}));

  EXPECT_EQ(narrowed, expected);
}

// (stride, extent) segments of every output dimension.
using DimensionSegments = std::vector<std::vector<std::pair<int64_t, int64_t>>>;

std::optional<DimensionSegments> GetDimensionSegments(
    const ShapeTracker& tracker) {
  std::optional<std::vector<ShapeTracker::BufferView>> views =
      tracker.TryGetStridedView();
  if (!views.has_value()) {
    return std::nullopt;
  }
  DimensionSegments result;
  for (const ShapeTracker::BufferView& view : *views) {
    std::vector<std::pair<int64_t, int64_t>>& segments = result.emplace_back();
    for (int64_t i = 0; i < view.strides().size(); ++i) {
      segments.push_back({view.strides()[i], view.extents()[i]});
    }
  }
  return result;
}

TEST(ShapeTrackerTryGetStridedViewTest, DegenerateDimensionsAreEmpty) {
  ShapeTracker tracker(ShapeUtil::MakeShape(F32, {2, 3}));
  ASSERT_OK(tracker.AppendReshape({2, 1, 3}));
  EXPECT_EQ(GetDimensionSegments(tracker),
            DimensionSegments({{{3, 2}}, {}, {{1, 3}}}));
}

TEST(ShapeTrackerTryGetStridedViewTest, BroadcastHasZeroStrides) {
  ShapeTracker tracker(ShapeUtil::MakeShape(F32, {48}));
  ASSERT_OK(tracker.AppendBroadcast({4, 16, 48}, {2}));
  ASSERT_OK(tracker.AppendReshape({64, 48}));
  EXPECT_EQ(GetDimensionSegments(tracker),
            DimensionSegments({{{0, 64}}, {{1, 48}}}));
}

TEST(ShapeTrackerTryGetStridedViewTest, CopyIsNotAStridedView) {
  ShapeTracker tracker(ShapeUtil::MakeShape(F32, {2, 3}));
  ASSERT_OK(tracker.AppendTranspose({1, 0}));
  ASSERT_OK(tracker.AppendReshape({6}));
  // Merging the transposed dimensions is still a strided view...
  EXPECT_EQ(GetDimensionSegments(tracker),
            DimensionSegments({{{1, 3}, {3, 2}}}));
  // ...but transposing a part of the merged dimension requires a copy.
  ASSERT_OK(tracker.AppendReshape({2, 3}));
  ASSERT_OK(tracker.AppendTranspose({1, 0}));
  EXPECT_EQ(GetDimensionSegments(tracker), std::nullopt);
}

TEST(ShapeTrackerTryGetStridedViewTest, ReductionIsNotAStridedView) {
  ShapeTracker tracker(ShapeUtil::MakeShape(F32, {4, 8}));
  ASSERT_OK(tracker.AppendReduce({0}));
  EXPECT_EQ(GetDimensionSegments(tracker), std::nullopt);
}

TEST(ShapeTrackerTryGetStridedViewTest, SingleElement) {
  ShapeTracker tracker(ShapeUtil::MakeShape(F32, {1, 1}));
  ASSERT_OK(tracker.AppendReshape({1, 1, 1}));
  EXPECT_EQ(GetDimensionSegments(tracker), DimensionSegments({{}, {}, {}}));
}

TEST(ShapeTrackerTryGetStridedViewTest, ZeroElementsIsNotAStridedView) {
  ShapeTracker tracker(ShapeUtil::MakeShape(F32, {2, 0}));
  EXPECT_EQ(tracker.TryGetStridedView(), std::nullopt);
}

}  // namespace
}  // namespace xla
