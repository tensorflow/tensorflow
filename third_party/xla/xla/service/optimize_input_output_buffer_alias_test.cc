/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/optimize_input_output_buffer_alias.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/test.h"

namespace xla {

// Tests that UserBufferAlias properly maps input and output buffer indices of
// various shapes for aliasing.
class OptimizeInputOutputBufferAliasTest : public HloTestBase {
 protected:
  OptimizeInputOutputBufferAliasTest() {
    r1f32_ = ShapeUtil::MakeShape(F32, {4});
    r2f32_ = ShapeUtil::MakeShape(F32, {4, 5});
    r3f32_ = ShapeUtil::MakeShape(F32, {4, 5, 6});
    r4f32_ = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});
    d1f32_ = ShapeUtil::MakeShape(F32, {256}, /*dynamic_dimensions=*/{true});
    d2f32_ = ShapeUtil::MakeShape(F32, {128, 128},
                                  /*dynamic_dimensions=*/{false, true});
    // Static shape with same size as dynamic shape `d1f32_`.
    d3f32_ = ShapeUtil::MakeShape(F32, {512});
  }
  void CreatePassAndBufferDonorConfig(
      bool registered_donor_buffer_only = false) {
    optimize_pass_ = std::make_unique<OptimizeInputOutputBufferAlias>(
        registered_donor_buffer_only);
    buffer_donor_config_ = HloBufferDonorConfig();
  }

  // Returns the number of output indices that aliases with the input.
  int64_t AliasCount() {
    int64_t count = 0;

    alias_config_.ForEachAlias(
        [&](const ShapeIndex&, const HloInputOutputAliasConfig::Alias&) {
          count++;
        });
    return count;
  }

  bool BuildAliasConfig(const std::vector<Shape>& input_shapes,
                        const Shape& output_shape) {
    alias_config_ = HloInputOutputAliasConfig(output_shape);

    auto changed = optimize_pass_->Build(input_shapes, output_shape,
                                         &alias_config_, &buffer_donor_config_);
    TF_CHECK_OK(changed.status());

    return changed.value();
  }

  std::unique_ptr<OptimizeInputOutputBufferAlias> optimize_pass_;

  HloInputOutputAliasConfig alias_config_;
  HloBufferDonorConfig buffer_donor_config_;

  Shape r1f32_;
  Shape r2f32_;
  Shape r3f32_;
  Shape r4f32_;
  // Used for dynamic shape aliasing check.
  Shape d1f32_;
  Shape d2f32_;
  Shape d3f32_;
};

// All shapes are different, so no aliasing is available.
TEST_F(OptimizeInputOutputBufferAliasTest, AllDifferentBufferSizes) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {ShapeUtil::MakeTupleShape({r1f32_, r2f32_})};
  Shape output = ShapeUtil::MakeTupleShape({r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_FALSE(changed);
  EXPECT_EQ(AliasCount(), 0);
}

// Input and output shapes are equal, so buffers can alias at the same index.
TEST_F(OptimizeInputOutputBufferAliasTest, OrderedNonNestedTuple) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {
      ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_})};
  Shape output = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);
  EXPECT_EQ(AliasCount(), 4);

  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {0}), ShapeIndex{0});
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {1}), ShapeIndex{1});
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {2}), ShapeIndex{2});
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {3}), ShapeIndex{3});
}

// Only a subset of the tuple element shapes match between the input and the
// output.
TEST_F(OptimizeInputOutputBufferAliasTest, PartialReuseNonNestedTuple) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {
      ShapeUtil::MakeTupleShape({r1f32_, r1f32_, r2f32_, r2f32_})};
  Shape output = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 2);

  EXPECT_TRUE(alias_config_.OutputHasAlias(ShapeIndex{0}));
  EXPECT_TRUE(alias_config_.OutputHasAlias(ShapeIndex{1}));
  EXPECT_FALSE(alias_config_.OutputHasAlias(ShapeIndex{2}));
  EXPECT_FALSE(alias_config_.OutputHasAlias(ShapeIndex{3}));
}

// The output shape is reverse of the input shape, but we can still reuse all
// the buffers.
TEST_F(OptimizeInputOutputBufferAliasTest, UnorderedNonNestedTuple) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {
      ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_})};
  Shape output = ShapeUtil::MakeTupleShape({r4f32_, r3f32_, r2f32_, r1f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 4);

  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {0}), ShapeIndex{3});
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {1}), ShapeIndex{2});
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {2}), ShapeIndex{1});
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {3}), ShapeIndex{0});
}

TEST_F(OptimizeInputOutputBufferAliasTest, UnorderedNestedTuple) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({r1f32_}), r2f32_, r3f32_, r4f32_})};
  Shape output = ShapeUtil::MakeTupleShape(
      {r1f32_, ShapeUtil::MakeTupleShape({r3f32_, r2f32_}), r2f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 3);

  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {0, 0}), ShapeIndex{0});
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {1}), ShapeIndex({1, 1}));
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {2}), ShapeIndex({1, 0}));
  EXPECT_FALSE(alias_config_.ParameterHasAlias(0, {0, 3}));
}

TEST_F(OptimizeInputOutputBufferAliasTest, MultipleParameters) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {{r1f32_, r2f32_, r3f32_, r4f32_}};
  Shape output = ShapeUtil::MakeTupleShape({r4f32_, r3f32_, r2f32_, r1f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 4);

  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {}), ShapeIndex{3});
  EXPECT_EQ(alias_config_.GetAliasedOutput(1, {}), ShapeIndex{2});
  EXPECT_EQ(alias_config_.GetAliasedOutput(2, {}), ShapeIndex{1});
  EXPECT_EQ(alias_config_.GetAliasedOutput(3, {}), ShapeIndex{0});
}

TEST_F(OptimizeInputOutputBufferAliasTest, BufferDonorOnly) {
  CreatePassAndBufferDonorConfig(true);
  std::vector<Shape> input = {ShapeUtil::MakeTupleShape({r1f32_, r2f32_})};
  Shape output = ShapeUtil::MakeTupleShape({r2f32_, r1f32_});

  TF_CHECK_OK(buffer_donor_config_.AddBufferDonor(0, {0}));
  EXPECT_TRUE(buffer_donor_config_.ParameterIsBufferDonor(0, {0}));

  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 1);

  EXPECT_FALSE(buffer_donor_config_.ParameterIsBufferDonor(0, {0}));
  EXPECT_EQ(alias_config_.GetAliasedOutput(0, {0}), ShapeIndex{1});
  EXPECT_FALSE(alias_config_.GetAliasedOutput(0, {1}));
}

// No aliasing on dynamic shapes with tuple.
TEST_F(OptimizeInputOutputBufferAliasTest, DynamicShapeWithTuple) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {ShapeUtil::MakeTupleShape({d1f32_, d2f32_})};
  Shape output = ShapeUtil::MakeTupleShape({d1f32_, d2f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_FALSE(changed);
  EXPECT_EQ(AliasCount(), 0);
}

// No aliasing on dynamic shapes with no input output tuple.
TEST_F(OptimizeInputOutputBufferAliasTest, DynamicShapeNoTuple) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {d1f32_, d2f32_};
  Shape output = d1f32_;
  bool changed = BuildAliasConfig(input, output);
  EXPECT_FALSE(changed);
  EXPECT_EQ(AliasCount(), 0);
}

// No aliasing from static input shape to dynamic output shapes even size is
// same.
TEST_F(OptimizeInputOutputBufferAliasTest, DynamicShapeBufferOutput) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {d1f32_, d2f32_};
  Shape output = d3f32_;
  bool changed = BuildAliasConfig(input, output);
  EXPECT_FALSE(changed);
  EXPECT_EQ(AliasCount(), 0);
}

// No aliasing from dynamic input shape to static output shapes even size is
// same.
TEST_F(OptimizeInputOutputBufferAliasTest, DynamicShapeBufferInput) {
  CreatePassAndBufferDonorConfig(false);
  std::vector<Shape> input = {d3f32_};
  Shape output = d1f32_;
  bool changed = BuildAliasConfig(input, output);
  EXPECT_FALSE(changed);
  EXPECT_EQ(AliasCount(), 0);
}

}  // namespace xla
