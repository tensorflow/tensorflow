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

#include "tensorflow/compiler/xla/service/optimize_input_output_buffer_alias.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/platform/test.h"

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

    auto size_func = [](const Shape& shape) {
      return ShapeUtil::ByteSizeOf(shape);
    };

    optimize_pass_ =
        absl::make_unique<OptimizeInputOutputBufferAlias>(size_func);
  }

  // Returns the number of output indices that aliases with the input.
  int64 AliasCount() {
    int64 count = 0;

    config_.ForEachAlias(
        [&](const ShapeIndex&, const HloInputOutputAliasConfig::Alias&) {
          count++;
        });
    return count;
  }

  bool BuildAliasConfig(const Shape& input_shape, const Shape& output_shape) {
    config_ = HloInputOutputAliasConfig(output_shape);
    auto changed = optimize_pass_->Build(input_shape, output_shape, &config_);
    TF_CHECK_OK(changed.status());

    return changed.ValueOrDie();
  }

  std::unique_ptr<OptimizeInputOutputBufferAlias> optimize_pass_;

  HloInputOutputAliasConfig config_;

  Shape r1f32_;
  Shape r2f32_;
  Shape r3f32_;
  Shape r4f32_;
};

// All shapes are different, so no aliasing is available.
TEST_F(OptimizeInputOutputBufferAliasTest, AllDifferentBufferSizes) {
  Shape input = ShapeUtil::MakeTupleShape({r1f32_, r2f32_});
  Shape output = ShapeUtil::MakeTupleShape({r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_FALSE(changed);
  EXPECT_EQ(AliasCount(), 0);
}

// Input and output shapes are equal, so buffers can alias at the same index.
TEST_F(OptimizeInputOutputBufferAliasTest, OrderedNonNestedTuple) {
  Shape input = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  Shape output = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);
  EXPECT_EQ(AliasCount(), 4);

  EXPECT_EQ(config_.GetAliasedOutput(0, {0}), ShapeIndex{0});
  EXPECT_EQ(config_.GetAliasedOutput(0, {1}), ShapeIndex{1});
  EXPECT_EQ(config_.GetAliasedOutput(0, {2}), ShapeIndex{2});
  EXPECT_EQ(config_.GetAliasedOutput(0, {3}), ShapeIndex{3});
}

// Only a subset of the tuple element shapes match between the input and the
// output.
TEST_F(OptimizeInputOutputBufferAliasTest, PartialReuseNonNestedTuple) {
  Shape input = ShapeUtil::MakeTupleShape({r1f32_, r1f32_, r2f32_, r2f32_});
  Shape output = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 2);

  EXPECT_EQ(config_.GetAliasedOutput(0, {0}), ShapeIndex{0});
  EXPECT_EQ(config_.GetAliasedOutput(0, {2}), ShapeIndex{1});
}

// The output shape is reverse of the input shape, but we can still reuse all
// the buffers.
TEST_F(OptimizeInputOutputBufferAliasTest, UnorderedNonNestedTuple) {
  Shape input = ShapeUtil::MakeTupleShape({r1f32_, r2f32_, r3f32_, r4f32_});
  Shape output = ShapeUtil::MakeTupleShape({r4f32_, r3f32_, r2f32_, r1f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 4);

  EXPECT_EQ(config_.GetAliasedOutput(0, {0}), ShapeIndex{3});
  EXPECT_EQ(config_.GetAliasedOutput(0, {1}), ShapeIndex{2});
  EXPECT_EQ(config_.GetAliasedOutput(0, {2}), ShapeIndex{1});
  EXPECT_EQ(config_.GetAliasedOutput(0, {3}), ShapeIndex{0});
}

TEST_F(OptimizeInputOutputBufferAliasTest, UnorderedNestedTuple) {
  Shape input = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({r1f32_}), r2f32_, r3f32_, r4f32_});
  Shape output = ShapeUtil::MakeTupleShape(
      {r1f32_, ShapeUtil::MakeTupleShape({r3f32_, r2f32_}), r2f32_});
  bool changed = BuildAliasConfig(input, output);
  EXPECT_TRUE(changed);

  EXPECT_EQ(AliasCount(), 3);

  EXPECT_EQ(config_.GetAliasedOutput(0, {0, 0}), ShapeIndex{0});
  EXPECT_EQ(config_.GetAliasedOutput(0, {1}), ShapeIndex({1, 1}));
  EXPECT_EQ(config_.GetAliasedOutput(0, {2}), ShapeIndex({1, 0}));
}

}  // namespace xla
