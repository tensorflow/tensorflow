/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/slice_util.h"

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
#include "tsl/platform/status_matchers.h"

namespace tensorflow {
namespace dtensor {
namespace slice_util {
namespace {

using ::testing::SizeIs;
using ::tsl::testing::IsOk;

TEST(TokenTest, NormalizeDynamic) {
  auto spec = Token(Token::REGULAR, /*begin=*/0, /*end=*/0, /*stride=*/1,
                    /*dynamic_mask=*/true,
                    /*begin_mask=*/true, /*end_mask=*/true);
  EXPECT_EQ(spec.normalize(4).begin, 0);
  EXPECT_EQ(spec.normalize(4).end, 0);
  EXPECT_EQ(spec.normalize(4).dynamic_mask, true);
  EXPECT_EQ(spec.normalize(4).begin_mask, true);
  EXPECT_EQ(spec.normalize(4).end_mask, true);
}

TEST(TokenTest, NormalizeFullPositiveStride) {
  auto spec = Token(Token::REGULAR, /*begin=*/0, /*end=*/4, /*stride=*/1);
  EXPECT_EQ(spec.normalize(4).begin, 0);
  EXPECT_EQ(spec.normalize(4).end, 4);

  spec = Token(Token::REGULAR, /*begin=*/0, /*end=*/4, /*stride=*/2);
  EXPECT_EQ(spec.normalize(4).begin, 0);
  EXPECT_EQ(spec.normalize(4).end, 4);

  spec = Token(Token::REGULAR, /*begin=*/0, /*end=*/4, /*stride=*/3);
  EXPECT_EQ(spec.normalize(4).begin, 0);
  EXPECT_EQ(spec.normalize(4).end, 6);

  spec = Token(Token::REGULAR, /*begin=*/0, /*end=*/4, /*stride=*/5);
  EXPECT_EQ(spec.normalize(4).begin, 0);
  EXPECT_EQ(spec.normalize(4).end, 5);
}

TEST(TokenTest, NormalizeFullNegativeStride) {
  auto spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/-1, /*stride=*/-1);
  EXPECT_EQ(spec.normalize(4).begin, 3);
  EXPECT_EQ(spec.normalize(4).end, -1);

  spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/-1, /*stride=*/-2);
  EXPECT_EQ(spec.normalize(4).begin, 3);
  EXPECT_EQ(spec.normalize(4).end, -1);

  spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/-1, /*stride=*/-3);
  EXPECT_EQ(spec.normalize(4).begin, 3);
  EXPECT_EQ(spec.normalize(4).end, -3);

  spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/-1, /*stride=*/-5);
  EXPECT_EQ(spec.normalize(4).begin, 3);
  EXPECT_EQ(spec.normalize(4).end, -2);
}

TEST(TokenTest, NormalizeZeroPositiveStride) {
  auto spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/3, /*stride=*/1);
  EXPECT_EQ(spec.normalize(7).begin, 3);
  EXPECT_EQ(spec.normalize(7).end, 3);

  spec = Token(Token::REGULAR, /*begin=*/0, /*end=*/0, /*stride=*/1);
  EXPECT_EQ(spec.normalize(7).begin, 0);
  EXPECT_EQ(spec.normalize(7).end, 0);
}

TEST(TokenTest, NormalizeZeroNegativeStride) {
  auto spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/3, /*stride=*/-1);
  EXPECT_EQ(spec.normalize(7).begin, 3);
  EXPECT_EQ(spec.normalize(7).end, 3);

  spec = Token(Token::REGULAR, /*begin=*/0, /*end=*/0, /*stride=*/-1);
  EXPECT_EQ(spec.normalize(7).begin, 0);
  EXPECT_EQ(spec.normalize(7).end, 0);
}

TEST(TokenTest, NormalizePartialPositiveStride) {
  auto spec = Token(Token::REGULAR, /*begin=*/1, /*end=*/5, /*stride=*/1);
  EXPECT_EQ(spec.normalize(7).begin, 1);
  EXPECT_EQ(spec.normalize(7).end, 5);

  spec = Token(Token::REGULAR, /*begin=*/1, /*end=*/5, /*stride=*/2);
  EXPECT_EQ(spec.normalize(7).begin, 1);
  EXPECT_EQ(spec.normalize(7).end, 5);

  spec = Token(Token::REGULAR, /*begin=*/1, /*end=*/5, /*stride=*/3);
  EXPECT_EQ(spec.normalize(7).begin, 1);
  EXPECT_EQ(spec.normalize(7).end, 7);

  spec = Token(Token::REGULAR, /*begin=*/1, /*end=*/5, /*stride=*/5);
  EXPECT_EQ(spec.normalize(7).begin, 1);
  EXPECT_EQ(spec.normalize(7).end, 6);

  spec = Token(Token::REGULAR, /*begin=*/1, /*end=*/-1, /*stride=*/1);
  EXPECT_EQ(spec.normalize(7).begin, 1);
  EXPECT_EQ(spec.normalize(7).end, 6);

  spec = Token(Token::REGULAR, /*begin=*/0, /*end=*/-1, /*stride=*/1);
  EXPECT_EQ(spec.normalize(7).begin, 0);
  EXPECT_EQ(spec.normalize(7).end, 6);
}

TEST(TokenTest, NormalizePartialNegativeStride) {
  auto spec = Token(Token::REGULAR, /*begin=*/6, /*end=*/2, /*stride=*/-1);
  EXPECT_EQ(spec.normalize(7).begin, 6);
  EXPECT_EQ(spec.normalize(7).end, 2);

  spec = Token(Token::REGULAR, /*begin=*/6, /*end=*/2, /*stride=*/-2);
  EXPECT_EQ(spec.normalize(7).begin, 6);
  EXPECT_EQ(spec.normalize(7).end, 2);

  spec = Token(Token::REGULAR, /*begin=*/6, /*end=*/2, /*stride=*/-3);
  EXPECT_EQ(spec.normalize(7).begin, 6);
  EXPECT_EQ(spec.normalize(7).end, 0);

  spec = Token(Token::REGULAR, /*begin=*/6, /*end=*/2, /*stride=*/-5);
  EXPECT_EQ(spec.normalize(7).begin, 6);
  EXPECT_EQ(spec.normalize(7).end, 1);
}

TEST(TokenTest, NormalizeFarFromCenter) {
  auto spec = Token(Token::REGULAR, /*begin=*/100, /*end=*/102, /*stride=*/1);
  EXPECT_EQ(spec.normalize(9).begin, 1);
  EXPECT_EQ(spec.normalize(9).end, 3);
}

TEST(TokenTest, NormalizeBeginMask) {
  auto spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/2, /*stride=*/1);
  spec.begin_mask = true;
  EXPECT_EQ(spec.normalize(7).begin, 0);

  spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/2, /*stride=*/-1);
  spec.begin_mask = true;
  EXPECT_EQ(spec.normalize(7).begin, 6);
}

TEST(TokenTest, NormalizeEndMask) {
  auto spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/2, /*stride=*/1);
  spec.end_mask = true;
  EXPECT_EQ(spec.normalize(7).end, 7);

  spec = Token(Token::REGULAR, /*begin=*/3, /*end=*/2, /*stride=*/-1);
  spec.end_mask = true;
  EXPECT_EQ(spec.normalize(7).end, -1);
}

class InferenceTest : public ::testing::Test {
 protected:
  Mesh GetMesh() {
    return Mesh::CreateMesh("MyMesh", /*dim_names=*/{"x", "y"},
                            /*mesh_shape=*/{2, 1},
                            /*global_device_ids=*/{0, 1},
                            /*global_devices_str=*/
                            {"/job:localhost/task:0/device:CPU:0",
                             "/job:localhost/task:0/device:CPU:1"},
                            /*local_device_ids=*/{0, 1},
                            /*local_devices_str=*/
                            {"/job:localhost/task:0/device:CPU:0",
                             "/job:localhost/task:0/device:CPU:1"},
                            /*use_xla_spmd=*/false);
  }
};

TEST_F(InferenceTest, FullyReplicatedInputs) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, Layout::kUnshardedDim},
      GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, Layout::kUnshardedDim},
      GetMesh());
  const auto specs = std::vector<Token>{
      Token(Token::REGULAR, /*begin=*/0, /*end=*/-1, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false),
      Token(Token::REGULAR, /*begin=*/0, /*end=*/2, /*stride=*/2,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false)};
  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(
      forward->expander_input_layout().sharding_spec_strs(),
      std::vector<std::string>({Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(2));
  EXPECT_EQ(forward->local_tokens()[0].end, -1);
  EXPECT_EQ(forward->local_tokens()[1].end, 2);

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(
      backward->expander_value_layout().sharding_spec_strs(),
      std::vector<std::string>({Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_EQ(
      backward->expander_input_layout().sharding_spec_strs(),
      std::vector<std::string>({Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_THAT(backward->local_tokens(), SizeIs(2));
  EXPECT_EQ(backward->local_tokens()[0].end, -1);
  EXPECT_EQ(backward->local_tokens()[1].end, 2);
}

TEST_F(InferenceTest, NewAxisMask) {
  const Layout input_layout =
      *Layout::GetLayout(std::vector<std::string>{"x", "y"}, GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, Layout::kUnshardedDim,
                               "x", "y"},
      GetMesh());

  const auto specs = std::vector<Token>{
      Token(Token::NEW_AXIS, /*begin=*/0, /*end=*/0, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false),
      Token(Token::NEW_AXIS, /*begin=*/0, /*end=*/0, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false),
      Token(Token::REGULAR, /*begin=*/0, /*end=*/2, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false),
      Token(Token::REGULAR, /*begin=*/0, /*end=*/4, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false)};

  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(forward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({"x", "y"}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(4));
  EXPECT_EQ(forward->local_tokens()[0].end, 0);
  EXPECT_EQ(forward->local_tokens()[1].end, 0);
  EXPECT_EQ(forward->local_tokens()[2].end, 1);  // dim_size(x) == 2.
  EXPECT_EQ(forward->local_tokens()[3].end, 4);

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(backward->expander_value_layout().sharding_spec_strs(),
            std::vector<std::string>(
                {Layout::kUnshardedDim, Layout::kUnshardedDim, "x", "y"}));
  EXPECT_EQ(backward->expander_input_layout(), input_layout);
  EXPECT_THAT(backward->local_tokens(), SizeIs(4));
  EXPECT_EQ(backward->local_tokens()[0].end, 0);
  EXPECT_EQ(backward->local_tokens()[1].end, 0);
  EXPECT_EQ(backward->local_tokens()[2].end, 1);  // dim_size(x) == 2.
  EXPECT_EQ(backward->local_tokens()[3].end, 4);
}

TEST_F(InferenceTest, ShrinkAxisMask) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, Layout::kUnshardedDim},
      GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim}, GetMesh());
  const auto specs = std::vector<Token>{
      Token(Token::REGULAR, /*begin=*/0, /*end=*/-1, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false),
      Token(Token::SHRINK_AXIS, /*begin=*/0, /*end=*/2,
            /*stride=*/1, /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false)};

  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(
      forward->expander_input_layout().sharding_spec_strs(),
      std::vector<std::string>({Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(2));
  EXPECT_EQ(forward->local_tokens()[0].end, -1);
  EXPECT_EQ(forward->local_tokens()[1].end, 2);

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(backward->expander_value_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim}));
  EXPECT_THAT(backward->local_tokens(), SizeIs(2));
  EXPECT_EQ(backward->expander_input_layout(), input_layout);
  EXPECT_EQ(backward->local_tokens()[0].end, -1);
  EXPECT_EQ(backward->local_tokens()[1].end, 2);
}

TEST_F(InferenceTest, EllipsisMask) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{"x", "y", Layout::kUnshardedDim}, GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{"x", "y", Layout::kUnshardedDim,
                               Layout::kUnshardedDim, Layout::kUnshardedDim},
      GetMesh());

  const auto specs =
      std::vector<Token>{Token(Token::ELLIPSIS, /*begin=*/0, /*end=*/0,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false),
                         Token(Token::NEW_AXIS, /*begin=*/0, /*end=*/0,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false),
                         Token(Token::NEW_AXIS, /*begin=*/0, /*end=*/0,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false)};

  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4, 6});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(forward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({"x", "y", Layout::kUnshardedDim}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  // No local specs for the ellipsis axes "x" and "y".
  EXPECT_THAT(forward->local_tokens(), SizeIs(3));
  EXPECT_EQ(forward->local_tokens()[0].end, 0);
  EXPECT_EQ(forward->local_tokens()[1].end, 0);
  EXPECT_EQ(forward->local_tokens()[2].end, 0);

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4, 6});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(
      backward->expander_value_layout().sharding_spec_strs(),
      std::vector<std::string>({"x", "y", Layout::kUnshardedDim,
                                Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_EQ(backward->expander_input_layout(), input_layout);
  // No local specs for the ellipsis axes "x" and "y".
  EXPECT_THAT(backward->local_tokens(), SizeIs(3));
  EXPECT_EQ(backward->local_tokens()[0].end, 0);
  EXPECT_EQ(backward->local_tokens()[1].end, 0);
  EXPECT_EQ(backward->local_tokens()[2].end, 0);
}

TEST_F(InferenceTest, EllipsisNewAxisEndMask) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim}, GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, Layout::kUnshardedDim},
      GetMesh());
  const auto specs = std::vector<Token>{
      Token(Token::ELLIPSIS, /*begin=*/0, /*end=*/0, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false),
      Token(Token::NEW_AXIS, /*begin=*/0, /*end=*/0, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/false,
            /*end_mask=*/false),
      Token(Token::REGULAR, /*begin=*/0, /*end=*/0, /*stride=*/1,
            /*dynamic_mask=*/false,
            /*begin_mask=*/true,
            /*end_mask=*/true),
  };
  auto forward = CreateAndRun<ForwardLayoutInference>(specs, input_layout,
                                                      std::vector<int64_t>{2});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(forward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(3));
  EXPECT_EQ(forward->local_tokens()[0].end, 0);
  EXPECT_EQ(forward->local_tokens()[1].end, 0);
  EXPECT_EQ(forward->local_tokens()[2].end, 2);

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(
      backward->expander_value_layout().sharding_spec_strs(),
      std::vector<std::string>({Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_EQ(backward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim}));
  EXPECT_THAT(backward->local_tokens(), SizeIs(3));
  EXPECT_EQ(backward->local_tokens()[0].end, 0);
  EXPECT_EQ(backward->local_tokens()[1].end, 0);
  EXPECT_EQ(backward->local_tokens()[2].end, 2);
}

TEST_F(InferenceTest, AdditionalAxes) {
  const Layout input_layout =
      *Layout::GetLayout(std::vector<std::string>{"x", "y"}, GetMesh());
  const Layout output_layout =
      *Layout::GetLayout(std::vector<std::string>{"x", "y"}, GetMesh());
  const auto specs =
      std::vector<Token>{Token(Token::REGULAR, /*begin=*/0, /*end=*/0,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/true,
                               /*end_mask=*/true)};

  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(forward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({"x", "y"}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(1));
  EXPECT_EQ(forward->local_tokens()[0].begin_mask, true);
  EXPECT_EQ(forward->local_tokens()[0].end_mask, true);

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(backward->expander_value_layout().sharding_spec_strs(),
            std::vector<std::string>({"x", "y"}));
  EXPECT_EQ(backward->expander_input_layout(), input_layout);
  EXPECT_THAT(backward->local_tokens(), SizeIs(1));
  EXPECT_EQ(forward->local_tokens()[0].begin_mask, true);
  EXPECT_EQ(forward->local_tokens()[0].end_mask, true);
}

TEST_F(InferenceTest, ShardingOnNonSlicedDimension) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{"x", Layout::kUnshardedDim}, GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{"x", Layout::kUnshardedDim}, GetMesh());
  const auto specs =
      std::vector<Token>{Token(Token::REGULAR, /*begin=*/0, /*end=*/2,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false),
                         Token(Token::REGULAR, /*begin=*/0, /*end=*/2,
                               /*stride=*/2, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false)};

  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_EQ(forward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({"x", Layout::kUnshardedDim}));
  EXPECT_THAT(forward->local_tokens(), SizeIs(2));
  EXPECT_EQ(forward->local_tokens()[0].end, 1);  // dim_size(x) == 2
  EXPECT_EQ(forward->local_tokens()[1].end, 2);

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(backward->expander_input_layout(), input_layout);
  EXPECT_EQ(backward->expander_value_layout().sharding_spec_strs(),
            std::vector<std::string>({"x", Layout::kUnshardedDim}));
  EXPECT_THAT(backward->local_tokens(), SizeIs(2));
  EXPECT_EQ(backward->local_tokens()[0].end, 1);  // dim_size(x) == 2
  EXPECT_EQ(backward->local_tokens()[1].end, 2);
}

TEST_F(InferenceTest, StrideOnShardedDimensionNoRelayout1) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, "x"}, GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, "x"}, GetMesh());
  const auto specs =
      std::vector<Token>{Token(Token::REGULAR, /*begin=*/0, /*end=*/2,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false),
                         Token(Token::REGULAR, /*begin=*/0, /*end=*/4,
                               /*stride=*/2, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false)};
  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(forward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim, "x"}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(2));
  EXPECT_EQ(forward->local_tokens()[0].end, 2);
  EXPECT_EQ(forward->local_tokens()[1].end, 2);  // dim_size(x) == 2

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(backward->expander_input_layout(), input_layout);
  EXPECT_EQ(backward->expander_value_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim, "x"}));
  EXPECT_THAT(backward->local_tokens(), SizeIs(2));
  EXPECT_EQ(backward->local_tokens()[0].end, 2);  // dim_size(x) == 2
  EXPECT_EQ(backward->local_tokens()[1].end, 2);
}

TEST_F(InferenceTest, StrideOnShardedDimensionNoRelayout2) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, "y"}, GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, "y"}, GetMesh());
  const auto specs =
      std::vector<Token>{Token(Token::REGULAR, /*begin=*/0, /*end=*/2,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false),
                         Token(Token::REGULAR, /*begin=*/0, /*end=*/4,
                               /*stride=*/2, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false)};

  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(forward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim, "y"}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(2));
  EXPECT_EQ(forward->local_tokens()[0].end, 2);
  EXPECT_EQ(forward->local_tokens()[1].end, 4);  // dim_size(x) == 1

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(backward->expander_input_layout(), input_layout);
  EXPECT_EQ(backward->expander_value_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim, "y"}));
  EXPECT_THAT(backward->local_tokens(), SizeIs(2));
  EXPECT_EQ(backward->local_tokens()[0].end, 2);  // dim_size(x) == 2
  EXPECT_EQ(backward->local_tokens()[1].end, 4);
}

TEST_F(InferenceTest, StrideOnShardedDimensionNoRelayout3) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, "x"}, GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, "x"}, GetMesh());
  const auto specs =
      std::vector<Token>{Token(Token::REGULAR, /*begin=*/0, /*end=*/2,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false),
                         Token(Token::REGULAR, /*begin=*/0, /*end=*/3,
                               /*stride=*/2, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false)};

  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(forward->expander_input_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim, "x"}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(2));
  EXPECT_EQ(forward->local_tokens()[0].end, 2);
  EXPECT_EQ(forward->local_tokens()[1].end, 2);  // dim_size(x) == 2

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  EXPECT_EQ(backward->expander_input_layout(), input_layout);
  EXPECT_EQ(backward->expander_value_layout().sharding_spec_strs(),
            std::vector<std::string>({Layout::kUnshardedDim, "x"}));
  EXPECT_THAT(backward->local_tokens(), SizeIs(2));
  EXPECT_EQ(backward->local_tokens()[0].end, 2);  // dim_size(x) == 2
  EXPECT_EQ(backward->local_tokens()[1].end, 2);
}

TEST_F(InferenceTest, StrideOnShardedDimensionNeedRelayout) {
  const Layout input_layout = *Layout::GetLayout(
      std::vector<std::string>{"x", Layout::kUnshardedDim}, GetMesh());
  const Layout output_layout = *Layout::GetLayout(
      std::vector<std::string>{Layout::kUnshardedDim, Layout::kUnshardedDim},
      GetMesh());

  const auto specs =
      std::vector<Token>{Token(Token::REGULAR, /*begin=*/0, /*end=*/-1,
                               /*stride=*/1, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false),
                         Token(Token::REGULAR, /*begin=*/0, /*end=*/4,
                               /*stride=*/3, /*dynamic_mask=*/false,
                               /*begin_mask=*/false,
                               /*end_mask=*/false)};

  auto forward = CreateAndRun<ForwardLayoutInference>(
      specs, input_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(forward, IsOk());
  EXPECT_EQ(
      forward->expander_input_layout().sharding_spec_strs(),
      std::vector<std::string>({Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_EQ(forward->expander_value_layout(), output_layout);
  EXPECT_THAT(forward->local_tokens(), SizeIs(2));
  EXPECT_EQ(forward->local_tokens()[0].end, -1);
  EXPECT_EQ(forward->local_tokens()[1].end, 4);

  auto backward = CreateAndRun<BackwardLayoutInference>(
      specs, output_layout, std::vector<int64_t>{2, 4});
  ASSERT_THAT(backward, IsOk());
  // The backward inferred input_layout prefers replicated layouts for this
  // case.
  EXPECT_EQ(
      backward->expander_input_layout().sharding_spec_strs(),
      std::vector<std::string>({Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_EQ(
      backward->expander_value_layout().sharding_spec_strs(),
      std::vector<std::string>({Layout::kUnshardedDim, Layout::kUnshardedDim}));
  EXPECT_THAT(backward->local_tokens(), SizeIs(2));
  EXPECT_EQ(backward->local_tokens()[0].end, -1);
  EXPECT_EQ(backward->local_tokens()[1].end, 4);
}

}  // namespace
}  // namespace slice_util
}  // namespace dtensor
}  // namespace tensorflow
