/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#define EIGEN_USE_THREADS

#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

using tsl::testing::StatusIs;

struct HloShardingTestParam {
  tensorflow::Tensor in_tensor;
  std::vector<tensorflow::Tensor> expected_out_tensors;
  std::vector<int> device_indices;
  xla::HloSharding sharding;
};

struct ShardingParamTestParam {
  tensorflow::Tensor in_tensor;
  std::vector<tensorflow::Tensor> expected_out_tensors;
  std::vector<int> device_indices;

  // Parameter to form ShardingParam
  std::vector<int64_t> dim_shards;
  llvm::SmallVector<int, 4> permutation;
  llvm::SmallVector<int, 4> axis_sizes;
};

using ShardingParamTest = ::testing::TestWithParam<ShardingParamTestParam>;
using HloShardingTest = ::testing::TestWithParam<HloShardingTestParam>;

// Wrapper functions for building sharding specs for a given shape with a
// natural device order.
xla::HloSharding Tile(absl::Span<const int64_t> dims) {
  return xla::HloSharding::IotaTile(dims);
}
xla::HloSharding PartialTile(absl::Span<const int64_t> dims) {
  return xla::HloSharding::PartialTile(xla::TileAssignment(dims));
}
xla::HloSharding Replicate() { return xla::HloSharding::Replicate(); }

TEST_P(HloShardingTest, MakeAssembledArrayFromHostBuffer) {
  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "Resharding", kMaxParallelism);

  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);

  auto input_tensor = GetParam().in_tensor;

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto device_list,
                          xla::ifrt::test_util::GetDevices(
                              client.get(), GetParam().device_indices));

  auto sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), GetParam().sharding);

  TF_ASSERT_OK_AND_ASSIGN(
      auto assembled_array,
      MakeAssembledArrayFromHostBuffer(*client, input_tensor,
                                       std::move(sharding), device));

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled_arrays,
                          assembled_array->DisassembleIntoSingleDeviceArrays(
                              xla::ifrt::ArrayCopySemantics::kAlwaysCopy));

  ASSERT_EQ(disassembled_arrays.size(), GetParam().expected_out_tensors.size());

  tensorflow::Tensor host_tensor(tensorflow::DT_INT32,
                                 tensorflow::TensorShape({1, 2}));

  for (int i = 0; i < disassembled_arrays.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("Array ", i, " of ", disassembled_arrays.size()));
    auto disassembled_array = disassembled_arrays[i];
    auto expected_out_tensor = GetParam().expected_out_tensors[i];
    ASSERT_EQ(disassembled_array->shape(),
              xla::ifrt::Shape(expected_out_tensor.shape().dim_sizes()));
    tensorflow::Tensor host_tensor(expected_out_tensor.dtype(),
                                   expected_out_tensor.shape());
    TF_ASSERT_OK(
        disassembled_array
            ->CopyToHostBuffer(host_tensor.data(), /*byte_strides=*/{},
                               xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
            .Await());
    EXPECT_THAT(expected_out_tensor, tensorflow::test::TensorEq(host_tensor));
  }
}

INSTANTIATE_TEST_SUITE_P(
    HloShardingTests, HloShardingTest,
    ::testing::ValuesIn<HloShardingTestParam>(
        {
            // Full replication.
            {
                .in_tensor = test::AsTensor<int32_t>({1}, TensorShape({})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1}, TensorShape({})),
                        test::AsTensor<int32_t>({1}, TensorShape({})),
                    },
                .device_indices = {0, 1},
                .sharding = Replicate(),
            },
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3},
                                                     TensorShape({3, 1})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2, 3}, TensorShape({3, 1})),
                        test::AsTensor<int32_t>({1, 2, 3}, TensorShape({3, 1})),
                    },
                .device_indices = {0, 1},
                .sharding = Replicate(),
            },
            // 1-D sharding
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({4})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({2})),
                    },
                .device_indices = {0, 1},
                .sharding = Tile({2}),
            },
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                    },
                .device_indices = {0, 1},
                .sharding = Tile({2, 1}),
            },
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({1, 2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 3}, TensorShape({1, 2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({1, 2, 1})),
                    },
                .device_indices = {0, 1},
                .sharding = Tile({1, 1, 2}),
            },
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                                     TensorShape({4, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({5, 6}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({7, 8}, TensorShape({1, 2})),
                    },
                .device_indices = {0, 1, 2, 3},
                .sharding = Tile({4, 1}),
            },
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                                     TensorShape({4, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 3, 5, 7},
                                                TensorShape({4, 1})),
                        test::AsTensor<int32_t>({2, 4, 6, 8},
                                                TensorShape({4, 1})),
                    },
                .device_indices = {0, 1},
                .sharding = Tile({1, 2}),
            },
            // 2-D sharding
            {
                .in_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    TensorShape({4, 4})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2, 5, 6},
                                                TensorShape({2, 2})),
                        test::AsTensor<int32_t>({3, 4, 7, 8},
                                                TensorShape({2, 2})),
                        test::AsTensor<int32_t>({9, 10, 13, 14},
                                                TensorShape({2, 2})),
                        test::AsTensor<int32_t>({11, 12, 15, 16},
                                                TensorShape({2, 2})),
                    },
                .device_indices = {0, 1, 2, 3},
                .sharding = Tile({2, 2}),
            },
            {
                .in_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    TensorShape({4, 1, 4})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2, 5, 6},
                                                TensorShape({2, 1, 2})),
                        test::AsTensor<int32_t>({3, 4, 7, 8},
                                                TensorShape({2, 1, 2})),
                        test::AsTensor<int32_t>({9, 10, 13, 14},
                                                TensorShape({2, 1, 2})),
                        test::AsTensor<int32_t>({11, 12, 15, 16},
                                                TensorShape({2, 1, 2})),
                    },
                .device_indices = {0, 1, 2, 3},
                .sharding = Tile({2, 1, 2}),
            },
            // Partial replication
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 3}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({1, 3}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({2, 1})),
                    },
                .device_indices = {0, 1, 2, 3},
                .sharding = PartialTile({1, 2, 2}),
            },
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                    },
                .device_indices = {0, 1, 2, 3},
                .sharding = PartialTile({2, 1, 2}),
            },
        }));

TEST_P(ShardingParamTest, MakeAssembledArrayFromHostBuffer) {
  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "Resharding", kMaxParallelism);

  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);

  auto input_tensor = GetParam().in_tensor;

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto device_list,
                          xla::ifrt::test_util::GetDevices(
                              client.get(), GetParam().device_indices));

  xla::ifrt::ShardingParam sharding_param{
      GetParam().dim_shards,
      xla::ifrt::ShardingParam::MinorToMajor(GetParam().permutation,
                                             GetParam().axis_sizes)};

  TF_ASSERT_OK_AND_ASSIGN(
      auto sharding, xla::ifrt::ShardingParamSharding::Create(
                         sharding_param, device_list, xla::ifrt::MemoryKind()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto assembled_array,
      MakeAssembledArrayFromHostBuffer(*client, input_tensor,
                                       std::move(sharding), device));

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled_arrays,
                          assembled_array->DisassembleIntoSingleDeviceArrays(
                              xla::ifrt::ArrayCopySemantics::kAlwaysCopy));

  ASSERT_EQ(disassembled_arrays.size(), GetParam().expected_out_tensors.size());

  tensorflow::Tensor host_tensor(tensorflow::DT_INT32,
                                 tensorflow::TensorShape({1, 2}));

  for (int i = 0; i < disassembled_arrays.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("Array ", i, " of ", disassembled_arrays.size()));
    auto disassembled_array = disassembled_arrays[i];
    auto expected_out_tensor = GetParam().expected_out_tensors[i];
    ASSERT_EQ(disassembled_array->shape(),
              xla::ifrt::Shape(expected_out_tensor.shape().dim_sizes()));
    tensorflow::Tensor host_tensor(expected_out_tensor.dtype(),
                                   expected_out_tensor.shape());
    TF_ASSERT_OK(
        disassembled_array
            ->CopyToHostBuffer(host_tensor.data(), /*byte_strides=*/{},
                               xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
            .Await());
    EXPECT_THAT(expected_out_tensor, tensorflow::test::TensorEq(host_tensor));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ShardingParamTests, ShardingParamTest,
    ::testing::ValuesIn<ShardingParamTestParam>(
        {
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),

                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                    },
                .device_indices = {0, 1},
                .dim_shards = {2, 1},
                .permutation = {0, 1},
                .axis_sizes = {2, 1},
            },
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 3}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({2, 1})),
                    },
                .device_indices = {0, 1},
                .dim_shards = {1, 2},
                .permutation = {0, 1},
                .axis_sizes = {1, 2},
            },
            {
                .in_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    TensorShape({4, 4})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2, 5, 6},
                                                TensorShape({2, 2})),
                        test::AsTensor<int32_t>({3, 4, 7, 8},
                                                TensorShape({2, 2})),
                        test::AsTensor<int32_t>({9, 10, 13, 14},
                                                TensorShape({2, 2})),
                        test::AsTensor<int32_t>({11, 12, 15, 16},
                                                TensorShape({2, 2})),
                    },
                .device_indices = {0, 1, 2, 3},
                .dim_shards = {2, 2},
                .permutation = {0, 1},
                .axis_sizes = {2, 2},
            },
            {
                .in_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    TensorShape({4, 4})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                                TensorShape({2, 4})),
                        test::AsTensor<int32_t>({9, 10, 11, 12, 13, 14, 15, 16},
                                                TensorShape({2, 4})),
                    },
                .device_indices = {0, 1},
                .dim_shards = {2, 1},
                .permutation = {1, 0},
                .axis_sizes = {2, 1},
            },
            // Full replication
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2, 3, 4},
                                                TensorShape({2, 2})),
                        test::AsTensor<int32_t>({1, 2, 3, 4},
                                                TensorShape({2, 2})),
                    },
                .device_indices = {0, 1},
                .dim_shards = {1, 1},
                .permutation = {0},
                .axis_sizes = {2},
            },
            // Partial replication (aka replicate_on_last_tile_dim = true)
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 3}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({1, 3}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({2, 1})),
                    },
                .device_indices = {0, 1, 2, 3},
                .dim_shards = {1, 2},
                .permutation = {0, 1},
                .axis_sizes = {2, 2},
            },
            // Partial replication that shards along the first dimension.
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                    },
                .device_indices = {0, 1, 2, 3},
                .dim_shards = {2, 1},
                .permutation = {0, 1},
                .axis_sizes = {2, 2},
            },
            // Partial replication with random device indices.
            {
                .in_tensor = test::AsTensor<int32_t>({1, 2, 3, 4},
                                                     TensorShape({2, 2})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1, 3}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({1, 3}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({2, 1})),
                    },
                .device_indices = {3, 1, 2, 0},
                .dim_shards = {1, 2},
                .permutation = {0, 1},
                .axis_sizes = {2, 2},
            },
        }));

TEST(ShardingUtilsTest, MismatchRank) {
  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "Resharding", kMaxParallelism);

  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);

  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, TensorShape({2, 1, 2}));

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_list, xla::ifrt::test_util::GetDevices(client.get(), {0, 1}));

  xla::ifrt::ShardingParam sharding_param = {
      /*dim_shards=*/{2, 1},
      xla::ifrt::ShardingParam::MinorToMajor(/*permutation=*/{0, 1},
                                             /*axis_sizes=*/{2, 1})};

  TF_ASSERT_OK_AND_ASSIGN(
      auto sharding, xla::ifrt::ShardingParamSharding::Create(
                         sharding_param, device_list, xla::ifrt::MemoryKind()));

  EXPECT_THAT(MakeAssembledArrayFromHostBuffer(*client, input_tensor,
                                               std::move(sharding), device),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Expect equal rank of 3 but got 2"));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
