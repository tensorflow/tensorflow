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
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_tensor_utils.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

using tensorflow::test::TensorEq;
using tsl::testing::StatusIs;

struct ReshardToTensorTestParam {
  // split tensors in natural device order.
  std::vector<tensorflow::Tensor> split_tensors;
  tensorflow::Tensor expected_out_tensor;
  std::vector<int> device_indices;
  xla::HloSharding sharding;
};

struct TensorToArrayTestParam {
  tensorflow::Tensor in_tensor;
  std::vector<tensorflow::Tensor> expected_out_tensors;
  std::vector<int> device_ids;
  xla::HloSharding sharding;
};

using ReshardToTensorTest = ::testing::TestWithParam<ReshardToTensorTestParam>;
using TensorToArrayTest = ::testing::TestWithParam<TensorToArrayTestParam>;

// Wrapper functions for building sharding specs for a given shape with a
// natural device order.
xla::HloSharding Tile(absl::Span<const int64_t> dims) {
  return xla::HloSharding::IotaTile(dims);
}
xla::HloSharding PartialTile(absl::Span<const int64_t> dims) {
  return xla::HloSharding::PartialTile(xla::TileAssignment(dims));
}
xla::HloSharding Replicate() { return xla::HloSharding::Replicate(); }
xla::HloSharding Maximal(int64_t device_index = 0) {
  return xla::HloSharding::AssignDevice(device_index);
}

TEST_P(ReshardToTensorTest, MakeHostTensorFromDeviceArrays) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto device_list,
                          xla::ifrt::test_util::GetDevices(
                              client.get(), GetParam().device_indices));

  std::vector<tsl::RCReference<xla::ifrt::Array>> split_arrays;
  for (int i = 0; i < GetParam().split_tensors.size(); ++i) {
    const auto& split_tensor = GetParam().split_tensors[i];
    auto single_device_sharding = xla::ifrt::SingleDeviceSharding::Create(
        device_list[i], xla::ifrt::MemoryKind());
    TF_ASSERT_OK_AND_ASSIGN(auto dtype, ToIfrtDType(split_tensor.dtype()));
    TF_ASSERT_OK_AND_ASSIGN(
        auto array,
        client->MakeArrayFromHostBuffer(
            split_tensor.data(), dtype, ToIfrtShape(split_tensor.shape()),
            /*byte_strides=*/{}, std::move(single_device_sharding),
            xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/{}));
    split_arrays.push_back(std::move(array));
  }

  auto ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), GetParam().sharding);
  tsl::RCReference<xla::ifrt::Array> assembled_array;

  TF_ASSERT_OK_AND_ASSIGN(
      assembled_array,
      client->AssembleArrayFromSingleDeviceArrays(
          ToIfrtShape(GetParam().expected_out_tensor.shape()),
          std::move(ifrt_sharding), absl::MakeSpan(split_arrays),
          xla::ifrt::ArrayCopySemantics::kAlwaysCopy));

  TF_ASSERT_OK_AND_ASSIGN(
      auto output_tensor,
      MakeTensorFromArray(*client, *assembled_array, GetParam().sharding,
                          device_list, thread_pool));

  EXPECT_THAT(GetParam().expected_out_tensor, TensorEq(output_tensor));
}

INSTANTIATE_TEST_SUITE_P(
    HloShardingTests, ReshardToTensorTest,
    ::testing::ValuesIn<ReshardToTensorTestParam>(
        {
            // Maximal
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({3}, TensorShape({})),
                    },
                .expected_out_tensor = test::AsTensor<int32_t>({3},
                                                               TensorShape({})),
                .device_indices = {0},
                .sharding = Maximal(0),
            },
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({3}, TensorShape({})),
                        test::AsTensor<int32_t>({4}, TensorShape({})),
                    },
                .expected_out_tensor = test::AsTensor<int32_t>({4},
                                                               TensorShape({})),
                .device_indices = {0, 1},
                .sharding = Maximal(1),
            },

            // Full replication.
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({1}, TensorShape({})),
                        test::AsTensor<int32_t>({1}, TensorShape({})),
                    },
                .expected_out_tensor = test::AsTensor<int32_t>({1},
                                                               TensorShape({})),
                .device_indices = {0, 1},
                .sharding = Replicate(),
            },
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2, 3}, TensorShape({3, 1})),
                        test::AsTensor<int32_t>({1, 2, 3}, TensorShape({3, 1})),
                    },
                .expected_out_tensor =
                    test::AsTensor<int32_t>({1, 2, 3}, TensorShape({3, 1})),
                .device_indices = {0, 1},
                .sharding = Replicate(),
            },

            // 1-D sharding
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({2})),
                    },
                .expected_out_tensor =
                    test::AsTensor<int32_t>({1, 2, 3, 4}, TensorShape({4})),
                .device_indices = {0, 1},
                .sharding = Tile({2}),
            },
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                    },
                .expected_out_tensor =
                    test::AsTensor<int32_t>({1, 2, 3, 4}, TensorShape({2, 2})),
                .device_indices = {0, 1},
                .sharding = Tile({2, 1}),
            },
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({1, 3}, TensorShape({1, 2, 1})),
                        test::AsTensor<int32_t>({2, 4}, TensorShape({1, 2, 1})),
                    },
                .expected_out_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4}, TensorShape({1, 2, 2})),
                .device_indices = {0, 1},
                .sharding = Tile({1, 1, 2}),
            },
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({1, 2}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({3, 4}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({5, 6}, TensorShape({1, 2})),
                        test::AsTensor<int32_t>({7, 8}, TensorShape({1, 2})),
                    },
                .expected_out_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8}, TensorShape({4, 2})),
                .device_indices = {0, 1, 2, 3},
                .sharding = Tile({4, 1}),
            },
            {
                .split_tensors =
                    {
                        test::AsTensor<int32_t>({1, 3, 5, 7},
                                                TensorShape({4, 1})),
                        test::AsTensor<int32_t>({2, 4, 6, 8},
                                                TensorShape({4, 1})),
                    },
                .expected_out_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8}, TensorShape({4, 2})),
                .device_indices = {0, 1},
                .sharding = Tile({1, 2}),
            },
            // 2-D sharding
            {
                .split_tensors =
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
                .expected_out_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    TensorShape({4, 4})),
                .device_indices = {0, 1, 2, 3},
                .sharding = Tile({2, 2}),
            },
            {
                .split_tensors =
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
                .expected_out_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    TensorShape({4, 1, 4})),
                .device_indices = {0, 1, 2, 3},
                .sharding = Tile({2, 1, 2}),
            },
            {
                .split_tensors =
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
                .expected_out_tensor = test::AsTensor<int32_t>(
                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    TensorShape({4, 1, 4})),
                .device_indices = {3, 2, 1, 0},
                .sharding = Tile({2, 1, 2}),
            },
        }));

TEST_P(TensorToArrayTest, MakeArrayFromTensor) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  auto input_tensor = GetParam().in_tensor;

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(
      auto assembled_array,
      MakeArrayFromTensor(*client, input_tensor,
                          absl::MakeSpan(GetParam().device_ids),
                          GetParam().sharding, thread_pool));

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled_arrays,
                          assembled_array->DisassembleIntoSingleDeviceArrays(
                              xla::ifrt::ArrayCopySemantics::kAlwaysCopy));

  ASSERT_EQ(disassembled_arrays.size(), GetParam().expected_out_tensors.size());

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
    EXPECT_THAT(expected_out_tensor, TensorEq(host_tensor));
  }
}

INSTANTIATE_TEST_SUITE_P(
    TensorToArrayTests, TensorToArrayTest,
    ::testing::ValuesIn<TensorToArrayTestParam>(
        {
            // Single device
            {
                .in_tensor = test::AsTensor<int32_t>({1}, TensorShape({})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1}, TensorShape({})),
                    },
                .device_ids = {0},
                .sharding = Replicate(),
            },
            {
                .in_tensor = test::AsTensor<int32_t>({2}, TensorShape({})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({2}, TensorShape({})),
                    },
                .device_ids = {0},
                .sharding = Maximal(0),
            },
            {
                .in_tensor = test::AsTensor<int32_t>({3}, TensorShape({})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({3}, TensorShape({})),
                    },
                .device_ids = {0, 1},
                .sharding = Maximal(1),
            },
            // Full replication.
            {
                .in_tensor = test::AsTensor<int32_t>({1}, TensorShape({})),
                .expected_out_tensors =
                    {
                        test::AsTensor<int32_t>({1}, TensorShape({})),
                        test::AsTensor<int32_t>({1}, TensorShape({})),
                    },
                .device_ids = {0, 1},
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
                .device_ids = {0, 1},
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
                .device_ids = {0, 1},
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
                .device_ids = {0, 1},
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
                .device_ids = {0, 1},
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
                .device_ids = {0, 1, 2, 3},
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
                .device_ids = {0, 1},
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
                .device_ids = {0, 1, 2, 3},
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
                .device_ids = {0, 1, 2, 3},
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
                .device_ids = {0, 1, 2, 3},
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
                .device_ids = {0, 1, 2, 3},
                .sharding = PartialTile({2, 1, 2}),
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
                .device_ids = {3, 2, 1, 0},
                .sharding = PartialTile({2, 1, 2}),
            },
        }));

TEST(ShardingUtilsTest, MismatchRank) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, TensorShape({2, 1, 2}));

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_list, xla::ifrt::test_util::GetDevices(client.get(), {0, 1}));

  xla::HloSharding sharding = Tile({2, 1});

  EXPECT_THAT(MakeArrayFromTensor(*client, input_tensor, device_list,
                                  std::move(sharding), thread_pool),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "shape must have 2 dimensions, but has 3 dimensions: "
                       "shape=[2,1,2], sharding={devices=[2,1]<=[2]}"));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
