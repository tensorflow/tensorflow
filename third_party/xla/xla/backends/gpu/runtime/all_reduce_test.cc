/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/all_reduce.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::stream_executor::gpu::AllReduceStrategy;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

se::StreamExecutor* GetGpuExecutor(int64_t device_ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(device_ordinal).value();
}

struct TestParams {
  AllReduceStrategy all_reduce_strategy;
  int64_t num_elements;
};

class AllReduceKernelTest : public ::testing::Test,
                            public ::testing::WithParamInterface<TestParams> {
 public:
  AllReduceKernelTest() : params_(GetParam()) {}

  template <typename T>
  absl::StatusOr<std::vector<Array<T>>> RunKernel(
      const std::vector<se::StreamExecutor*>& executors,
      const std::vector<Array<T>>& input_data, ReductionKind reduction_kind) {
    const int64_t num_ranks = input_data.size();
    const LaunchDimensions launch_dimensions = AllReduceLaunchDimensions(
        input_data[0].num_elements(), num_ranks, params_.all_reduce_strategy);

    int64_t num_elements = input_data[0].num_elements();

    TF_RETURN_IF_ERROR(executors[0]->EnablePeerAccessTo(executors[1]));
    TF_RETURN_IF_ERROR(executors[1]->EnablePeerAccessTo(executors[0]));

    std::vector<std::unique_ptr<se::Stream>> streams;
    std::vector<se::DeviceMemoryHandle> local_input_buffers;
    std::vector<se::DeviceMemoryHandle> data_buffers;
    std::vector<se::DeviceMemoryHandle> signal_flags_buffers;
    std::vector<se::DeviceMemoryBase> remote_input_buffers_span;
    std::vector<se::DeviceMemoryBase> signal_flags_buffers_span;

    for (int i = 0; i < num_ranks; ++i) {
      auto* executor = executors[i];
      streams.push_back(executor->CreateStream().value());

      local_input_buffers.emplace_back(
          executor, executor->AllocateArray<T>(num_elements));
      TF_RET_CHECK(!local_input_buffers[i].memory().is_null());

      data_buffers.emplace_back(executor,
                                executor->AllocateArray<T>(num_elements));
      TF_RET_CHECK(!data_buffers[i].memory().is_null());

      signal_flags_buffers.emplace_back(
          executor, executor->AllocateArray<uint32_t>(
                        num_ranks * launch_dimensions.num_blocks()));
      TF_RET_CHECK(!signal_flags_buffers[i].memory().is_null());

      TF_RETURN_IF_ERROR(executor->SynchronousMemZero(
          signal_flags_buffers[i].memory_ptr(),
          signal_flags_buffers[i].memory().size()));

      TF_RETURN_IF_ERROR(streams[i]->Memcpy(local_input_buffers[i].memory_ptr(),
                                            input_data[i].data(),
                                            num_elements * sizeof(T)));

      remote_input_buffers_span.push_back(data_buffers[i].memory());
      signal_flags_buffers_span.push_back(signal_flags_buffers[i].memory());
    }

    for (int i = 0; i < num_ranks; ++i) {
      TF_RETURN_IF_ERROR(streams[i]->BlockHostUntilDone());
    }

    for (int i = 0; i < num_ranks; ++i) {
      auto active_context = executors[i]->Activate();
      TF_RETURN_IF_ERROR(RunAllReduceKernel(
          streams[i].get(), launch_dimensions,
          primitive_util::NativeToPrimitiveType<T>(),
          /*reduction_kind=*/reduction_kind,
          /*all_reduce_strategy=*/params_.all_reduce_strategy,
          /*remote_input_buffers=*/remote_input_buffers_span,
          // Memory is aliased for both input and output (similar to what nccl
          // would do).
          /*local_input_buffer=*/local_input_buffers[i].memory(),
          /*output_buffer=*/local_input_buffers[i].memory(),
          /*rank=*/RankId(i), /*num_ranks=*/num_ranks,
          /*num_elements=*/num_elements,
          /*signal_flags_buffers=*/signal_flags_buffers_span,
          /*signal_value=*/1));
    }

    for (int i = 0; i < num_ranks; ++i) {
      TF_RETURN_IF_ERROR(streams[i]->BlockHostUntilDone());
    }

    std::vector<Array<T>> results;

    for (int i = 0; i < num_ranks; ++i) {
      Array<T> output_results({num_elements});
      TF_RETURN_IF_ERROR(streams[i]->Memcpy(output_results.data(),
                                            local_input_buffers[i].memory(),
                                            num_elements * sizeof(T)));

      results.push_back(std::move(output_results));
    }

    return results;
  }

  int64_t num_elements() const { return params_.num_elements; }

 private:
  TestParams params_;
};

TEST_P(AllReduceKernelTest, KernelTestAddF32) {
  constexpr int64_t kNumRanks = 2;

  std::vector<se::StreamExecutor*> executors = {GetGpuExecutor(0),
                                                GetGpuExecutor(1)};

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires direct peer memory access between devices.";
  }

  Array<float> expected_output({num_elements()});
  std::vector<Array<float>> inputs;

  for (int i = 0; i < kNumRanks; ++i) {
    Array<float> input_data({num_elements()});
    input_data.FillRandom(0.0f, 10.0f, /*seed=*/i);

    expected_output.Each([&](absl::Span<const int64_t> indices, float* val) {
      *val += input_data(indices);
    });

    inputs.push_back(std::move(input_data));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto results, RunKernel<float>(executors, inputs, ReductionKind::SUM));

  for (int i = 0; i < kNumRanks; ++i) {
    EXPECT_EQ(results[i], expected_output);
  }
}

TEST_P(AllReduceKernelTest, KernelTestAddBF16) {
  constexpr int64_t kNumRanks = 2;

  std::vector<se::StreamExecutor*> executors = {GetGpuExecutor(0),
                                                GetGpuExecutor(1)};

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires direct peer memory access between devices.";
  }

  Array<bfloat16> expected_output({num_elements()});
  std::vector<Array<bfloat16>> inputs;

  for (int i = 0; i < kNumRanks; ++i) {
    Array<bfloat16> input_data({num_elements()});
    input_data.FillRandom(static_cast<bfloat16>(0.0f),
                          static_cast<bfloat16>(10.0f), /*seed=*/i);

    expected_output.Each([&](absl::Span<const int64_t> indices, bfloat16* val) {
      *val += input_data(indices);
    });

    inputs.push_back(std::move(input_data));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto results, RunKernel<bfloat16>(executors, inputs, ReductionKind::SUM));

  for (int i = 0; i < kNumRanks; ++i) {
    EXPECT_EQ(results[i], expected_output);
  }
}

TEST_P(AllReduceKernelTest, KernelTestOrPred) {
  constexpr int64_t kNumRanks = 2;

  std::vector<se::StreamExecutor*> executors = {GetGpuExecutor(0),
                                                GetGpuExecutor(1)};

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires direct peer memory access between devices.";
  }

  Array<bool> expected_output({num_elements()});
  std::vector<Array<bool>> inputs;

  for (int i = 0; i < kNumRanks; ++i) {
    Array<bool> input_data({num_elements()});
    input_data.FillRandomBool(/*seed=*/i);

    expected_output.Each([&](absl::Span<const int64_t> indices, bool* val) {
      *val |= input_data(indices);
    });

    inputs.push_back(std::move(input_data));
  }

  // There are no logical operations in all-reduce reduction kind, so OR is
  // simulated with MAX on uint8.
  TF_ASSERT_OK_AND_ASSIGN(
      auto results, RunKernel<bool>(executors, inputs, ReductionKind::MAX));

  for (int i = 0; i < kNumRanks; ++i) {
    EXPECT_EQ(results[i], expected_output);
  }
}

TEST_P(AllReduceKernelTest, KernelTestAddPred_Unsupported) {
  constexpr int64_t kNumRanks = 2;
  std::vector<se::StreamExecutor*> executors = {GetGpuExecutor(0),
                                                GetGpuExecutor(1)};

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires direct peer memory access between devices.";
  }

  Array<bool> expected_output({num_elements()});
  std::vector<Array<bool>> inputs(kNumRanks, Array<bool>({num_elements()}));

  auto results = RunKernel<bool>(executors, inputs, ReductionKind::SUM);
  EXPECT_THAT(results.status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(results.status().message(),
              ::testing::HasSubstr("AllReduce kernel is not supported"));
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceKernelTest, AllReduceKernelTest,
    ::testing::ConvertGenerator(
        ::testing::Combine(::testing::Values(AllReduceStrategy::kOneShot,
                                             AllReduceStrategy::kTwoShot),
                           ::testing::Values(128000, 124000)),
        [](const std::tuple<AllReduceStrategy, int64_t>& params) {
          return TestParams{std::get<0>(params), std::get<1>(params)};
        }),
    [](const ::testing::TestParamInfo<TestParams>& info) {
      return absl::StrFormat("%v_%d", info.param.all_reduce_strategy,
                             info.param.num_elements);
    });

class AllReduceHloTest : public HloHardwareIndependentTestBase {};

TEST_F(AllReduceHloTest, NullDeviceAssnWithHloRunner) {
  // xla::HloRunner passes a null device assignment to the XLA executable.
  // Test this returns an error gracefully.
  const char* const hlo_string = R"(
    HloModule module, replica_count=2

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY test {
      param = f32[1024] parameter(0)
      ROOT result = f32[1024] all-reduce(param), to_apply=add, replica_groups={{0,1}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloRunner runner(PlatformUtil::GetDefaultPlatform().value());
  Literal input = LiteralUtil::CreateR1<float>(std::vector<float>(1, 2));

  EXPECT_THAT(
      runner.Execute(std::move(module), {std::move(input)}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Device assignment is null, but must be specified when "
                    "running a collective thunk.")));
}

}  // namespace
}  // namespace xla::gpu
