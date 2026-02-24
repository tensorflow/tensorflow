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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::stream_executor::gpu::AllReduceStrategy;
using ::testing::HasSubstr;

se::StreamExecutor* GetGpuExecutor(int64_t device_ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(device_ordinal).value();
}

struct TestParams {
  AllReduceStrategy all_reduce_strategy;
  int64_t num_elements;
};

struct PtrFormatter {
  void operator()(std::string* out, const void* ptr) const {
    absl::StrAppend(out, absl::StrFormat("%p", ptr));
  }
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

    std::unique_ptr<se::gpu::MulticastMemory> multicast_memory;
    if (params_.all_reduce_strategy == AllReduceStrategy::kMultimem) {
      TF_ASSIGN_OR_RETURN(
          multicast_memory,
          dynamic_cast<se::gpu::GpuExecutor*>(executors[0])
              ->CreateMulticastMemory(num_elements * sizeof(T), num_ranks));

      for (int i = 0; i < num_ranks; ++i) {
        TF_RETURN_IF_ERROR(multicast_memory->SubscribeDevice(i));
      }
    }

    std::vector<std::unique_ptr<se::Stream>> streams;
    std::vector<se::DeviceAddressBase> allocated_buffers;
    std::vector<se::DeviceAddressBase> input_buffers;
    std::vector<se::DeviceAddressBase> output_buffers;
    std::vector<se::DeviceAddressBase> symmetric_input_buffers;
    std::vector<se::DeviceAddressBase> signal_flags_buffers;

    uint64_t input_size = num_elements * sizeof(T);
    uint64_t aligned_input_size =
        xla::RoundUpTo<uint64_t>(input_size, kXlaAllocatedBufferAlignBytes);
    uint64_t signal_size =
        num_ranks * launch_dimensions.num_blocks() * sizeof(int32_t);
    uint64_t aligned_signal_size =
        xla::RoundUpTo<uint64_t>(signal_size, kXlaAllocatedBufferAlignBytes);
    for (int i = 0; i < num_ranks; ++i) {
      auto* executor = executors[i];
      streams.push_back(executor->CreateStream().value());

      uint64_t total_size =
          /*input_buffer_size=*/aligned_input_size +
          /*symmetric_input_buffer_size=*/aligned_input_size +
          /*output_buffer_size=*/aligned_input_size +
          /*signal_buffer_size=*/aligned_signal_size;
      allocated_buffers.emplace_back(executor->AllocateArray<T>(
          total_size,
          static_cast<int64_t>(stream_executor::MemorySpace::kP2P)));
      input_buffers.emplace_back(
          allocated_buffers[i].GetByteSlice(0, aligned_input_size));
      TF_RET_CHECK(!input_buffers[i].is_null());

      symmetric_input_buffers.emplace_back(allocated_buffers[i].GetByteSlice(
          aligned_input_size, aligned_input_size));
      TF_RET_CHECK(!symmetric_input_buffers[i].is_null());

      output_buffers.emplace_back(allocated_buffers[i].GetByteSlice(
          2 * aligned_input_size, aligned_input_size));
      TF_RET_CHECK(!output_buffers[i].is_null());
      TF_RETURN_IF_ERROR(
          executor->SynchronousMemZero(&output_buffers[i], aligned_input_size));

      signal_flags_buffers.emplace_back(allocated_buffers[i].GetByteSlice(
          3 * aligned_input_size, aligned_signal_size));
      TF_RET_CHECK(!signal_flags_buffers[i].is_null());
      TF_RETURN_IF_ERROR(executor->SynchronousMemZero(&signal_flags_buffers[i],
                                                      aligned_signal_size));
      TF_RETURN_IF_ERROR(streams[i]->Memcpy(&input_buffers[i],
                                            input_data[i].data(), input_size));
      XLA_VLOG_DEVICE(1, i)
          << "Allocated buffer: " << allocated_buffers[i].opaque()
          << ", Input buffer: " << input_buffers[i].opaque()
          << ", Symmetric input buffer: " << symmetric_input_buffers[i].opaque()
          << ", Output buffer: " << output_buffers[i].opaque()
          << ", Signal buffer: " << signal_flags_buffers[i].opaque();
    }

    std::vector<se::DeviceAddressBase> metadata_buffers;
    for (int i = 0; i < num_ranks; ++i) {
      CollectiveKernelMetadata metadata;
      metadata.rank = i;
      std::vector<void*> param_to_peers_ptrs;

      if (params_.all_reduce_strategy == AllReduceStrategy::kMultimem) {
        // Multimem also need to have an output buffer.
        constexpr int kNumPeerParameters = 3;
        // Multimem needs to address input and output buffers on the peer
        // devices. Also an offset between the root exchanged pointer and
        // the multimem address space should be the same.
        for (int buffer_id = 0; buffer_id < kNumPeerParameters; ++buffer_id) {
          for (int rank = 0; rank < num_ranks; ++rank) {
            param_to_peers_ptrs.push_back(allocated_buffers[rank].opaque());
          }
        }

        se::gpu::GpuExecutor* gpu_executor =
            dynamic_cast<se::gpu::GpuExecutor*>(executors[i]);
        TF_RET_CHECK(gpu_executor != nullptr);
        TF_ASSIGN_OR_RETURN(
            void* mapped_memory,
            multicast_memory->MapMemory(allocated_buffers[i], gpu_executor));
        std::vector<void*> param_to_multimem_addresses =
            std::vector<void*>(kNumPeerParameters, mapped_memory);

        const size_t param_to_peers_size =
            sizeof(void*) * param_to_peers_ptrs.size();
        const size_t param_to_multimem_addresses_byte_size =
            sizeof(void*) * param_to_multimem_addresses.size();
        // First map from parameter to peer ptrs and then metadata.
        metadata_buffers.emplace_back(executors[i]->AllocateArray<uint64_t>(
            sizeof(CollectiveKernelMetadata) + param_to_peers_size +
            param_to_multimem_addresses_byte_size));

        se::DeviceAddressBase param_to_multimem_addresses_buffer =
            metadata_buffers[i].GetByteSlice(
                sizeof(CollectiveKernelMetadata) + param_to_peers_size,
                param_to_multimem_addresses_byte_size);
        metadata.param_to_multimem_addresses = reinterpret_cast<void**>(
            param_to_multimem_addresses_buffer.opaque());
        TF_RETURN_IF_ERROR(
            streams[i]->Memcpy(&param_to_multimem_addresses_buffer,
                               param_to_multimem_addresses.data(),
                               param_to_multimem_addresses_byte_size));
        XLA_VLOG_DEVICE(1, i)
            << "Constructed device state {"
            << " metadata rank: " << metadata.rank << ", param_to_peers: ("
            << absl::StrJoin(param_to_peers_ptrs, ", ", PtrFormatter{})
            << "), multimem_addresses: ("
            << absl::StrJoin(param_to_multimem_addresses, ", ", PtrFormatter{})
            << ")}";
      } else {
        for (const se::DeviceAddressBase& input_buffer : input_buffers) {
          param_to_peers_ptrs.push_back(input_buffer.opaque());
        }
        for (const se::DeviceAddressBase& signal_flags_buffer :
             signal_flags_buffers) {
          param_to_peers_ptrs.push_back(signal_flags_buffer.opaque());
        }
        metadata_buffers.emplace_back(executors[i]->AllocateArray<uint64_t>(
            sizeof(CollectiveKernelMetadata) +
            param_to_peers_ptrs.size() * sizeof(void*)));

        XLA_VLOG_DEVICE(1, executors[i]->device_ordinal())
            << "Constructed device state {"
            << " metadata rank: " << metadata.rank << ", param_to_peers: ("
            << absl::StrJoin(param_to_peers_ptrs, ", ", PtrFormatter{}) << ")}";
      }

      const size_t param_to_peers_size_bytes =
          param_to_peers_ptrs.size() * sizeof(void*);
      se::DeviceAddressBase param_to_peers_ptrs_buffer =
          metadata_buffers[i].GetByteSlice(sizeof(CollectiveKernelMetadata),
                                           param_to_peers_size_bytes);
      metadata.param_to_peers =
          reinterpret_cast<void**>(param_to_peers_ptrs_buffer.opaque());
      TF_RETURN_IF_ERROR(streams[i]->Memcpy(&param_to_peers_ptrs_buffer,
                                            param_to_peers_ptrs.data(),
                                            param_to_peers_size_bytes));
      TF_RETURN_IF_ERROR(streams[i]->Memcpy(&metadata_buffers[i], &metadata,
                                            sizeof(CollectiveKernelMetadata)));
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
          /*symmetric_input_buffer=*/symmetric_input_buffers[i],
          // Memory is aliased for both input and output (similar to what nccl
          // would do).
          /*local_input_buffer=*/input_buffers[i],
          /*output_buffer=*/output_buffers[i],
          /*rank=*/RankId(i), /*num_ranks=*/num_ranks,
          /*num_elements=*/num_elements,
          /*symmetric_signal_buffer=*/signal_flags_buffers[i],
          /*signal_value=*/1,
          /*metadata=*/metadata_buffers[i]));
    }

    for (int i = 0; i < num_ranks; ++i) {
      TF_RETURN_IF_ERROR(streams[i]->BlockHostUntilDone());
    }

    std::vector<Array<T>> results;
    for (int i = 0; i < num_ranks; ++i) {
      Array<T> output_results({num_elements});
      TF_RETURN_IF_ERROR(streams[i]->Memcpy(
          output_results.data(), output_buffers[i], num_elements * sizeof(T)));

      results.push_back(std::move(output_results));
    }

    return results;
  }

  int64_t num_elements() const { return params_.num_elements; }

  AllReduceStrategy strategy() const { return params_.all_reduce_strategy; }

 private:
  TestParams params_;
};

TEST_P(AllReduceKernelTest, KernelTestAddF32) {
  constexpr int64_t kNumRanks = 2;

  std::vector<se::StreamExecutor*> executors = {GetGpuExecutor(0),
                                                GetGpuExecutor(1)};
  if (strategy() == AllReduceStrategy::kMultimem &&
      !dynamic_cast<se::gpu::GpuExecutor*>(executors[0])
           ->is_multicast_supported()) {
    GTEST_SKIP() << "Multimem not supported on this device.";
  }

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

  const Literal expected_output_literal =
      LiteralUtil::CreateFromArray<float>(expected_output);
  for (int i = 0; i < kNumRanks; ++i) {
    const Literal actual_output_literal =
        LiteralUtil::CreateFromArray<float>(results[i]);
    EXPECT_TRUE(
        LiteralTestUtil::Equal(expected_output_literal, actual_output_literal));
  }
}

TEST_P(AllReduceKernelTest, KernelTestAddBF16) {
  if (strategy() == AllReduceStrategy::kMultimem) {
    GTEST_SKIP() << "Multimem does not support BF16.";
  }
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
  if (strategy() == AllReduceStrategy::kMultimem) {
    GTEST_SKIP() << "Multimem does not support predicates.";
  }
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
  if (strategy() == AllReduceStrategy::kMultimem) {
    GTEST_SKIP() << "Multimem does not support predicates.";
  }
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
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(results.status().message(),
              ::testing::HasSubstr("AllReduce kernel is not supported"));
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceKernelTest, AllReduceKernelTest,
    ::testing::ConvertGenerator(
        ::testing::Combine(::testing::Values(AllReduceStrategy::kOneShot,
                                             AllReduceStrategy::kTwoShot,
                                             AllReduceStrategy::kMultimem),
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
      absl_testing::StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "Collective parameters and device assignment are required")));
}

}  // namespace
}  // namespace xla::gpu
