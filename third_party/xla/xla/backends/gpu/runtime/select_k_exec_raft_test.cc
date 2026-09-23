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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// Returns the first GPU StreamExecutor
se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  CHECK(platform != nullptr);
  CHECK_OK(platform->ExecutorForDevice(0));
  return platform->ExecutorForDevice(0).value();
}

// Trait: map Data type to Mask type and a default kStartBits
template <typename T>
struct MaskFor;

template <>
struct MaskFor<float> {
  using type = uint32_t;
  static constexpr type kStartBits = 0x3C000000;  // float32: 1/128
};

template <>
struct MaskFor<::xla::bfloat16> {
  using type = uint16_t;
  static constexpr type kStartBits = 0x3C00;  // bfloat16: 1/128
};

template <>
struct MaskFor<uint64_t> {
  using type = uint64_t;
  static constexpr type kStartBits = 1000;  // uint64_t: arbitrary point
};

// Fills vector with unique values using bit patterns starting from kStartBits
template <typename T>
void append_unique_numbers(size_t count, std::vector<T>& arr) {
  using Traits = MaskFor<T>;
  using MaskT = typename Traits::type;
  MaskT bits = Traits::kStartBits;

  for (size_t i = 0; i < count; ++i, ++bits) {
    T val = absl::bit_cast<T>(bits);
    arr.push_back(val);
  }
}

}  // namespace

// Template test function for raft select_k
template <typename T>
void RunSelectKTest() {
  se::StreamExecutor* stream_executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());
  int device_ordinal = stream_executor->device_ordinal();
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);

  std::uint32_t batch = 4;
  std::uint32_t n = 4096;
  std::uint32_t k = 32;
  absl::BitGen gen;

  // Prepare unique values for Top-K testing
  std::vector<T> topk;
  topk.reserve(n);
  append_unique_numbers<T>(n, topk);

  // Populate input matrix (batch x n) with shuffled topk values
  std::vector<T> h_data_in(batch * n);
  for (int j = 0; j < batch; ++j) {
    std::shuffle(topk.begin(), topk.end(), gen);
    absl::c_copy(topk, h_data_in.begin() + j * n);
  }

  // Compute golden Top-K values for verification
  std::sort(topk.begin(), topk.end(), std::greater<T>());
  topk.resize(k);

  // Allocate device memory for input and outputs
  se::DeviceAddress<T> d_data_in =
      stream_executor->AllocateArray<T>(batch * n, 0);
  se::DeviceAddress<T> d_data_out =
      stream_executor->AllocateArray<T>(batch * k, 0);
  se::DeviceAddress<uint32_t> d_indices_out =
      stream_executor->AllocateArray<uint32_t>(batch * k, 0);
  se::DeviceAddress<uint8_t> d_scratch =
      stream_executor->AllocateArray<uint8_t>(32 * 1024 * 1024, 0);

  // Copy host to device
  ASSERT_OK(stream->MemcpyH2D(absl::Span<const T>(h_data_in), &d_data_in));

  // Run raft select_k
  ASSERT_OK(select_k_exec<T>(device_ordinal, &allocator, stream.get(),
                             d_data_in, d_data_out, d_indices_out, batch, n, k,
                             d_scratch));

  // Copy results back to host
  std::vector<T> h_data_out(batch * k);
  std::vector<uint32_t> h_indices_out(batch * k);
  ASSERT_OK(stream->MemcpyD2H(d_data_out, absl::Span<T>(h_data_out)));
  ASSERT_OK(
      stream->MemcpyD2H(d_indices_out, absl::Span<uint32_t>(h_indices_out)));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Verify Top-K values and corresponding indices
  for (int j = 0; j < batch; ++j) {
    for (int i = 0; i < k; ++i) {
      EXPECT_EQ(h_data_out[j * k + i], topk[i]) << "batch=" << j << " i=" << i;
      auto idx = h_indices_out[j * k + i];
      EXPECT_EQ(h_data_in[j * n + idx], topk[i]) << "batch=" << j << " i=" << i;
    }
  }
}

TEST(RaftSelectKExecTest, SelectKFloat) { RunSelectKTest<float>(); }

TEST(RaftSelectKExecTest, SelectKBFloat16) {
  RunSelectKTest<::xla::bfloat16>();
}

TEST(RaftSelectKExecTest, SelectKUint64) { RunSelectKTest<uint64_t>(); }

TEST(RaftSelectKExecTest, CommandBuffer) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);

  std::uint32_t batch = 4;
  std::uint32_t n = 4096;
  std::uint32_t k = 32;
  absl::BitGen gen;

  std::vector<float> topk;
  topk.reserve(n);
  append_unique_numbers<float>(n, topk);

  std::vector<float> h_data_in(batch * n);
  for (int j = 0; j < batch; ++j) {
    std::shuffle(topk.begin(), topk.end(), gen);
    absl::c_copy(topk, h_data_in.begin() + j * n);
  }

  std::sort(topk.begin(), topk.end(), std::greater<float>());
  topk.resize(k);

  se::DeviceAddress<float> d_data_in =
      stream_executor->AllocateArray<float>(batch * n, 0);
  se::DeviceAddress<float> d_data_out =
      stream_executor->AllocateArray<float>(batch * k, 0);
  se::DeviceAddress<uint32_t> d_indices_out =
      stream_executor->AllocateArray<uint32_t>(batch * k, 0);
  se::DeviceAddress<uint8_t> d_scratch =
      stream_executor->AllocateArray<uint8_t>(32 * 1024 * 1024, 0);

  ASSERT_OK(stream->MemcpyH2D(absl::Span<const float>(h_data_in), &d_data_in));

  BufferAllocation alloc_in(/*index=*/0, d_data_in.size(), /*color=*/0);
  BufferAllocation alloc_out_val(/*index=*/1, d_data_out.size(), /*color=*/0);
  BufferAllocation alloc_out_idx(/*index=*/2, d_indices_out.size(),
                                 /*color=*/0);
  BufferAllocation alloc_scratch(/*index=*/3, d_scratch.size(), /*color=*/0);

  BufferAllocation::Slice slice_in(&alloc_in, 0, d_data_in.size());
  BufferAllocation::Slice slice_out_val(&alloc_out_val, 0, d_data_out.size());
  BufferAllocation::Slice slice_out_idx(&alloc_out_idx, 0,
                                        d_indices_out.size());
  BufferAllocation::Slice slice_scratch(&alloc_scratch, 0, d_scratch.size());

  std::vector<NullableShapedSlice> operands = {
      ShapedSlice{slice_in, ShapeUtil::MakeShape(F32, {batch, n})},
  };
  std::vector<NullableShapedSlice> results = {
      ShapedSlice{slice_out_val, ShapeUtil::MakeShape(F32, {batch, k})},
      ShapedSlice{slice_out_idx, ShapeUtil::MakeShape(S32, {batch, k})},
      ShapedSlice{
          slice_scratch,
          ShapeUtil::MakeShape(U8, {static_cast<int64_t>(d_scratch.size())})},
  };

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> custom_call_thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(), std::string(kTopKCustomCallTarget),
          std::move(operands), std::move(results), /*attributes=*/{},
          /*called_computation=*/nullptr,
          stream_executor->GetPlatform()->Name(),
          stream_executor->GetDeviceDescription().gpu_compute_capability()));

  CommandSequence commands;
  commands.Append(std::move(custom_call_thunk));
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(
                           std::move(commands),
                           CommandExecutor::SynchronizationMode::kSerialize));
  CommandBufferThunk cmd_buffer_thunk(std::move(executor), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations(
      {d_data_in, d_data_out, d_indices_out, d_scratch}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr,
      nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      absl::Span<const BufferAllocation::Index>());

  Thunk::InitializeParams init_params;
  init_params.executor = stream_executor;
  init_params.stream = stream.get();
  init_params.buffer_allocations = &allocations;
  ASSERT_OK(cmd_buffer_thunk.Initialize(init_params));

  auto verify_results = [&]() {
    std::vector<float> h_data_out(batch * k);
    std::vector<uint32_t> h_indices_out(batch * k);
    ASSERT_OK(stream->MemcpyD2H(d_data_out, absl::Span<float>(h_data_out)));
    ASSERT_OK(
        stream->MemcpyD2H(d_indices_out, absl::Span<uint32_t>(h_indices_out)));
    ASSERT_OK(stream->BlockHostUntilDone());

    for (int j = 0; j < batch; ++j) {
      for (int i = 0; i < k; ++i) {
        EXPECT_EQ(h_data_out[j * k + i], topk[i])
            << "batch=" << j << " i=" << i;
        auto idx = h_indices_out[j * k + i];
        EXPECT_EQ(h_data_in[j * n + idx], topk[i])
            << "batch=" << j << " i=" << i;
      }
    }
  };

  // First execution: records the command buffer and executes it.
  ASSERT_OK(cmd_buffer_thunk.ExecuteOnStream(params));
  verify_results();

  // Zero output buffers and run a second time to verify command buffer replay.
  ASSERT_OK(stream->MemZero(&d_data_out, d_data_out.size()));
  ASSERT_OK(stream->MemZero(&d_indices_out, d_indices_out.size()));
  ASSERT_OK(cmd_buffer_thunk.ExecuteOnStream(params));
  verify_results();
}

}  // namespace xla::gpu
