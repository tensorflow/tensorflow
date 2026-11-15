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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_requests.h"
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
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
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
  CommandBufferThunk cmd_buffer_thunk(std::move(executor), Thunk::ThunkInfo(),
                                      /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations(
      {d_data_in, d_data_out, d_indices_out, d_scratch}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr,
      nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      absl::Span<const BufferAllocation::Index>());

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = stream_executor;
  init_params.buffer_allocations = &allocations;
  init_params.stream = stream.get();
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

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// Fixture for testing scratch buffer reuse between Phase 1 (radix_topk) and
// Phase 2 (segmented_sort_by_key).
//
// For float with (batch=128, n=16384, k=1024), RAFT selects kRadix8bits:
// - Phase 1 (radix_topk_one_block) allocates 1.00 MB (bufs).
// - Phase 2 (segmented_sort_by_key) allocates ~2.00 MB (offsets + out_inds +
//   out_dists + CUB temp storage).
// Cumulative allocations are ~3.00 MB, whereas max(Phase 1, Phase 2) is
// ~2.00 MB. A 2.5 MB scratch buffer succeeds only if Phase 1's memory is
// reclaimed before Phase 2 allocates.
class RaftSelectKScratchReuseTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kBatch = 128;
  static constexpr uint32_t kN = 16384;
  static constexpr uint32_t kK = 1024;
  static constexpr size_t kScratchBytes = 2560 * 1024;  // 2.5 MB

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(std::string platform_name,
                         PlatformUtil::CanonicalPlatformName("gpu"));
    ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                         se::PlatformManager::PlatformWithName(
                             absl::AsciiStrToUpper(platform_name)));
    ASSERT_OK_AND_ASSIGN(stream_executor_, platform->ExecutorForDevice(0));

    device_ordinal_ = stream_executor_->device_ordinal();
    allocator_ =
        std::make_unique<se::StreamExecutorAddressAllocator>(stream_executor_);
    ASSERT_OK_AND_ASSIGN(stream_, stream_executor_->CreateStream());

    topk_.reserve(kN);
    append_unique_numbers<float>(kN, topk_);

    std::mt19937 gen(42);
    h_data_in_.resize(static_cast<size_t>(kBatch) * kN);
    for (uint32_t j = 0; j < kBatch; ++j) {
      std::shuffle(topk_.begin(), topk_.end(), gen);
      absl::c_copy(topk_, h_data_in_.begin() + static_cast<size_t>(j) * kN);
    }

    absl::c_sort(topk_, std::greater<float>());
    topk_.resize(kK);

    ASSERT_OK_AND_ASSIGN(
        d_data_in_raw_,
        allocator_->Allocate(device_ordinal_,
                             h_data_in_.size() * sizeof(float)));
    ASSERT_OK_AND_ASSIGN(
        d_data_out_raw_,
        allocator_->Allocate(device_ordinal_,
                             static_cast<size_t>(kBatch) * kK * sizeof(float)));
    ASSERT_OK_AND_ASSIGN(
        d_indices_out_raw_,
        allocator_->Allocate(device_ordinal_, static_cast<size_t>(kBatch) * kK *
                                                  sizeof(uint32_t)));
    ASSERT_OK_AND_ASSIGN(d_scratch_,
                         allocator_->Allocate(device_ordinal_, kScratchBytes));

    d_data_in_ = se::DeviceAddress<float>(d_data_in_raw_.cref());
    d_data_out_ = se::DeviceAddress<float>(d_data_out_raw_.cref());
    d_indices_out_ = se::DeviceAddress<uint32_t>(d_indices_out_raw_.cref());
    ASSERT_THAT(
        stream_->MemcpyH2D(absl::MakeConstSpan(h_data_in_), &d_data_in_),
        IsOk());
    ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
  }

  void VerifyTopKResults() {
    std::vector<float> h_data_out(static_cast<size_t>(kBatch) * kK);
    std::vector<uint32_t> h_indices_out(static_cast<size_t>(kBatch) * kK);
    ASSERT_THAT(stream_->MemcpyD2H(d_data_out_, absl::MakeSpan(h_data_out)),
                IsOk());
    ASSERT_THAT(
        stream_->MemcpyD2H(d_indices_out_, absl::MakeSpan(h_indices_out)),
        IsOk());
    ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());

    for (uint32_t j = 0; j < kBatch; ++j) {
      for (uint32_t i = 0; i < kK; ++i) {
        size_t out_offset = static_cast<size_t>(j) * kK + i;
        ASSERT_EQ(h_data_out[out_offset], topk_[i])
            << "batch=" << j << " i=" << i;
        uint32_t idx = h_indices_out[out_offset];
        ASSERT_LT(idx, kN);
        ASSERT_EQ(h_data_in_[static_cast<size_t>(j) * kN + idx], topk_[i])
            << "batch=" << j << " i=" << i;
      }
    }
  }

  se::StreamExecutor* stream_executor_ = nullptr;
  int device_ordinal_ = 0;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator_;
  std::unique_ptr<se::Stream> stream_;
  std::vector<float> topk_;
  std::vector<float> h_data_in_;
  se::ScopedDeviceAddress<uint8_t> d_data_in_raw_;
  se::ScopedDeviceAddress<uint8_t> d_data_out_raw_;
  se::ScopedDeviceAddress<uint8_t> d_indices_out_raw_;
  se::ScopedDeviceAddress<uint8_t> d_scratch_;
  se::DeviceAddress<float> d_data_in_;
  se::DeviceAddress<float> d_data_out_;
  se::DeviceAddress<uint32_t> d_indices_out_;
};

TEST_F(RaftSelectKScratchReuseTest, ReusesScratchAcrossRadixAndSortPhases) {
  // 1. 512 KB fails in Phase 1 (< 1.00 MB).
  se::DeviceAddressBase scratch_512kb =
      d_scratch_.cref().GetByteSlice(0, 512 * 1024);
  EXPECT_THAT(select_k_exec<float>(
                  device_ordinal_, allocator_.get(), stream_.get(), d_data_in_,
                  d_data_out_, d_indices_out_, kBatch, kN, kK, scratch_512kb),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("scratch capacity exceeded")));

  // 2. 1.5 MB passes Phase 1 (1.00 MB), rewinds to 0, and fails midway through
  // Phase 2 (~2.00 MB > 1.5 MB), exercising exception stack unwinding while
  // multiple allocations are live.
  se::DeviceAddressBase scratch_1536kb =
      d_scratch_.cref().GetByteSlice(0, 1536 * 1024);
  EXPECT_THAT(select_k_exec<float>(
                  device_ordinal_, allocator_.get(), stream_.get(), d_data_in_,
                  d_data_out_, d_indices_out_, kBatch, kN, kK, scratch_1536kb),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("scratch capacity exceeded")));

  // 3. 2.5 MB (> max(Phase 1, Phase 2) ~ 2.00 MB, < Phase 1 + Phase 2 ~ 3.00
  // MB) succeeds on the same stream and produces exact Top-K outputs. Zero
  // output buffers first since Sub-case 2's Phase 1 wrote unsorted candidates.
  ASSERT_THAT(stream_->MemZero(&d_data_out_, d_data_out_.size()), IsOk());
  ASSERT_THAT(stream_->MemZero(&d_indices_out_, d_indices_out_.size()), IsOk());
  ASSERT_THAT(
      select_k_exec<float>(device_ordinal_, allocator_.get(), stream_.get(),
                           d_data_in_, d_data_out_, d_indices_out_, kBatch, kN,
                           kK, d_scratch_.cref()),
      IsOk());
  ASSERT_NO_FATAL_FAILURE(VerifyTopKResults());
}

TEST_F(RaftSelectKScratchReuseTest, CommandBufferReusesScratchAcrossPhases) {
  const se::DeviceDescription& device_desc =
      stream_executor_->GetDeviceDescription();
  if (device_desc.gpu_compute_capability().cuda_compute_capability() ==
          nullptr ||
      std::min(device_desc.driver_version(),
               device_desc.compile_time_toolkit_version()) <
          se::SemanticVersion(12, 3, 0)) {
    GTEST_SKIP() << "CUDA graph tracing requires CUDA 12.3 or newer";
  }

  BufferAllocation alloc_in(/*index=*/0, d_data_in_.size(), /*color=*/0);
  BufferAllocation alloc_out_val(/*index=*/1, d_data_out_.size(), /*color=*/0);
  BufferAllocation alloc_out_idx(/*index=*/2, d_indices_out_.size(),
                                 /*color=*/0);
  BufferAllocation alloc_scratch(/*index=*/3, d_scratch_.cref().size(),
                                 /*color=*/0);

  BufferAllocation::Slice slice_in(&alloc_in, 0, d_data_in_.size());
  BufferAllocation::Slice slice_out_val(&alloc_out_val, 0, d_data_out_.size());
  BufferAllocation::Slice slice_out_idx(&alloc_out_idx, 0,
                                        d_indices_out_.size());
  BufferAllocation::Slice slice_scratch(&alloc_scratch, 0,
                                        d_scratch_.cref().size());

  std::vector<NullableShapedSlice> operands = {
      ShapedSlice{slice_in, ShapeUtil::MakeShape(F32, {kBatch, kN})},
  };
  std::vector<NullableShapedSlice> results = {
      ShapedSlice{slice_out_val, ShapeUtil::MakeShape(F32, {kBatch, kK})},
      ShapedSlice{slice_out_idx, ShapeUtil::MakeShape(S32, {kBatch, kK})},
      ShapedSlice{slice_scratch,
                  ShapeUtil::MakeShape(
                      U8, {static_cast<int64_t>(d_scratch_.cref().size())})},
  };

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> custom_call_thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(), std::string(kTopKCustomCallTarget),
          std::move(operands), std::move(results), /*attributes=*/{},
          /*called_computation=*/nullptr,
          stream_executor_->GetPlatform()->Name(),
          stream_executor_->GetDeviceDescription().gpu_compute_capability()));

  CommandSequence commands;
  commands.Append(std::move(custom_call_thunk));
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(
                           std::move(commands),
                           CommandExecutor::SynchronizationMode::kSerialize));
  CommandBufferThunk cmd_buffer_thunk(std::move(executor), Thunk::ThunkInfo(),
                                      /*devices_in_process=*/1);

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations(
      {d_data_in_, d_data_out_, d_indices_out_, d_scratch_.cref()}, 0,
      allocator_.get());
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream_.get(), stream_.get(), nullptr, nullptr,
      nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      absl::Span<const BufferAllocation::Index>());

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = stream_executor_;
  init_params.buffer_allocations = &allocations;
  init_params.stream = stream_.get();
  ASSERT_THAT(cmd_buffer_thunk.Initialize(init_params), IsOk());

  // Record + execute, then replay on zeroed output buffers.
  ASSERT_THAT(cmd_buffer_thunk.ExecuteOnStream(params), IsOk());
  ASSERT_NO_FATAL_FAILURE(VerifyTopKResults());

  ASSERT_THAT(stream_->MemZero(&d_data_out_, d_data_out_.size()), IsOk());
  ASSERT_THAT(stream_->MemZero(&d_indices_out_, d_indices_out_.size()), IsOk());
  ASSERT_THAT(cmd_buffer_thunk.ExecuteOnStream(params), IsOk());
  ASSERT_NO_FATAL_FAILURE(VerifyTopKResults());
}

}  // namespace
}  // namespace xla::gpu
