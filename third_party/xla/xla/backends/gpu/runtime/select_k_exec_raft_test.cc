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
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/buffer_allocations.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/raft_select_k_utils.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_test_utils.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/platform_util.h"
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
  // Starts at the smallest positive normal so up to 32,512 unique finite
  // values can be generated without reaching +Inf (0x7F80).
  static constexpr type kStartBits = 0x0080;
};

template <>
struct MaskFor<uint64_t> {
  using type = uint64_t;
  // Crosses 2^63 for n >= 2048 to verify unsigned ordering across the sign bit.
  static constexpr type kStartBits = 0x7FFFFFFFFFFFFC00ULL;
};

template <>
struct MaskFor<uint32_t> {
  using type = uint32_t;
  // Crosses 2^31 for n >= 2048 to verify unsigned ordering across the sign bit.
  static constexpr type kStartBits = 0x7FFFFC00u;
};

// Fills vector with unique values using bit patterns starting from kStartBits
template <typename T>
void append_unique_numbers(size_t count, std::vector<T>& arr) {
  using Traits = MaskFor<T>;
  using MaskT = typename Traits::type;
  MaskT bits = Traits::kStartBits;

  for (size_t i = 0; i < count; ++i) {
    if constexpr (std::is_unsigned_v<T>) {
      if (i == 0) {
        // Include the minimum valid non-zero key (0 is reserved as RAFT
        // WarpSort's dummy sentinel for select_min=false).
        arr.push_back(T{1});
        continue;
      }
    }
    T val = absl::bit_cast<T>(bits);
    arr.push_back(val);
    ++bits;
  }
}

template <typename T>
absl::StatusOr<int64_t> EstimateScratchBytes(uint32_t batch, uint32_t n,
                                             uint32_t k, int sm_count) {
  if constexpr (std::is_same_v<T, float>) {
    return EstimateRaftSelectKScratchBytesFloat(
        batch, n, k, sm_count, RaftScratchPhaseMode::kMaxAcrossPhases);
  } else if constexpr (std::is_same_v<T, ::xla::bfloat16>) {
    return EstimateRaftSelectKScratchBytesBFloat16(
        batch, n, k, sm_count, RaftScratchPhaseMode::kMaxAcrossPhases);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return EstimateRaftSelectKScratchBytesUInt64(
        batch, n, k, sm_count, RaftScratchPhaseMode::kMaxAcrossPhases);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return EstimateRaftSelectKScratchBytesUInt32(
        batch, n, k, sm_count, RaftScratchPhaseMode::kMaxAcrossPhases);
  } else {
    static_assert(!std::is_same_v<T, T>,
                  "Unsupported type for EstimateScratchBytes");
  }
}

struct SelectKShape {
  uint32_t batch;
  uint32_t n;
  uint32_t k;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SelectKShape& shape) {
    sink.Append(absl::StrCat("{batch=", shape.batch, ", n=", shape.n,
                             ", k=", shape.k, "}"));
  }
};

template <typename T>
struct SelectKTestCase {
  SelectKShape shape;
  se::StreamExecutor* stream_executor = nullptr;
  int device_ordinal = 0;
  std::unique_ptr<se::Stream> stream;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator;
  int64_t scratch_bytes = 0;
  std::vector<T> h_data_in;
  std::vector<T> golden_topk;
  se::ScopedDeviceAddress<uint8_t> d_data_in_raw;
  se::ScopedDeviceAddress<uint8_t> d_data_out_raw;
  se::ScopedDeviceAddress<uint8_t> d_indices_out_raw;
  se::ScopedDeviceAddress<uint8_t> d_scratch;

  se::DeviceAddress<T> d_data_in() const {
    return se::DeviceAddress<T>(d_data_in_raw.cref());
  }
  se::DeviceAddress<T> d_data_out() const {
    return se::DeviceAddress<T>(d_data_out_raw.cref());
  }
  se::DeviceAddress<uint32_t> d_indices_out() const {
    return se::DeviceAddress<uint32_t>(d_indices_out_raw.cref());
  }
};

template <typename T>
absl::StatusOr<SelectKTestCase<T>> CreateSelectKTestCase(
    const SelectKShape& shape) {
  SelectKTestCase<T> test_case;
  test_case.shape = shape;
  test_case.stream_executor = GpuExecutor();
  test_case.device_ordinal = test_case.stream_executor->device_ordinal();
  ABSL_ASSIGN_OR_RETURN(test_case.stream, test_case.stream_executor->CreateStream());
  test_case.allocator = std::make_unique<se::StreamExecutorAddressAllocator>(
      test_case.stream_executor);

  const se::DeviceDescription& device_desc =
      test_case.stream_executor->GetDeviceDescription();
  const int sm_count =
      device_desc.gpu_compute_capability().cuda_compute_capability() != nullptr
          ? GetRaftSelectKMaxSmCount(
                *device_desc.gpu_compute_capability().cuda_compute_capability(),
                device_desc.core_count())
          : kDefaultRaftSelectKMaxSmCount;

  ABSL_ASSIGN_OR_RETURN(
      test_case.scratch_bytes,
      EstimateScratchBytes<T>(shape.batch, shape.n, shape.k, sm_count));

  // Prepare unique values for Top-K testing.
  test_case.golden_topk.reserve(shape.n);
  append_unique_numbers<T>(shape.n, test_case.golden_topk);

  // Populate input matrix (batch x n) with shuffled unique values.
  std::mt19937 gen(42);
  test_case.h_data_in.resize(static_cast<size_t>(shape.batch) * shape.n);
  for (uint32_t j = 0; j < shape.batch; ++j) {
    absl::c_shuffle(test_case.golden_topk, gen);
    absl::c_copy(test_case.golden_topk, test_case.h_data_in.begin() +
                                            static_cast<size_t>(j) * shape.n);
  }

  // Compute golden Top-K values for verification.
  absl::c_sort(test_case.golden_topk, std::greater<T>());
  test_case.golden_topk.resize(shape.k);

  // Allocate device memory for input, outputs, and estimated scratch buffer.
  ABSL_ASSIGN_OR_RETURN(
      test_case.d_data_in_raw,
      test_case.allocator->Allocate(test_case.device_ordinal,
                                    test_case.h_data_in.size() * sizeof(T)));
  ABSL_ASSIGN_OR_RETURN(test_case.d_data_out_raw,
                   test_case.allocator->Allocate(
                       test_case.device_ordinal,
                       static_cast<size_t>(shape.batch) * shape.k * sizeof(T)));
  ABSL_ASSIGN_OR_RETURN(
      test_case.d_indices_out_raw,
      test_case.allocator->Allocate(
          test_case.device_ordinal,
          static_cast<size_t>(shape.batch) * shape.k * sizeof(uint32_t)));
  ABSL_ASSIGN_OR_RETURN(test_case.d_scratch,
                   test_case.allocator->Allocate(test_case.device_ordinal,
                                                 test_case.scratch_bytes));

  se::DeviceAddress<T> d_data_in = test_case.d_data_in();
  ABSL_RETURN_IF_ERROR(test_case.stream->MemcpyH2D(
      absl::Span<const T>(test_case.h_data_in), &d_data_in));
  ABSL_RETURN_IF_ERROR(test_case.stream->BlockHostUntilDone());
  return test_case;
}

template <typename T>
void VerifySelectKResults(const SelectKTestCase<T>& test_case) {
  const uint32_t batch = test_case.shape.batch;
  const uint32_t n = test_case.shape.n;
  const uint32_t k = test_case.shape.k;

  std::vector<T> h_data_out(static_cast<size_t>(batch) * k);
  std::vector<uint32_t> h_indices_out(static_cast<size_t>(batch) * k);
  ASSERT_OK(test_case.stream->MemcpyD2H(test_case.d_data_out(),
                                        absl::Span<T>(h_data_out)));
  ASSERT_OK(test_case.stream->MemcpyD2H(test_case.d_indices_out(),
                                        absl::Span<uint32_t>(h_indices_out)));
  ASSERT_OK(test_case.stream->BlockHostUntilDone());

  for (uint32_t j = 0; j < batch; ++j) {
    for (uint32_t i = 0; i < k; ++i) {
      size_t out_offset = static_cast<size_t>(j) * k + i;
      EXPECT_EQ(h_data_out[out_offset], test_case.golden_topk[i])
          << "batch=" << j << " i=" << i;
      uint32_t idx = h_indices_out[out_offset];
      ASSERT_LT(idx, n) << "batch=" << j << " i=" << i;
      EXPECT_EQ(test_case.h_data_in[static_cast<size_t>(j) * n + idx],
                test_case.golden_topk[i])
          << "batch=" << j << " i=" << i;
    }
  }
}

// Template test function for raft select_k
template <typename T>
void RunSelectKTest(const SelectKShape& shape) {
  ASSERT_OK_AND_ASSIGN(SelectKTestCase<T> test_case,
                       CreateSelectKTestCase<T>(shape));

  ASSERT_OK(select_k_exec<T>(test_case.device_ordinal,
                             test_case.allocator.get(), test_case.stream.get(),
                             test_case.d_data_in(), test_case.d_data_out(),
                             test_case.d_indices_out(), shape.batch, shape.n,
                             shape.k, test_case.d_scratch.cref()));

  ASSERT_NO_FATAL_FAILURE(VerifySelectKResults(test_case));
}

class RaftSelectKShapeTest : public testing::TestWithParam<SelectKShape> {};

TEST_P(RaftSelectKShapeTest, SelectKFloat) {
  RunSelectKTest<float>(GetParam());
}

TEST_P(RaftSelectKShapeTest, SelectKBFloat16) {
  RunSelectKTest<::xla::bfloat16>(GetParam());
}

TEST_P(RaftSelectKShapeTest, SelectKUint64) {
  RunSelectKTest<uint64_t>(GetParam());
}

TEST_P(RaftSelectKShapeTest, SelectKUint32) {
  RunSelectKTest<uint32_t>(GetParam());
}

// Covers WarpSort (kWarpImmediate including k == n so minimum non-zero
// unsigned keys land in top-k, kWarpFiltered, kWarpDistributedShm) and both
// one-block (n <= 16384) and multi-block (n > 16384) Radix paths: kRadix8bits,
// kRadix11bits (float/uint64_t/uint32_t), kRadix11bitsExtraPass (bfloat16).
INSTANTIATE_TEST_SUITE_P(
    Shapes, RaftSelectKShapeTest,
    testing::Values(SelectKShape{4, 32, 32}, SelectKShape{4, 4096, 32},
                    SelectKShape{64, 2048, 200}, SelectKShape{1024, 2048, 32},
                    SelectKShape{4, 4096, 512}, SelectKShape{300, 4096, 512},
                    SelectKShape{4, 28672, 500}, SelectKShape{8, 28672, 500}),
    [](const testing::TestParamInfo<SelectKShape>& info) {
      return absl::StrCat("B", info.param.batch, "_N", info.param.n, "_K",
                          info.param.k);
    });

template <typename T>
void RunCommandBufferTest() {
  const se::DeviceDescription& device_desc =
      GpuExecutor()->GetDeviceDescription();
  if (device_desc.gpu_compute_capability().cuda_compute_capability() ==
          nullptr ||
      std::min(device_desc.driver_version(),
               device_desc.compile_time_toolkit_version()) <
          se::SemanticVersion(12, 3, 0)) {
    GTEST_SKIP() << "CUDA graph tracing requires CUDA 12.3 or newer";
  }

  constexpr SelectKShape kShape = {/*batch=*/4, /*n=*/4096, /*k=*/32};
  constexpr PrimitiveType kDtype = primitive_util::NativeToPrimitiveType<T>();

  ASSERT_OK_AND_ASSIGN(SelectKTestCase<T> test_case,
                       CreateSelectKTestCase<T>(kShape));
  se::DeviceAddress<T> d_data_in = test_case.d_data_in();
  se::DeviceAddress<T> d_data_out = test_case.d_data_out();
  se::DeviceAddress<uint32_t> d_indices_out = test_case.d_indices_out();
  se::DeviceAddressBase d_scratch = test_case.d_scratch.cref();

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
      ShapedSlice{slice_in,
                  ShapeUtil::MakeShape(kDtype, {kShape.batch, kShape.n})},
  };
  std::vector<NullableShapedSlice> results = {
      ShapedSlice{slice_out_val,
                  ShapeUtil::MakeShape(kDtype, {kShape.batch, kShape.k})},
      ShapedSlice{slice_out_idx,
                  ShapeUtil::MakeShape(S32, {kShape.batch, kShape.k})},
      ShapedSlice{slice_scratch,
                  ShapeUtil::MakeShape(U8, {test_case.scratch_bytes})},
  };

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> custom_call_thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(), std::string(kTopKCustomCallTarget),
          std::move(operands), std::move(results), /*attributes=*/{},
          /*called_computation=*/nullptr,
          test_case.stream_executor->GetPlatform()->Name(),
          test_case.stream_executor->GetDeviceDescription()
              .gpu_compute_capability()));

  CommandSequence commands;
  commands.Append(std::move(custom_call_thunk));
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(
                           std::move(commands),
                           CommandExecutor::SynchronizationMode::kSerialize));
  CommandBufferThunk cmd_buffer_thunk(std::move(executor), Thunk::ThunkInfo(),
                                      /*devices_in_process=*/1);

  BufferAllocations allocations(
      {d_data_in, d_data_out, d_indices_out, d_scratch}, 0,
      test_case.allocator.get());
  TestThunkParams thunk_params(test_case.stream_executor, &allocations,
                               test_case.stream.get());
  thunk_params.WithCommandBufferTraceStream(test_case.stream.get());
  ASSERT_OK(cmd_buffer_thunk.Initialize(thunk_params.InitializeParams()));
  thunk_params.WithPersistentAllocIndices({});
  Thunk::ExecuteParams params = thunk_params.ExecuteParams();

  // First execution: records the command buffer and executes it.
  ASSERT_OK(cmd_buffer_thunk.ExecuteOnStream(params));
  ASSERT_NO_FATAL_FAILURE(VerifySelectKResults(test_case));

  // Zero output buffers and run a second time to verify command buffer replay.
  ASSERT_OK(test_case.stream->MemZero(&d_data_out, d_data_out.size()));
  ASSERT_OK(test_case.stream->MemZero(&d_indices_out, d_indices_out.size()));
  ASSERT_OK(cmd_buffer_thunk.ExecuteOnStream(params));
  ASSERT_NO_FATAL_FAILURE(VerifySelectKResults(test_case));
}

TEST(RaftSelectKExecTest, CommandBufferFloat) { RunCommandBufferTest<float>(); }

TEST(RaftSelectKExecTest, CommandBufferBFloat16) {
  RunCommandBufferTest<::xla::bfloat16>();
}

TEST(RaftSelectKExecTest, CommandBufferUint64) {
  RunCommandBufferTest<uint64_t>();
}

TEST(RaftSelectKExecTest, CommandBufferUint32) {
  RunCommandBufferTest<uint32_t>();
}

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

  BufferAllocations allocations(
      {d_data_in_, d_data_out_, d_indices_out_, d_scratch_.cref()}, 0,
      allocator_.get());
  TestThunkParams thunk_params(stream_executor_, &allocations, stream_.get());
  thunk_params.WithCommandBufferTraceStream(stream_.get());
  ASSERT_THAT(cmd_buffer_thunk.Initialize(thunk_params.InitializeParams()),
              IsOk());
  thunk_params.WithPersistentAllocIndices({});
  Thunk::ExecuteParams params = thunk_params.ExecuteParams();

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
