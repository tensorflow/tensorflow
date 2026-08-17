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
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
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
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());
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

  // Copy host to device
  TF_ASSERT_OK(stream->MemcpyH2D(absl::Span<const T>(h_data_in), &d_data_in));

  // Run raft select_k
  TF_ASSERT_OK(select_k_exec<T>(device_ordinal, &allocator, stream.get(),
                                d_data_in, d_data_out, d_indices_out, batch, n,
                                k));

  // Copy results back to host
  std::vector<T> h_data_out(batch * k);
  std::vector<uint32_t> h_indices_out(batch * k);
  TF_ASSERT_OK(stream->MemcpyD2H(d_data_out, absl::Span<T>(h_data_out)));
  TF_ASSERT_OK(
      stream->MemcpyD2H(d_indices_out, absl::Span<uint32_t>(h_indices_out)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

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

}  // namespace xla::gpu
