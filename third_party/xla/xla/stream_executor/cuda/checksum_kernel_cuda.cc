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

#include <bit>
#include <cstdint>
#include <cstring>

#include "absl/base/attributes.h"
#include "xla/stream_executor/cuda/cuda_helpers.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace se = stream_executor;

namespace {

// Implementation based on
// https://github.com/veorq/SipHash/blob/master/halfsiphash.c
__device__ uint64_t HalfSipHash13(const uint8_t* bytes, uint64_t len,
                                  uint64_t key) {
  constexpr int kCRounds = 1;
  constexpr int kDRounds = 3;

  auto half_sip_round = [](uint32_t& v0, uint32_t& v1, uint32_t& v2,
                           uint32_t& v3) {
    v0 += v1;
    v1 = std::rotl(v1, 5);
    v1 ^= v0;
    v0 = std::rotl(v0, 16);
    v2 += v3;
    v3 = std::rotl(v3, 8);
    v3 ^= v2;
    v0 += v3;
    v3 = std::rotl(v3, 7);
    v3 ^= v0;
    v2 += v1;
    v1 = std::rotl(v1, 13);
    v1 ^= v2;
    v2 = std::rotl(v2, 16);
  };

  uint32_t v0 = 0;
  uint32_t v1 = 0;
  uint32_t v2 = 0x6c796765;
  uint32_t v3 = 0x74656462;
  uint32_t k0 = key & 0xffffffff;
  uint32_t k1 = (key >> 32) & 0xffffffff;
  uint32_t b = static_cast<uint32_t>(len) << 24;

  v3 ^= k1;
  v2 ^= k0;
  v1 ^= k1;
  v0 ^= k0;

  // output is 64-bit
  v1 ^= 0xee;

  const uint8_t* end = bytes + (len & ~3);
  for (; bytes != end; bytes += 4) {
    // assumed little-endian
    uint32_t m;
    memcpy(&m, bytes, sizeof(m));

    v3 ^= m;
    for (int i = 0; i < kCRounds; ++i) {
      half_sip_round(v0, v1, v2, v3);
    }
    v0 ^= m;
  }

  const int last_chunk_len = len & 3;
  switch (last_chunk_len) {
    case 3:
      b |= static_cast<uint32_t>(bytes[2]) << 16;
      ABSL_FALLTHROUGH_INTENDED;
    case 2:
      b |= static_cast<uint32_t>(bytes[1]) << 8;
      ABSL_FALLTHROUGH_INTENDED;
    case 1:
      b |= static_cast<uint32_t>(bytes[0]);
  }

  v3 ^= b;

  for (int i = 0; i < kCRounds; ++i) {
    half_sip_round(v0, v1, v2, v3);
  }
  v0 ^= b;

  // output is 64-bit
  v2 ^= 0xee;

  for (int i = 0; i < kDRounds; ++i) {
    half_sip_round(v0, v1, v2, v3);
  }

  b = v1 ^ v3;
  const uint32_t out_low = b;

  v1 ^= 0xdd;

  for (int i = 0; i < kDRounds; ++i) {
    half_sip_round(v0, v1, v2, v3);
  }

  b = v1 ^ v3;
  const uint32_t out_high = b;

  return static_cast<uint64_t>(out_high) << 32 | static_cast<uint64_t>(out_low);
}

__global__ void HalfSipHash13Kernel(const uint8_t* input, uint64_t length,
                                    uint64_t key, uint64_t* output) {
  *output = HalfSipHash13(input, length, key);
}

}  // namespace

absl::Status LaunchHalfSipHash13Kernel(se::Stream* stream,
                                       se::DeviceMemory<uint8_t>* input,
                                       uint64_t key,
                                       se::DeviceMemory<uint64_t>* output) {
  se::StreamExecutor* executor = stream->parent();
  TF_ASSIGN_OR_RETURN(
      auto kernel,
      (se::TypedKernelFactory<
          se::DeviceMemory<uint8_t>, uint64_t, uint64_t,
          se::DeviceMemory<uint64_t>>::Create(executor, "HalfSipHash13Kernel",
                                              reinterpret_cast<void*>(
                                                  HalfSipHash13Kernel))));

  se::ThreadDim thread_dim(1, 1, 1);
  se::BlockDim block_dim(1, 1, 1);

  return kernel.Launch(thread_dim, block_dim, stream, *input,
                       input->ElementCount(), key, *output);
}
