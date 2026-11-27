/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/libraries/cub/cub_sort_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_types.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_elf_utils.h"
#include "xla/stream_executor/cuda/cuda_fatbin.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;
using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

// Reference comparison matching CUB's radix sort ordering:
// - Integers: numeric `<` / `>`.
// - Floats: total-order-like except -0.0 and +0.0 compare equal (preserving
//   input order stably via std::stable_sort).
template <typename Key>
bool KeyLessThan(Key a, Key b, bool descending) {
  if constexpr (std::is_integral_v<Key>) {
    return descending ? (a > b) : (a < b);
  } else {
    auto to_ordered = [](Key v) -> uint64_t {
      using U = UnsignedIntegerTypeForSizeType<sizeof(Key)>;
      constexpr U kSignBit = static_cast<U>(U{1} << (sizeof(Key) * 8 - 1));
      U bits = absl::bit_cast<U>(v);
      if ((bits & static_cast<U>(~kSignBit)) == 0) {
        bits = 0;  // Equate -0.0 and +0.0.
      }
      const U mask = (bits & kSignBit) ? static_cast<U>(~U{0}) : kSignBit;
      return static_cast<uint64_t>(static_cast<U>(bits ^ mask));
    };
    uint64_t oa = to_ordered(a);
    uint64_t ob = to_ordered(b);
    return descending ? (oa > ob) : (oa < ob);
  }
}

template <typename Key>
std::vector<Key> GenerateKeys(int64_t count, uint32_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<Key> keys(count);
  if constexpr (std::is_integral_v<Key>) {
    std::uniform_int_distribution<uint64_t> dist(0, 1000000);
    for (int64_t i = 0; i < count; ++i) {
      // Mix wide range and duplicates to exercise tie-breaking and stability.
      uint64_t raw = (i % 7 == 0) ? (dist(rng) % 16) : rng();
      keys[i] = static_cast<Key>(raw);
    }
  } else {
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (int64_t i = 0; i < count; ++i) {
      double val = (i % 11 == 0) ? static_cast<double>((i % 5) - 2) : dist(rng);
      if (i % 37 == 0) {
        val = -0.0;
      } else if (i % 41 == 0) {
        val = 0.0;
      }
      keys[i] = static_cast<Key>(val);
    }
  }
  return keys;
}

template <typename Key>
void ExpectKeysBitEqual(const std::vector<Key>& actual,
                        const std::vector<Key>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  using U = UnsignedIntegerTypeForSizeType<sizeof(Key)>;
  for (size_t i = 0; i < actual.size(); ++i) {
    const uint64_t a = static_cast<uint64_t>(absl::bit_cast<U>(actual[i]));
    const uint64_t e = static_cast<uint64_t>(absl::bit_cast<U>(expected[i]));
    if (a != e) {
      ADD_FAILURE() << "Mismatch at index " << i << ": actual=0x" << std::hex
                    << a << " expected=0x" << e << std::dec;
      return;
    }
  }
}

class CubSortKernelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                         se::PlatformManager::PlatformWithName("CUDA"));
    ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    allocator_ =
        std::make_unique<se::StreamExecutorAddressAllocator>(executor_);
    cc_ = executor_->GetDeviceDescription().cuda_compute_capability();
    sm_version_ = CanonicalCubSortPolicySmVersion(cc_.major);
  }

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(int64_t bytes) {
    return allocator_->Allocate(executor_->device_ordinal(), bytes);
  }

  template <typename Key>
  void RunSingleTileKeysWithInput(PrimitiveType key_type,
                                  const std::vector<Key>& h_in,
                                  int32_t segment_size, int64_t batch_size,
                                  bool descending, bool split_chunks = false) {
    ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec spec,
                         GetCubSortSingleTileKernelSpec(key_type, std::nullopt,
                                                        descending, cc_));
    using KernelFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               int32_t>;
    ASSERT_OK_AND_ASSIGN(auto kernel, KernelFactory::Create(executor_, spec));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(Key), /*value_byte_size=*/0,
        std::is_same_v<Key, float>);
    ASSERT_LE(segment_size, policy.single_tile.tile_items());

    const int64_t total_items = batch_size * segment_size;
    ASSERT_EQ(h_in.size(), total_items);
    std::vector<Key> h_expected = h_in;
    for (int64_t b = 0; b < batch_size; ++b) {
      auto begin = h_expected.begin() + b * segment_size;
      std::stable_sort(begin, begin + segment_size, [descending](Key a, Key b) {
        return KeyLessThan(a, b, descending);
      });
    }

    const int64_t byte_size = total_items * sizeof(Key);
    const int64_t alloc_bytes = std::max<int64_t>(byte_size, sizeof(Key));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_in,
                         Allocate(alloc_bytes));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_out,
                         Allocate(alloc_bytes));
    if (byte_size > 0) {
      ASSERT_OK(stream_->Memcpy(d_in.ptr(), h_in.data(), byte_size));
    }
    ASSERT_OK(stream_->MemZero(d_out.ptr(), alloc_bytes));

    // Pass explicit 0 dynamic shared memory: CUB kernels use static __shared__
    // storage only, and TypedKernel::Launch without an explicit shmem_bytes
    // argument forwards static shared memory metadata as dynamic shared memory
    // (doubling shared memory usage and exceeding T4's 64 KiB opt-in limit).
    if (split_chunks && batch_size >= 2) {
      const int64_t first = batch_size / 2;
      const int64_t first_bytes = first * segment_size * sizeof(Key);
      se::DeviceAddressBase d_in_second(
          static_cast<char*>(d_in->opaque()) + first_bytes,
          byte_size - first_bytes);
      se::DeviceAddressBase d_out_second(
          static_cast<char*>(d_out->opaque()) + first_bytes,
          byte_size - first_bytes);
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(first), /*shmem_bytes=*/0,
                              stream_.get(), *d_in, *d_out, segment_size));
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(batch_size - first),
                              /*shmem_bytes=*/0, stream_.get(), d_in_second,
                              d_out_second, segment_size));
    } else {
      ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                              se::BlockDim(batch_size), /*shmem_bytes=*/0,
                              stream_.get(), *d_in, *d_out, segment_size));
    }

    std::vector<Key> h_out(total_items);
    if (byte_size > 0) {
      ASSERT_OK(stream_->Memcpy(h_out.data(), *d_out, byte_size));
    }
    ASSERT_OK(stream_->BlockHostUntilDone());

    ExpectKeysBitEqual(h_out, h_expected);
  }

  template <typename Key>
  void RunSingleTileKeys(PrimitiveType key_type, int32_t segment_size,
                         int64_t batch_size, bool descending,
                         bool split_chunks = false) {
    RunSingleTileKeysWithInput<Key>(
        key_type, GenerateKeys<Key>(batch_size * segment_size, 12345),
        segment_size, batch_size, descending, split_chunks);
  }

  template <typename Key, typename Value>
  void RunSingleTilePairs(PrimitiveType key_type, PrimitiveType val_type,
                          const std::vector<Key>& h_keys_in,
                          int32_t segment_size, int64_t batch_size,
                          bool descending) {
    ASSERT_OK_AND_ASSIGN(
        se::KernelLoaderSpec spec,
        GetCubSortSingleTileKernelSpec(key_type, val_type, descending, cc_));
    using KernelFactory =
        se::TypedKernelFactory<se::DeviceAddressBase, se::DeviceAddressBase,
                               se::DeviceAddressBase, se::DeviceAddressBase,
                               int32_t>;
    ASSERT_OK_AND_ASSIGN(auto kernel, KernelFactory::Create(executor_, spec));

    const CubRadixSortPolicySpec policy = GetCubRadixSortPolicySpec(
        sm_version_, sizeof(Key), sizeof(Value), std::is_same_v<Key, float>);
    ASSERT_LE(segment_size, policy.single_tile.tile_items());

    const int64_t total_items = batch_size * segment_size;
    ASSERT_EQ(h_keys_in.size(), total_items);
    std::vector<Value> h_vals_in(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      h_vals_in[i] = static_cast<Value>(i);
    }

    std::vector<std::pair<Key, Value>> pairs(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      pairs[i] = {h_keys_in[i], h_vals_in[i]};
    }
    for (int64_t b = 0; b < batch_size; ++b) {
      auto begin = pairs.begin() + b * segment_size;
      std::stable_sort(begin, begin + segment_size,
                       [descending](const auto& a, const auto& b) {
                         return KeyLessThan(a.first, b.first, descending);
                       });
    }
    std::vector<Key> expected_keys(total_items);
    std::vector<Value> expected_vals(total_items);
    for (int64_t i = 0; i < total_items; ++i) {
      expected_keys[i] = pairs[i].first;
      expected_vals[i] = pairs[i].second;
    }

    const int64_t key_bytes = total_items * sizeof(Key);
    const int64_t val_bytes = total_items * sizeof(Value);
    const int64_t key_alloc = std::max<int64_t>(key_bytes, sizeof(Key));
    const int64_t val_alloc = std::max<int64_t>(val_bytes, sizeof(Value));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_keys_in,
                         Allocate(key_alloc));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_keys_out,
                         Allocate(key_alloc));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_vals_in,
                         Allocate(val_alloc));
    ASSERT_OK_AND_ASSIGN(se::ScopedDeviceAddress<uint8_t> d_vals_out,
                         Allocate(val_alloc));
    if (key_bytes > 0) {
      ASSERT_OK(stream_->Memcpy(d_keys_in.ptr(), h_keys_in.data(), key_bytes));
      ASSERT_OK(stream_->Memcpy(d_vals_in.ptr(), h_vals_in.data(), val_bytes));
    }

    ASSERT_OK(kernel.Launch(se::ThreadDim(policy.single_tile.block_threads),
                            se::BlockDim(batch_size), /*shmem_bytes=*/0,
                            stream_.get(), *d_keys_in, *d_keys_out, *d_vals_in,
                            *d_vals_out, segment_size));

    std::vector<Key> h_keys_out(total_items);
    std::vector<Value> h_vals_out(total_items);
    if (key_bytes > 0) {
      ASSERT_OK(stream_->Memcpy(h_keys_out.data(), *d_keys_out, key_bytes));
      ASSERT_OK(stream_->Memcpy(h_vals_out.data(), *d_vals_out, val_bytes));
    }
    ASSERT_OK(stream_->BlockHostUntilDone());

    ExpectKeysBitEqual(h_keys_out, expected_keys);
    EXPECT_EQ(h_vals_out, expected_vals);
  }

  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator_;
  se::CudaComputeCapability cc_;
  int sm_version_ = 800;
};

TEST_F(CubSortKernelsTest, SingleTileKeysAllTypes) {
  for (bool descending : {false, true}) {
    RunSingleTileKeys<bfloat16>(BF16, 257, 1, descending);
    RunSingleTileKeys<half>(F16, 257, 1, descending);
    RunSingleTileKeys<float>(F32, 1024, 4, descending, /*split_chunks=*/true);
    RunSingleTileKeys<double>(F64, 513, 2, descending);
    RunSingleTileKeys<int8_t>(S8, 300, 1, descending);
    RunSingleTileKeys<int16_t>(S16, 300, 1, descending);
    RunSingleTileKeys<int32_t>(S32, 4864, 1, descending);
    RunSingleTileKeys<int64_t>(S64, 2304, 1, descending);
    RunSingleTileKeys<uint8_t>(U8, 300, 3, descending);
    RunSingleTileKeys<uint16_t>(U16, 300, 3, descending);
    RunSingleTileKeys<uint32_t>(U32, 512, 2, descending);
    RunSingleTileKeys<uint64_t>(U64, 256, 2, descending);
  }
}

TEST_F(CubSortKernelsTest, SingleTilePairsAllTypes) {
  for (bool descending : {false, true}) {
    RunSingleTilePairs<uint8_t, uint16_t>(
        U8, U16, GenerateKeys<uint8_t>(300, 1), 300, 1, descending);
    RunSingleTilePairs<uint8_t, uint32_t>(
        U8, U32, GenerateKeys<uint8_t>(600, 2), 300, 2, descending);
    RunSingleTilePairs<uint8_t, uint64_t>(
        U8, U64, GenerateKeys<uint8_t>(300, 3), 300, 1, descending);

    RunSingleTilePairs<uint16_t, uint16_t>(
        U16, U16, GenerateKeys<uint16_t>(300, 4), 300, 1, descending);
    RunSingleTilePairs<uint16_t, uint32_t>(
        U16, U32, GenerateKeys<uint16_t>(600, 5), 300, 2, descending);
    RunSingleTilePairs<uint16_t, uint64_t>(
        U16, U64, GenerateKeys<uint16_t>(300, 6), 300, 1, descending);

    RunSingleTilePairs<int32_t, uint16_t>(
        S32, U16, GenerateKeys<int32_t>(300, 7), 300, 1, descending);
    RunSingleTilePairs<int32_t, uint32_t>(
        S32, U32, GenerateKeys<int32_t>(1024, 8), 512, 2, descending);
    RunSingleTilePairs<int32_t, uint64_t>(
        S32, U64, GenerateKeys<int32_t>(300, 9), 300, 1, descending);

    RunSingleTilePairs<uint32_t, uint16_t>(
        U32, U16, GenerateKeys<uint32_t>(300, 10), 300, 1, descending);
    RunSingleTilePairs<uint32_t, uint32_t>(
        U32, U32, GenerateKeys<uint32_t>(300, 11), 300, 1, descending);
    RunSingleTilePairs<uint32_t, uint64_t>(
        U32, U64, GenerateKeys<uint32_t>(300, 12), 300, 1, descending);

    RunSingleTilePairs<uint64_t, uint16_t>(
        U64, U16, GenerateKeys<uint64_t>(300, 13), 300, 1, descending);
    RunSingleTilePairs<uint64_t, uint32_t>(
        U64, U32, GenerateKeys<uint64_t>(300, 14), 300, 1, descending);
    RunSingleTilePairs<uint64_t, uint64_t>(
        U64, U64, GenerateKeys<uint64_t>(300, 15), 300, 1, descending);

    RunSingleTilePairs<float, uint16_t>(F32, U16, GenerateKeys<float>(300, 16),
                                        300, 1, descending);
    RunSingleTilePairs<float, uint32_t>(F32, U32, GenerateKeys<float>(1024, 17),
                                        512, 2, descending);
    RunSingleTilePairs<float, uint64_t>(F32, U64, GenerateKeys<float>(300, 18),
                                        300, 1, descending);
  }
}

TEST_F(CubSortKernelsTest, FloatEdgeCasesAndNegativeZeroStability) {
  const float neg_nan = -std::numeric_limits<float>::quiet_NaN();
  const float pos_nan = std::numeric_limits<float>::quiet_NaN();
  const float max_oob_f32 = absl::bit_cast<float>(uint32_t{0x7FFFFFFFu});
  const float min_oob_f32 = absl::bit_cast<float>(uint32_t{0xFFFFFFFFu});
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const float pos_inf = std::numeric_limits<float>::infinity();

  // Include exact MakeDefaultOobKey sentinels (0x7FFFFFFF / 0xFFFFFFFF) in a
  // partial tile to verify valid sentinel keys stay ahead of OOB lanes.
  std::vector<float> base = {
      pos_nan,     max_oob_f32, 3.0f,    -0.0f,   +0.0f, neg_inf, -1.5f, +0.0f,
      min_oob_f32, -0.0f,       pos_inf, neg_nan, -0.0f, 0.0f,    3.0f,  -1.5f};

  // SingleTile stability with -0.0 / +0.0 / NaNs / Infs / OOB sentinels.
  for (bool descending : {false, true}) {
    RunSingleTilePairs<float, uint32_t>(F32, U32, base, base.size(), 1,
                                        descending);
  }

  // Integer OOB sentinel ties in partial tiles.
  const std::vector<int32_t> s32_sentinels = {
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min(),
      0,
      -1,
      1,
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min()};
  const std::vector<uint32_t> u32_sentinels = {
      std::numeric_limits<uint32_t>::max(), 0, 42,
      std::numeric_limits<uint32_t>::max(), 0};
  for (bool descending : {false, true}) {
    RunSingleTilePairs<int32_t, uint32_t>(S32, U32, s32_sentinels,
                                          s32_sentinels.size(), 1, descending);
    RunSingleTilePairs<uint32_t, uint32_t>(U32, U32, u32_sentinels,
                                           u32_sentinels.size(), 1, descending);
  }

  // BF16, F16, and F64 edge cases (NaN sentinels, Inf, subnormals, +/-0.0).
  const std::vector<bfloat16> bf16_edge = {
      absl::bit_cast<bfloat16>(uint16_t{0x7FFFu}),
      absl::bit_cast<bfloat16>(uint16_t{0xFFFFu}),
      absl::bit_cast<bfloat16>(uint16_t{0x7F80u}),
      absl::bit_cast<bfloat16>(uint16_t{0xFF80u}),
      absl::bit_cast<bfloat16>(uint16_t{0x0001u}),
      absl::bit_cast<bfloat16>(uint16_t{0x8001u}),
      absl::bit_cast<bfloat16>(uint16_t{0x8000u}),
      absl::bit_cast<bfloat16>(uint16_t{0x0000u}),
      bfloat16(1.5f),
      bfloat16(-2.5f),
  };
  const std::vector<half> f16_edge = {
      absl::bit_cast<half>(uint16_t{0x7FFFu}),
      absl::bit_cast<half>(uint16_t{0xFFFFu}),
      absl::bit_cast<half>(uint16_t{0x7C00u}),
      absl::bit_cast<half>(uint16_t{0xFC00u}),
      absl::bit_cast<half>(uint16_t{0x0001u}),
      absl::bit_cast<half>(uint16_t{0x8001u}),
      absl::bit_cast<half>(uint16_t{0x8000u}),
      absl::bit_cast<half>(uint16_t{0x0000u}),
      half(1.5f),
      half(-2.5f),
  };
  const std::vector<double> f64_edge = {
      absl::bit_cast<double>(uint64_t{0x7FFFFFFFFFFFFFFFULL}),
      absl::bit_cast<double>(uint64_t{0xFFFFFFFFFFFFFFFFULL}),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::denorm_min(),
      -std::numeric_limits<double>::denorm_min(),
      -0.0,
      +0.0,
      3.25,
      -7.5,
  };
  for (bool descending : {false, true}) {
    RunSingleTileKeysWithInput<bfloat16>(BF16, bf16_edge, bf16_edge.size(), 1,
                                         descending);
    RunSingleTileKeysWithInput<half>(F16, f16_edge, f16_edge.size(), 1,
                                     descending);
    RunSingleTileKeysWithInput<double>(F64, f64_edge, f64_edge.size(), 1,
                                       descending);
  }
}

TEST_F(CubSortKernelsTest, SingleTileBoundarySizesAndValueDispatch) {
  // Boundary segment sizes: 0 (no OOB access), 1, and full asymmetric 8B-value
  // tile (U8 keys + U64 values -> DominantType is 8 bytes -> tile_items 2304).
  for (bool descending : {false, true}) {
    RunSingleTileKeys<uint32_t>(U32, /*segment_size=*/0, /*batch_size=*/2,
                                descending);
    RunSingleTileKeys<uint32_t>(U32, /*segment_size=*/1, /*batch_size=*/3,
                                descending);
    RunSingleTilePairs<uint32_t, uint32_t>(U32, U32, {}, /*segment_size=*/0,
                                           /*batch_size=*/2, descending);
    RunSingleTilePairs<uint32_t, uint32_t>(U32, U32, {7, 3}, /*segment_size=*/1,
                                           /*batch_size=*/2, descending);
    RunSingleTilePairs<uint8_t, uint64_t>(
        U8, U64, GenerateKeys<uint8_t>(2304, 77), /*segment_size=*/2304,
        /*batch_size=*/1, descending);
  }

  // Non-unsigned value types exercising 16/32/64-bit value-width dispatch.
  RunSingleTilePairs<float, float>(F32, F32, GenerateKeys<float>(128, 19), 128,
                                   1, /*descending=*/false);
  RunSingleTilePairs<uint32_t, bfloat16>(
      U32, BF16, GenerateKeys<uint32_t>(128, 20), 128, 1, /*descending=*/true);
  RunSingleTilePairs<uint32_t, int64_t>(
      U32, S64, GenerateKeys<uint32_t>(128, 21), 128, 1, /*descending=*/false);
}

TEST_F(CubSortKernelsTest, SingleArchCubinExtractionAndUnsupportedArchError) {
  ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec spec,
                       GetCubSortSingleTileKernelSpec(
                           F32, std::nullopt, /*descending=*/false, cc_));
  ASSERT_TRUE(spec.cuda_cubin_in_memory().has_value());
  // Architectures compiled natively into the fatbinary (SM <= 10.x in standard
  // CUDA builds) extract an uncompressed single-arch CUBIN ELF rather than
  // falling back to the full multi-arch fatbinary.
  if (cc_.major <= 10) {
    EXPECT_TRUE(se::cuda::IsCudaElf(spec.cuda_cubin_in_memory()->cubin_bytes));
  }

  // Unsupported old archs with neither CUBIN nor compatible PTX must fail
  // rather than silently returning an un-runnable fatbinary. Test both below
  // the SM 5.0 floor (3.0) and above it (5.0, which enters PTX validation).
  EXPECT_THAT(
      GetCubSortSingleTileKernelSpec(F32, std::nullopt, /*descending=*/false,
                                     se::CudaComputeCapability(3, 0)),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      GetCubSortSingleTileKernelSpec(F32, std::nullopt, /*descending=*/false,
                                     se::CudaComputeCapability(5, 0)),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

std::vector<uint8_t> MakeTestPtxFatbin(
    absl::Span<const se::CudaComputeCapability> ptx_archs) {
  static constexpr uint8_t kDummyPtx[8] = {'.', 'v', 'e', 'r',
                                           's', 'i', 'o', 'n'};
  std::vector<uint8_t> entries;
  for (const se::CudaComputeCapability& cc : ptx_archs) {
    se::cuda::FatEntryHeader entry{};
    entry.kind = se::cuda::kFatbinKindPtx;
    entry.unknown = 0x0101;
    entry.header_size = sizeof(se::cuda::FatEntryHeader);
    entry.size = sizeof(kDummyPtx);
    entry.arch = static_cast<uint32_t>(cc.major * 10 + cc.minor);
    entry.flags = se::cuda::kFatbinFlag64Bit;
    if (cc.feature_extension ==
        se::CudaComputeCapability::FeatureExtension::kAcceleratedFeatures) {
      entry.flags |= se::cuda::kFatbinFlagAccelerated;
    } else if (cc.feature_extension ==
               se::CudaComputeCapability::FeatureExtension::
                   kFamilyCompatibleFeatures) {
      entry.flags |= se::cuda::kFatbinFlagFamilyCompatible;
    }
    const auto* entry_bytes = reinterpret_cast<const uint8_t*>(&entry);
    entries.insert(entries.end(), entry_bytes, entry_bytes + sizeof(entry));
    entries.insert(entries.end(), kDummyPtx, kDummyPtx + sizeof(kDummyPtx));
  }

  se::cuda::FatHeader header{};
  header.magic = se::cuda::kFatbinMagicCompressed;
  header.version = 1;
  header.header_size = sizeof(se::cuda::FatHeader);
  header.fat_size = entries.size();

  std::vector<uint8_t> fatbin(sizeof(header) + entries.size());
  std::memcpy(fatbin.data(), &header, sizeof(header));
  if (!entries.empty()) {
    std::memcpy(fatbin.data() + sizeof(header), entries.data(), entries.size());
  }
  return fatbin;
}

TEST(ValidateFatbinPtxPolicyTierTest, CoversValidationAndCandidateSelection) {
  using FeatureExtension = se::CudaComputeCapability::FeatureExtension;

  // Non-fatbinary payload fails before candidate selection.
  const std::vector<uint8_t> not_a_fatbin = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  not_a_fatbin, se::CudaComputeCapability(9, 0)),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Empty fatbinary (no PTX entries) fails.
  const std::vector<uint8_t> empty_fatbin = MakeTestPtxFatbin({});
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  empty_fatbin, se::CudaComputeCapability(9, 0)),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // SM < 5.0 is rejected even if a fatbinary is well-formed.
  const std::vector<uint8_t> sm80_sm90_fatbin = MakeTestPtxFatbin(
      {se::CudaComputeCapability(8, 0), se::CudaComputeCapability(9, 0)});
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  sm80_sm90_fatbin, se::CudaComputeCapability(3, 0)),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Target older than all embedded PTX archs has no compatible candidate.
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  sm80_sm90_fatbin, se::CudaComputeCapability(7, 5)),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Highest compatible (major, minor) PTX is selected and matched against the
  // target's canonical policy tier:
  // - sm_86 selects compute_80 (tier 800 == 800) -> OK.
  // - sm_90 selects compute_90 (tier 900 == 900) -> OK.
  // - sm_100 selects compute_90 (tier 900 != 1000) -> FailedPrecondition.
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  sm80_sm90_fatbin, se::CudaComputeCapability(8, 6)),
              IsOk());
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  sm80_sm90_fatbin, se::CudaComputeCapability(9, 0)),
              IsOk());
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  sm80_sm90_fatbin, se::CudaComputeCapability(10, 0)),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Feature-extension compatibility (`SupportsAllFeaturesOf`):
  // - compute_100a PTX is compatible with sm_100a, NOT sm_103a.
  // - compute_100f PTX is compatible with sm_103a (same family), NOT sm_120.
  const std::vector<uint8_t> sm100a_fatbin =
      MakeTestPtxFatbin({se::CudaComputeCapability(
          10, 0, FeatureExtension::kAcceleratedFeatures)});
  EXPECT_THAT(
      internal::ValidateFatbinPtxPolicyTier(
          sm100a_fatbin, se::CudaComputeCapability(
                             10, 0, FeatureExtension::kAcceleratedFeatures)),
      IsOk());
  EXPECT_THAT(
      internal::ValidateFatbinPtxPolicyTier(
          sm100a_fatbin, se::CudaComputeCapability(
                             10, 3, FeatureExtension::kAcceleratedFeatures)),
      StatusIs(absl::StatusCode::kFailedPrecondition));

  const std::vector<uint8_t> sm100f_fatbin =
      MakeTestPtxFatbin({se::CudaComputeCapability(
          10, 0, FeatureExtension::kFamilyCompatibleFeatures)});
  EXPECT_THAT(
      internal::ValidateFatbinPtxPolicyTier(
          sm100f_fatbin, se::CudaComputeCapability(
                             10, 3, FeatureExtension::kAcceleratedFeatures)),
      IsOk());
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  sm100f_fatbin, se::CudaComputeCapability(12, 0)),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Generic compute_100 PTX matches future SM 12.0 because both map to the top
  // canonical policy tier (SM 1000).
  const std::vector<uint8_t> sm100_fatbin =
      MakeTestPtxFatbin({se::CudaComputeCapability(10, 0)});
  EXPECT_THAT(internal::ValidateFatbinPtxPolicyTier(
                  sm100_fatbin, se::CudaComputeCapability(12, 0)),
              IsOk());
}

TEST_F(CubSortKernelsTest, KernelSpecSupportMatchesCubSortTypesAcrossAllTypes) {
  auto verify_types = [&](PrimitiveType key_type,
                          std::optional<PrimitiveType> val_type) {
    const bool supported = IsCubSortSupported(key_type, val_type);
    for (bool descending : {false, true}) {
      const auto single_tile =
          GetCubSortSingleTileKernelSpec(key_type, val_type, descending, cc_);
      if (supported) {
        EXPECT_THAT(single_tile, IsOk())
            << PrimitiveType_Name(key_type) << ", "
            << (val_type ? PrimitiveType_Name(*val_type) : "keys-only")
            << ", descending=" << descending;
      } else {
        EXPECT_THAT(single_tile, StatusIs(absl::StatusCode::kInvalidArgument))
            << PrimitiveType_Name(key_type) << ", "
            << (val_type ? PrimitiveType_Name(*val_type) : "keys-only")
            << ", descending=" << descending;
      }
    }
  };

  for (int k = PrimitiveType_MIN; k <= PrimitiveType_MAX; ++k) {
    if (!PrimitiveType_IsValid(k)) {
      continue;
    }
    const auto key_type = static_cast<PrimitiveType>(k);
    verify_types(key_type, std::nullopt);
    for (int v = PrimitiveType_MIN; v <= PrimitiveType_MAX; ++v) {
      if (!PrimitiveType_IsValid(v)) {
        continue;
      }
      verify_types(key_type, static_cast<PrimitiveType>(v));
    }
  }
}

}  // namespace
}  // namespace xla::gpu
