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

#include "xla/backends/gpu/libraries/cub/cub_sort_kernels_impl.h"

#include <cstdint>
#include <cstring>
#include <type_traits>

// Clang can't always unroll all CUB loops; silence the warning to avoid build
// breaks with -Werror (matches stream_executor/cuda/cub_sort_kernel_cuda_impl).
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

#include "cub/agent/agent_radix_sort_downsweep.cuh"
#include "cub/block/block_load.cuh"
#include "cub/block/block_radix_sort.cuh"
#include "cub/util_device.cuh"
#include "cub/util_ptx.cuh"
#include "cub/util_type.cuh"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::internal {
namespace {

// Device-side CUB policy constructed directly from GetCubRadixSortPolicySpec so
// that host planning and compiled device CUBINs stay strictly locked across
// CCCL versions.
template <int kSmVersion, typename Key, typename Value>
struct XlaCubRadixSortPolicy {
  static constexpr bool kKeysOnly = std::is_same_v<Value, cub::NullType>;
  static constexpr int kValueByteSize =
      kKeysOnly ? 0 : static_cast<int>(sizeof(Value));
  using DominantType =
      std::conditional_t<(sizeof(Value) > sizeof(Key)), Value, Key>;

  static constexpr CubRadixSortPolicySpec kSpec =
      GetCubRadixSortPolicySpec(kSmVersion, static_cast<int>(sizeof(Key)),
                                kValueByteSize, std::is_same_v<Key, float>);

  using SingleTilePolicy = cub::AgentRadixSortDownsweepPolicy<
      kSpec.single_tile.nominal_threads, kSpec.single_tile.nominal_items,
      DominantType, cub::BLOCK_LOAD_DIRECT, cub::LOAD_LDG,
      cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_WARP_SCANS,
      kSpec.single_tile.radix_bits>;

  static_assert(SingleTilePolicy::BLOCK_THREADS ==
                kSpec.single_tile.block_threads);
  static_assert(SingleTilePolicy::ITEMS_PER_THREAD ==
                kSpec.single_tile.items_per_thread);
  static_assert(SingleTilePolicy::RADIX_BITS == kSpec.single_tile.radix_bits);
};

// Major-SM policy chain (500..1000) so minor-SM CUBIN fallbacks (e.g., sm_100
// on sm_103) always match CanonicalCubSortPolicySmVersion(cc.major).
template <typename Key, typename Value>
struct XlaChainedPolicy {
  struct Policy500 : cub::ChainedPolicy<500, Policy500, Policy500>,
                     XlaCubRadixSortPolicy<500, Key, Value> {};
  struct Policy600 : cub::ChainedPolicy<600, Policy600, Policy500>,
                     XlaCubRadixSortPolicy<600, Key, Value> {};
  struct Policy700 : cub::ChainedPolicy<700, Policy700, Policy600>,
                     XlaCubRadixSortPolicy<700, Key, Value> {};
  struct Policy800 : cub::ChainedPolicy<800, Policy800, Policy700>,
                     XlaCubRadixSortPolicy<800, Key, Value> {};
  struct Policy900 : cub::ChainedPolicy<900, Policy900, Policy800>,
                     XlaCubRadixSortPolicy<900, Key, Value> {};
  struct Policy1000 : cub::ChainedPolicy<1000, Policy1000, Policy900>,
                      XlaCubRadixSortPolicy<1000, Key, Value> {};

  using MaxPolicy = Policy1000;
};

// Returns a padding sentinel key for out-of-bounds lanes when a segment is
// smaller than the single-tile capacity (`segment_size < TILE_ITEMS`).
// `BlockRadixSort` always sorts a full tile, so padding with the extreme
// bit-order key (`MAX_KEY` for ascending, `LOWEST_KEY` for descending) ensures
// OOB lanes sort to the tail of the tile and are skipped by the guarded store.
// `std::memcpy` from `UnsignedBits` preserves exact bit patterns (including
// IEEE-754 NaN payloads).
template <SortDirection kDirection, typename Key>
__device__ __forceinline__ Key MakeDefaultOobKey() {
  using Traits = cub::Traits<Key>;
  typename Traits::UnsignedBits default_key_bits =
      kDirection == SortDirection::kDescending ? Traits::LOWEST_KEY
                                               : Traits::MAX_KEY;
  Key default_key;
  std::memcpy(&default_key, &default_key_bits, sizeof(Key));
  return default_key;
}

//===----------------------------------------------------------------------===//
// 1. Unified SingleTile kernels (unbatched and batched segments <= tile_items)
//===----------------------------------------------------------------------===//

template <typename ChainedPolicy, SortDirection kDirection, typename Key,
          typename Value>
__device__ __forceinline__ void RunSingleTileSort(
    const Key* __restrict__ keys_in, Key* __restrict__ keys_out,
    const Value* __restrict__ values_in, Value* __restrict__ values_out,
    int32_t segment_size) {
  using Policy = typename ChainedPolicy::ActivePolicy::SingleTilePolicy;
  constexpr int kBlockThreads = Policy::BLOCK_THREADS;
  constexpr int kItemsPerThread = Policy::ITEMS_PER_THREAD;
  constexpr bool kKeysOnly = std::is_same_v<Value, cub::NullType>;

  using BlockRadixSort =
      cub::BlockRadixSort<Key, kBlockThreads, kItemsPerThread, Value,
                          Policy::RADIX_BITS,
                          (Policy::RANK_ALGORITHM == cub::RADIX_RANK_MEMOIZE),
                          Policy::SCAN_ALGORITHM>;
  using BlockLoadKeys = cub::BlockLoad<Key, kBlockThreads, kItemsPerThread,
                                       Policy::LOAD_ALGORITHM>;
  struct EmptyStorage {};
  using BlockLoadValuesStorage = std::conditional_t<
      kKeysOnly, EmptyStorage,
      typename cub::BlockLoad<Value, kBlockThreads, kItemsPerThread,
                              Policy::LOAD_ALGORITHM>::TempStorage>;

  __shared__ union {
    typename BlockRadixSort::TempStorage sort;
    typename BlockLoadKeys::TempStorage load_keys;
    BlockLoadValuesStorage load_values;
  } temp_storage;

  const int64_t base_offset =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(segment_size);
  const Key* segment_keys_in = keys_in + base_offset;
  Key* segment_keys_out = keys_out + base_offset;

  Key keys[kItemsPerThread];
  Value values[kItemsPerThread];

  BlockLoadKeys(temp_storage.load_keys)
      .Load(segment_keys_in, keys, segment_size,
            MakeDefaultOobKey<kDirection, Key>());
  __syncthreads();

  int32_t valid_items = segment_size;
  if constexpr (!kKeysOnly) {
    // Mirror CUB's register-pressure workaround: moving segment_size through
    // shfl prevents ptxas from holding prior guarded-load predicates across
    // loads.
    static_assert(kBlockThreads % 32 == 0);
    valid_items = cub::ShuffleIndex<32>(segment_size, 0, 0xffffffff);
    using BlockLoadValues =
        cub::BlockLoad<Value, kBlockThreads, kItemsPerThread,
                       Policy::LOAD_ALGORITHM>;
    BlockLoadValues(temp_storage.load_values)
        .Load(values_in + base_offset, values, valid_items);
    __syncthreads();
  }

  constexpr int kEndBit = sizeof(Key) * 8;
  if constexpr (kKeysOnly) {
    if constexpr (kDirection == SortDirection::kDescending) {
      BlockRadixSort(temp_storage.sort)
          .SortDescendingBlockedToStriped(keys, /*begin_bit=*/0, kEndBit);
    } else {
      BlockRadixSort(temp_storage.sort)
          .SortBlockedToStriped(keys, /*begin_bit=*/0, kEndBit);
    }
  } else {
    if constexpr (kDirection == SortDirection::kDescending) {
      BlockRadixSort(temp_storage.sort)
          .SortDescendingBlockedToStriped(keys, values, /*begin_bit=*/0,
                                          kEndBit);
    } else {
      BlockRadixSort(temp_storage.sort)
          .SortBlockedToStriped(keys, values, /*begin_bit=*/0, kEndBit);
    }
  }

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int item_offset =
        item * kBlockThreads + static_cast<int>(threadIdx.x);
    if (item_offset < valid_items) {
      segment_keys_out[item_offset] = keys[item];
      if constexpr (!kKeysOnly) {
        values_out[base_offset + item_offset] = values[item];
      }
    }
  }
}

template <typename ChainedPolicy, SortDirection kDirection, typename Key>
__launch_bounds__(ChainedPolicy::ActivePolicy::SingleTilePolicy::BLOCK_THREADS)
    __global__
    void CubRadixSortSingleTileKeysKernel(const Key* __restrict__ keys_in,
                                          Key* __restrict__ keys_out,
                                          int32_t segment_size) {
  RunSingleTileSort<ChainedPolicy, kDirection, Key, cub::NullType>(
      keys_in, keys_out, /*values_in=*/nullptr, /*values_out=*/nullptr,
      segment_size);
}

template <typename ChainedPolicy, SortDirection kDirection, typename Key,
          typename Value>
__launch_bounds__(ChainedPolicy::ActivePolicy::SingleTilePolicy::BLOCK_THREADS)
    __global__
    void CubRadixSortSingleTilePairsKernel(const Key* __restrict__ keys_in,
                                           Key* __restrict__ keys_out,
                                           const Value* __restrict__ values_in,
                                           Value* __restrict__ values_out,
                                           int32_t segment_size) {
  RunSingleTileSort<ChainedPolicy, kDirection, Key, Value>(
      keys_in, keys_out, values_in, values_out, segment_size);
}

template <typename Return, typename... Args>
KernelHostRef MakeKernelHostRef(Return (*fn)(Args...)) {
  return {reinterpret_cast<const void*>(fn), sizeof...(Args)};
}

template <SortDirection kDirection, typename Key>
CubSortTypeKernelRefs MakeKeysRefs() {
  using Policy = typename XlaChainedPolicy<Key, cub::NullType>::MaxPolicy;
  return {
      /*single_tile=*/MakeKernelHostRef(
          &CubRadixSortSingleTileKeysKernel<Policy, kDirection, Key>),
  };
}

template <SortDirection kDirection, typename Key, typename Value>
CubSortTypeKernelRefs MakePairsRefs() {
  using Policy = typename XlaChainedPolicy<Key, Value>::MaxPolicy;
  return {
      /*single_tile=*/MakeKernelHostRef(
          &CubRadixSortSingleTilePairsKernel<Policy, kDirection, Key, Value>),
  };
}

template <PrimitiveType kKeyType>
struct CudaKeyType;
template <>
struct CudaKeyType<BF16> {
  using type = __nv_bfloat16;
};
template <>
struct CudaKeyType<F16> {
  using type = __half;
};
template <>
struct CudaKeyType<F32> {
  using type = float;
};
template <>
struct CudaKeyType<F64> {
  using type = double;
};
template <>
struct CudaKeyType<S8> {
  using type = int8_t;
};
template <>
struct CudaKeyType<S16> {
  using type = int16_t;
};
template <>
struct CudaKeyType<S32> {
  using type = int32_t;
};
template <>
struct CudaKeyType<S64> {
  using type = int64_t;
};
template <>
struct CudaKeyType<U8> {
  using type = uint8_t;
};
template <>
struct CudaKeyType<U16> {
  using type = uint16_t;
};
template <>
struct CudaKeyType<U32> {
  using type = uint32_t;
};
template <>
struct CudaKeyType<U64> {
  using type = uint64_t;
};

template <int kBitWidth>
struct CudaValueType;
template <>
struct CudaValueType<16> {
  using type = uint16_t;
};
template <>
struct CudaValueType<32> {
  using type = uint32_t;
};
template <>
struct CudaValueType<64> {
  using type = uint64_t;
};

}  // namespace

template <PrimitiveType kKeyType>
CubSortTypeKernelRefs GetCubSortKeysRefs(SortDirection direction) {
  using Key = typename CudaKeyType<kKeyType>::type;
  return direction == SortDirection::kDescending
             ? MakeKeysRefs<SortDirection::kDescending, Key>()
             : MakeKeysRefs<SortDirection::kAscending, Key>();
}

template <PrimitiveType kKeyType, int kValueBitWidth>
CubSortTypeKernelRefs GetCubSortPairsRefs(SortDirection direction) {
  using Key = typename CudaKeyType<kKeyType>::type;
  using Value = typename CudaValueType<kValueBitWidth>::type;
  return direction == SortDirection::kDescending
             ? MakePairsRefs<SortDirection::kDescending, Key, Value>()
             : MakePairsRefs<SortDirection::kAscending, Key, Value>();
}

// Floating point keys.
#ifdef CUB_TYPE_BF16
template CubSortTypeKernelRefs GetCubSortKeysRefs<BF16>(SortDirection);
#endif
#ifdef CUB_TYPE_F16
template CubSortTypeKernelRefs GetCubSortKeysRefs<F16>(SortDirection);
#endif
#ifdef CUB_TYPE_F32
template CubSortTypeKernelRefs GetCubSortKeysRefs<F32>(SortDirection);
#endif
#ifdef CUB_TYPE_F64
template CubSortTypeKernelRefs GetCubSortKeysRefs<F64>(SortDirection);
#endif

// Signed integer keys.
#ifdef CUB_TYPE_S8
template CubSortTypeKernelRefs GetCubSortKeysRefs<S8>(SortDirection);
#endif
#ifdef CUB_TYPE_S16
template CubSortTypeKernelRefs GetCubSortKeysRefs<S16>(SortDirection);
#endif
#ifdef CUB_TYPE_S32
template CubSortTypeKernelRefs GetCubSortKeysRefs<S32>(SortDirection);
#endif
#ifdef CUB_TYPE_S64
template CubSortTypeKernelRefs GetCubSortKeysRefs<S64>(SortDirection);
#endif

// Unsigned integer keys.
#ifdef CUB_TYPE_U8
template CubSortTypeKernelRefs GetCubSortKeysRefs<U8>(SortDirection);
#endif
#ifdef CUB_TYPE_U16
template CubSortTypeKernelRefs GetCubSortKeysRefs<U16>(SortDirection);
#endif
#ifdef CUB_TYPE_U32
template CubSortTypeKernelRefs GetCubSortKeysRefs<U32>(SortDirection);
#endif
#ifdef CUB_TYPE_U64
template CubSortTypeKernelRefs GetCubSortKeysRefs<U64>(SortDirection);
#endif

// Pairs with 8-bit key.
#ifdef CUB_TYPE_U8_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_U8_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_U8_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U8, 64>(SortDirection);
#endif

// Pairs with 16-bit key.
#ifdef CUB_TYPE_U16_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_U16_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_U16_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U16, 64>(SortDirection);
#endif

// Pairs with signed 32-bit key.
#ifdef CUB_TYPE_S32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_S32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_S32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<S32, 64>(SortDirection);
#endif

// Pairs with unsigned 32-bit key.
#ifdef CUB_TYPE_U32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_U32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_U32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U32, 64>(SortDirection);
#endif

// Pairs with 64-bit key.
#ifdef CUB_TYPE_U64_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_U64_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_U64_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<U64, 64>(SortDirection);
#endif

// Pairs with f32 key.
#ifdef CUB_TYPE_F32_B16
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 16>(SortDirection);
#endif
#ifdef CUB_TYPE_F32_B32
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 32>(SortDirection);
#endif
#ifdef CUB_TYPE_F32_B64
template CubSortTypeKernelRefs GetCubSortPairsRefs<F32, 64>(SortDirection);
#endif

}  // namespace xla::gpu::internal
