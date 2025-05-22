/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_TOPK_KERNEL_CUDA_COMMON_CU_H_
#define XLA_STREAM_EXECUTOR_CUDA_TOPK_KERNEL_CUDA_COMMON_CU_H_

// This file contains bespoke and optimized implementation for TopK shapes. When
// adding support for new shapes/dtypes, you also need to modify the rewriter
// on topk_specializer.cc for these changes to be picked up.

#include <cstddef>
#include <cstdint>

#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/topk_kernel.h"
#include "xla/tsl/lib/math/math_util.h"

#define WAVEFRONT_SIZE 32

namespace stream_executor::cuda {

enum class ShflType { kSync, kUp, kDown, kXor };

template <ShflType Type, class NT>
__device__ __forceinline__ NT GpuShuffle(NT val, uint32_t idx,
                                         uint32_t allmsk = 0xffffffffu) {
  constexpr uint32_t SZ =
      tsl::MathUtil::CeilOfRatio(sizeof(NT), sizeof(uint32_t));
  union S {
    NT v;
    uint32_t d[SZ];
  };
  S in{val}, res{};

#pragma unroll
  for (uint32_t i = 0; i < SZ; i++) {
    if constexpr (Type == ShflType::kSync)
      res.d[i] = __shfl_sync(allmsk, in.d[i], idx);
    else if constexpr (Type == ShflType::kUp)
      res.d[i] = __shfl_up_sync(allmsk, in.d[i], idx);
    else if constexpr (Type == ShflType::kDown)
      res.d[i] = __shfl_down_sync(allmsk, in.d[i], idx);
    else if constexpr (Type == ShflType::kXor)
      res.d[i] = __shfl_xor_sync(allmsk, in.d[i], idx);
  }
  return res.v;
}

// Default implementation for KV holder. Useful for testing while adding support
// for a new type, but generally bitpacking those values is more efficient. See
// implementations below.
template <typename T, typename V>
struct Descending {
  struct KVT {
    T key;
    V idx;
  };

  __device__ __forceinline__ static bool cmp(const KVT& lhs, const KVT& rhs) {
    return lhs.key == rhs.key ? lhs.idx < rhs.idx : lhs.key > rhs.key;
  }
};

// TopK implements a faster TopK for K < 16.
//
// To compute the final largest K elements, we shard the data threads and each
// of them computes the top k elements for the data in its slice. When all lanes
// in a warp are done with their TopK, we merge all the lane-local topks into
// lane 0 using warp-local reductions. The lane-local topk is computed at
// PerWarpTopK() and the warp reduction is computed in Reduce(). The warp-local
// results are stored in shared memory.
//
// Once all warps are done, we load all previously produced results into a
// single warp and repeat the reduction described above. This is implemented in
// MergeTopKs() and we reuse the Reduce() implementation described above. On
// MergeTopKs we also write the final results to the user-provided buffer.
//
// === Detailed Design
//
// The high level goals of this implementations are:
//  - Low latency for small N (i.e. kilobytes).
//  - High throughput for large N and/or large batch.
//
// Non-goals:
//  - K > 32. Register pressure will be too high.
//  - Sharding over multiple SMs. As explained later, we can use TopK's
//    structure to get this "for free".
//
// The core observation of this implementation is that reading/writing to main
// memory is the bottleneck in usual the Sort/TopK implementations and that for
// K<16 a linear scan with in-register data is faster than using a heap with
// shared memory, especially when K is a power of two.
//
// The heap for K=7 looks like:
//
//             a0
//        a1        a2
//      a3  a4    a5  a6
//
// When performing a push/pop, in the worst case scenario we need to compare it
// with the root, both of its children, and one of the two subtrees. This means
// that using a heap for K=7 only save us 2/7 comparison. Additionally, if the
// tree were unbalanced(e.g. K=8), we would not be able to unroll this
// computation.
//
// If we're using linear insertion, the worst case results in the full K
// comparisons comparisons, but with care all of those values can be kept in
// registers, replacing somewhat load/store instructions with movs. This
// performance are more than enough to surpass the heap.
//
// We split the data evenly over T (<=1024) threads, and use the algorithm above
// to maintain a sorted list of K elements in registers and perform linear
// insertions on every new element. Once a warp is done with their local slice,
// we reduce the slice-local data using shfl and the insertion described above,
// by adding the other lane's TopK results to the local lane. Once the warp is
// done, lane 0 writes its results to shared memory. This step has complexity:
//    theta(k * slice_size + k^2 * log2(k))
//
// On a second pass, we use a single warp to consume the results of the previous
// step and merge them into a final topk, using an analogous algorithm to what
// has been previously described. Complexity of this stage is:
//    theta(k^2 * log2(k)).
//
// This algorithm only uses a single block per batch dimension, but for large N,
// we can split the input into B batches of size N/B, calculate each of their
// topks and then compute a final topk, fixing the indices in the process.
//
// Future improvements:
//  - Use optimal sort/merge networks to reduce the complexity the algorithm and
//    allow better scaling past K=16. This is fairly tricky to implement
//    efficiently, so it was let out of v1.
//

template <size_t K, typename KT, typename VT,
          template <class, class> class Traits = Descending>
struct TopK {
  using Trait = Traits<KT, VT>;
  using KVT = typename Trait::KVT;

  __device__ TopK(void* buffer, int num_outputs)
      : buffer_(reinterpret_cast<KVT*>(buffer)), num_outputs_(num_outputs) {}

  __device__ __forceinline__ uint32_t Idx(uint32_t i) {
    return blockDim.x * i + threadIdx.x;
  }

  // Compute a per-warp topk of a slice of data.
  __device__ void PerWarpTopK(KT* key, int n) {
    KVT tmp[K];
    // TODO(doak): Use bitonic sort.
#pragma unroll
    for (int i = 0; i < K; i++) {
      tmp[i] = {key[Idx(i)], VT(Idx(i))};
    }
#pragma unroll
    for (int i = 0; i < K; i++) {
#pragma unroll
      for (int j = i + 1; j < K; j++) {
        KVT ti = tmp[i];
        KVT tj = tmp[j];
        bool res = Trait::cmp(ti, tj);
        tmp[i] = res ? ti : tj;
        tmp[j] = res ? tj : ti;
      }
    }
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;

    for (int idx = K; idx < n; idx++) {
      KVT kv{key[Idx(idx)], VT(Idx(idx))};
      Push(tmp, kv);
    }
    Reduce(tmp, WarpSize);

    if (threadIdx.x % WarpSize != 0) return;
    int warp_id = threadIdx.x / WarpSize;
#pragma unroll
    for (int i = 0; i < K; i++) {
      buffer_[i * WarpSize + warp_id] = tmp[i];
    }
  }

  // Merge the per-warp topks into a single topk. The final data is written to
  // `keys` and `idxs`
  __device__ void MergeTopKs(KT* keys, uint32_t* idxs) {
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
    KVT tmp[K];
    // We only use one warp for this step.
    if (threadIdx.x >= WarpSize) return;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < K; i++) {
      tmp[i] = buffer_[i * WarpSize + threadIdx.x];
    }
    Reduce(tmp, blockDim.x / WarpSize);
    if (threadIdx.x != 0) return;
    for (int i = 0; i < num_outputs_; ++i) {
      keys[i] = tmp[i].key;
      idxs[i] = tmp[i].idx;
    }
  }

  // Merge `tmp` (a reverse-sorted array) from (0, `num_lanes`) lanes. The
  // resulting array is stored in the tmp array of lane 0. For all other lanes,
  // `tmp` is unspecified after this function is called.
  __device__ __forceinline__ void Reduce(KVT tmp[K], int num_lanes) {
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
    int lane_id = threadIdx.x % WarpSize;
    for (int offset = num_lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int i = 0; i < K; i++) {
        KVT kv = GpuShuffle<ShflType::kDown>(tmp[i], offset);
        if (lane_id >= offset) continue;
        Push(tmp, kv);
      }
    }
  }

  // Given a K-array of previously reverse-sorted KVTs, add kv to it and
  // remove the smallest element of the resulting array. Preserves the sorted
  // order of `tmp`.
  // We are careful to write this code in a way that nvcc/ptxas will use
  // predication rather than branching. If we don't get this right, then we
  // can greatly expands the code size of the generated PTX and SASS by
  // tens of thousands of instructions. This increased the size of the
  // compressed JAX wheel by 25MiB, so be very careful to check the generated
  // code size when changing this function.
  static __device__ __forceinline__ void Push(KVT tmp[K], const KVT& kv) {
    bool p = Trait::cmp(tmp[K - 1], kv);
    tmp[K - 1] = p ? tmp[K - 1] : kv;
#pragma unroll
    for (int i = static_cast<int>(K) - 2; i >= 0; --i) {
      // Note: even though we could exit early as soon as the first time we
      // see a value greater than kv, we don't do this because it makes nvcc
      // generate terrible code.
      bool p = Trait::cmp(tmp[i], kv);
      auto t = tmp[i];
      tmp[i] = p ? tmp[i] : tmp[i + 1];
      tmp[i + 1] = p ? tmp[i + 1] : t;
    }
  }

  KVT* buffer_;
  int num_outputs_;
};

// This shared memory buffer needs to be declared outside of the templated
// Run(), as otherwise it would generate name conflicts from the multiple
// instantiations of Run() from the multiple monomorphizations of Run().
extern __device__ __shared__ int shmem[];

template <size_t K, typename KT, typename VT>
__launch_bounds__(stream_executor::gpu::kTopKMaxThreadsPerBlock, 1) __global__
    void Run(KT* data, int n, KT* result, uint32_t* result_idxs, int k) {
  TopK<K, KT, VT> obj(shmem, k);

  const uint32_t bidx = blockIdx.x;
  auto in = data + n * bidx;
  auto vals_out = result + k * bidx;
  auto idxs_out = result_idxs + k * bidx;
  int slice_size = n / blockDim.x;
  if (threadIdx.x < n % blockDim.x) {
    slice_size++;
  }

  obj.PerWarpTopK(in, slice_size);
  obj.MergeTopKs(vals_out, idxs_out);
}

#define KERNEL_TRAIT(K_VAL, TYPE, VT) \
  stream_executor::gpu::TopKKernel<K_VAL, TYPE, VT>
#define REGISTER_TOPK_KERNEL(K_VAL, TYPE, VT)                                 \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                             \
      TopKKernelCuda_K##K_VAL##_##TYPE##_##VT, KERNEL_TRAIT(K_VAL, TYPE, VT), \
      stream_executor::cuda::kCudaPlatformId, ([](size_t arity) {             \
        stream_executor::MultiKernelLoaderSpec spec(arity);                   \
        spec.AddInProcessSymbol(absl::bit_cast<void*>(&Run<K_VAL, TYPE, VT>), \
                                "topk_k" #K_VAL "_" #TYPE "_" #VT);           \
        return spec;                                                          \
      }));

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_TOPK_KERNEL_CUDA_COMMON_CU_H_
