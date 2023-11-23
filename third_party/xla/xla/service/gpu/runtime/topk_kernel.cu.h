/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_CU_H_
#define XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_CU_H_

// This file contains bespoke and optimized implementation for TopK shapes. When
// adding support for new shapes/dtypes, you also need to modify the rewritter
// on topk_specializer.cc for these changes to be picked up.

#include <cstddef>
#include <cstdint>
#include <limits>

#include "xla/service/gpu/runtime/gpu_kernel_helper.h"
#include "xla/service/gpu/runtime/topk_kernel_common.h"

namespace xla::gpu {

// Default implementation for KV holder. Useful for testing while adding support
// for a new type, but generally bitpacking those values is more efficient. See
// implementations below.
template <typename T, typename V>
struct Descending {
  struct KVT {
    T key;
    V idx;
  };

  __device__ FORCEINLINE static bool cmp(const KVT& lhs, const KVT& rhs) {
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
// that using a heap for K=7 only save us 2/7 comparions. Additionally, if the
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

  __device__ FORCEINLINE uint32_t Idx(uint32_t i) {
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
  __device__ FORCEINLINE void Reduce(KVT tmp[K], int num_lanes) {
    constexpr uint32_t WarpSize = WAVEFRONT_SIZE;
    int lane_id = threadIdx.x % WarpSize;
    for (int offset = num_lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int i = 0; i < K; i++) {
        KVT kv = GpuShuffle<ShflType::Down>(tmp[i], offset);
        if (lane_id >= offset) continue;
        Push(tmp, kv);
      }
    }
  }

  // Given a K-array of previously reverse-sorted KVTs, add kv to it and
  // remove the smallest element of the resulting array. Preserves the sorted
  // order of `tmp`.
  static __device__ FORCEINLINE bool Push(KVT tmp[K], const KVT& kv) {
    if (Trait::cmp(tmp[K - 1], kv)) return false;
    tmp[K - 1] = kv;  // (K-1)th is the smallest element out of K
#pragma unroll
    for (int i = (int)K - 2; i >= 0; --i) {
      if (Trait::cmp(tmp[i], kv)) break;
      // Swap
      auto t = tmp[i];
      tmp[i] = tmp[i + 1];
      tmp[i + 1] = t;
    }
    return true;
  }

  KVT* buffer_;
  int num_outputs_;
};

// This shared memory buffer needs to be declared outside of the templated
// Run(), as otherwise it would generate name conflicts from the multiple
// instantiations of Run() from the multiple monomorphizations of Run().
extern __device__ __shared__ int shmem[];

template <size_t K, typename KT, typename VT>
__launch_bounds__(kTopKMaxThreadsPerBlock, 1) __global__
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

template <typename T, size_t K>
void* GetTopKKernelForK(int n) {
  // TODO(doak): Switch to uint32_t if we don't have an efficient
  // implemementation for uint16_t.
  return n < std::numeric_limits<uint16_t>::max()
             ? reinterpret_cast<void*>(&Run<K, T, uint16_t>)
             : reinterpret_cast<void*>(&Run<K, T, uint32_t>);
}

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_TOPK_KERNEL_CU_H_
