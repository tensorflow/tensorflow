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

#ifndef XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_

#include <cstdint>

#if NCCL_VERSION_CODE >= 22900
#include "third_party/nccl/nccl_device.h"
#endif  // NCCL_VERSION_CODE >= 22900

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 &&    \
    defined(CUDART_VERSION) && CUDART_VERSION >= 13030 && \
    NCCL_VERSION_CODE >= 23200
#define XLA_GPU_ENABLE_NCCL_CFT 1
#else
#define XLA_GPU_ENABLE_NCCL_CFT 0
#endif  // XLA_GPU_ENABLE_NCCL_CFT

namespace stream_executor::gpu {

template <int64_t kSize>
struct alignas(kSize) DeviceVec {
  uint8_t data[kSize];
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
namespace ragged_all_to_all_tma {

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr,
                                              uint32_t arrive_count) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;\n" ::"r"(arrive_count),
               "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(
    uint64_t* mbar_ptr, int num_bytes) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n" ::"r"(
                   num_bytes),
               "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile(
      "mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr,
                                              uint32_t& phase) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
      "@P1 bra DONE_%=;\n"
      "bra LAB_WAIT_%=;\n"
      "DONE_%=:\n"
      "}\n" ::"r"(mbar_int_ptr),
      "r"(phase), "r"(0x989680));
  phase ^= 1;
}

__device__ __forceinline__ void tma_load_1d(const void* smem_ptr,
                                            const void* gmem_ptr,
                                            uint64_t* mbar_ptr, int num_bytes) {
  auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];\n" ::"r"(smem_int_ptr),
      "l"(gmem_ptr), "r"(num_bytes), "r"(mbar_int_ptr)
      : "memory");
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr,
                                             const void* gmem_ptr,
                                             int num_bytes) {
  auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group"
      " [%0], [%1], %2;\n" ::"l"(gmem_ptr),
      "r"(smem_int_ptr), "r"(num_bytes)
      : "memory");
}

__device__ __forceinline__ void tma_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
}

template <int N>
__device__ __forceinline__ void tma_store_wait_write() {
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_fabric_release() {
  asm volatile("fence.proxy.generic::fabric.alias.release.sys;\n" ::: "memory");
}

}  // namespace ragged_all_to_all_tma
#endif  // __CUDA_ARCH__ >= 900

#if NCCL_VERSION_CODE >= 22900

template <int64_t kVectorSize>
struct RaggedAllToAllUpdateMetadata {
  int peer;
  int update;
  int64_t meta_idx;
  int64_t send_size;
  int64_t src_byte_offset;
  int64_t dst_byte_offset;
  int64_t byte_count;
};

template <int64_t kVectorSize>
__device__ bool LoadRaggedAllToAllUpdateMetadata(
    int64_t flat_idx, int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    RaggedAllToAllUpdateMetadata<kVectorSize>* meta) {
  meta->peer = flat_idx / num_updates_per_replica;
  meta->update = flat_idx % num_updates_per_replica;
  meta->meta_idx = meta->peer * num_updates_per_replica + meta->update;
  meta->send_size = send_sizes_ptr[meta->meta_idx];
  if (meta->send_size == 0) {
    return false;
  }

  const int64_t input_offset = input_offsets_ptr[meta->meta_idx];
  const int64_t output_offset = output_offsets_ptr[meta->meta_idx];
  meta->src_byte_offset =
      input_buffer_offset_bytes + input_offset * num_row_elements * kVectorSize;
  meta->dst_byte_offset = output_buffer_offset_bytes +
                          output_offset * num_row_elements * kVectorSize;
  meta->byte_count = meta->send_size * num_row_elements * kVectorSize;
  return true;
}

// Shared by RaggedAllToAllTmaUnitCopy and RaggedAllToAllCftUnitCopy: two
// separate `__shared__` allocations would exceed the 48 KB static cap.
inline constexpr int kTmaStageBytes = 8192;
inline constexpr int kTmaStages = 4;
static_assert((kTmaStages & (kTmaStages - 1)) == 0,
              "kTmaStages must be a power of 2 (ring index uses bitmask).");
static_assert(kTmaStageBytes % 16 == 0,
              "kTmaStageBytes must be a multiple of 16 (cp.async.bulk size).");

struct RaggedAllToAllTmaSharedStorage {
  alignas(128) uint8_t tma_smem[kTmaStages][kTmaStageBytes];
  alignas(16) uint64_t full_mbar[kTmaStages];
  alignas(16) uint64_t empty_mbar[kTmaStages];
#if XLA_GPU_ENABLE_NCCL_CFT
  alignas(16) ncclCftSmem cft_smem_storage;
#endif  // XLA_GPU_ENABLE_NCCL_CFT
};
static_assert(
    sizeof(RaggedAllToAllTmaSharedStorage) <= 49152,
    "RaggedAllToAllTmaSharedStorage must fit in 48 KB static __shared__.");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// Copies `byte_count` bytes from `src` (a local send-window pointer) to
// `dst` (a peer recv-window pointer), staged through shared memory using
// TMA.
//
// One producer warp and one consumer warp, lane 0 of each doing the work:
//   Producer (warp 0, lane 0): global -> shared, via cp.async.bulk.
//   Consumer (warp 1, lane 0): shared -> peer global, via cp.async.bulk.
//
// The staging buffer is a kTmaStages-deep ring. Each slot has two
// mbarriers: full_mbar (producer marks the slot full) and empty_mbar
// (consumer marks it free again). The consumer keeps up to kTmaInFlight
// stores outstanding and then waits on the oldest.
//
// Each stage is copied by a single lane. Splitting a stage across all
// 32 lanes would drop each cp.async.bulk below TMA's per-descriptor
// peak-throughput range (1-32 KB) and multiply commit_group bookkeeping
// by 32x.
//
__device__ __forceinline__ void RaggedAllToAllTmaUnitCopy(
    RaggedAllToAllTmaSharedStorage* storage, const void* src, void* dst,
    int64_t byte_count, int my_cta_in_unit, int ctas_per_unit) {
  constexpr int kTmaInFlight = 3;
  constexpr int kProducerWarp = 0;
  constexpr int kConsumerWarp = 1;
  static_assert(kTmaInFlight < kTmaStages,
                "kTmaInFlight must be smaller than kTmaStages.");

  auto(&tma_smem)[kTmaStages][kTmaStageBytes] = storage->tma_smem;
  auto(&full_mbar)[kTmaStages] = storage->full_mbar;
  auto(&empty_mbar)[kTmaStages] = storage->empty_mbar;

  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTmaStages; ++s) {
      ragged_all_to_all_tma::mbarrier_init(&full_mbar[s], 1);
      ragged_all_to_all_tma::mbarrier_init(&empty_mbar[s], 1);
    }
    ragged_all_to_all_tma::fence_barrier_init();
  }
  __syncthreads();

  const int64_t total_chunks =
      (byte_count + kTmaStageBytes - 1) / kTmaStageBytes;

  if (warp_id == kProducerWarp && lane == 0) {
    uint32_t empty_phase[kTmaStages] = {0};
    int prod_k = 0;
    for (int64_t c = my_cta_in_unit; c < total_chunks; c += ctas_per_unit) {
      const int s = prod_k & (kTmaStages - 1);
      if (prod_k >= kTmaStages) {
        ragged_all_to_all_tma::mbarrier_wait(&empty_mbar[s], empty_phase[s]);
      }
      const int64_t off = c * static_cast<int64_t>(kTmaStageBytes);
      const int64_t remaining = byte_count - off;
      const int cur_bytes = static_cast<int>(
          remaining < kTmaStageBytes ? remaining : kTmaStageBytes);
      ragged_all_to_all_tma::mbarrier_arrive_and_expect_tx(&full_mbar[s],
                                                           cur_bytes);
      ragged_all_to_all_tma::tma_load_1d(tma_smem[s],
                                         static_cast<const uint8_t*>(src) + off,
                                         &full_mbar[s], cur_bytes);
      ++prod_k;
    }
  } else if (warp_id == kConsumerWarp && lane == 0) {
    // Single-lane consumer: one cp.async.bulk + one commit_group per stage.
    uint32_t full_phase[kTmaStages] = {0};
    int cons_k = 0;
    for (int64_t c = my_cta_in_unit; c < total_chunks; c += ctas_per_unit) {
      const int s = cons_k & (kTmaStages - 1);
      ragged_all_to_all_tma::mbarrier_wait(&full_mbar[s], full_phase[s]);

      const int64_t off = c * static_cast<int64_t>(kTmaStageBytes);
      const int64_t remaining = byte_count - off;
      const int cur_bytes = static_cast<int>(
          remaining < kTmaStageBytes ? remaining : kTmaStageBytes);
      ragged_all_to_all_tma::tma_store_1d(
          &tma_smem[s][0], static_cast<uint8_t*>(dst) + off, cur_bytes);
      ragged_all_to_all_tma::tma_commit_group();
      ++cons_k;

      if (cons_k > kTmaInFlight) {
        ragged_all_to_all_tma::tma_store_wait<kTmaInFlight>();
        const int old_s = (cons_k - kTmaInFlight - 1) & (kTmaStages - 1);
        ragged_all_to_all_tma::mbarrier_arrive(&empty_mbar[old_s]);
      }
    }
    // Drain the last kTmaInFlight commit groups before we let the ring
    // buffer be reused. fence.proxy.async then publishes the TMA writes
    // so the LSA barrier that follows this function can see them.
    ragged_all_to_all_tma::tma_store_wait_write<0>();
    ragged_all_to_all_tma::fence_proxy_async();
  }
  __syncthreads();
}

#if XLA_GPU_ENABLE_NCCL_CFT
// Same TMA-staged producer as RaggedAllToAllTmaUnitCopy, but the consumer
// warp issues NCCL Compute Fabric Transport (CFT) puts to the peer's
// unicast logical endpoint instead of TMA stores. Must not be called for
// the self peer -- a fabric put to the issuing rank's own logical endpoint
// faults with CUDA_ERROR_ILLEGAL_ADDRESS.
__device__ __forceinline__ void RaggedAllToAllCftUnitCopy(
    RaggedAllToAllTmaSharedStorage* storage, const void* src, ncclCftLeId leId,
    size_t leOffset, int64_t byte_count, int my_cta_in_unit,
    int ctas_per_unit) {
  constexpr int kProducerWarp = 0;
  constexpr int kConsumerWarp = 1;

  auto(&tma_smem)[kTmaStages][kTmaStageBytes] = storage->tma_smem;
  auto(&full_mbar)[kTmaStages] = storage->full_mbar;
  auto(&empty_mbar)[kTmaStages] = storage->empty_mbar;
  ncclCftSmem& cft_smem_storage = storage->cft_smem_storage;

  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTmaStages; ++s) {
      ragged_all_to_all_tma::mbarrier_init(&full_mbar[s], 1);
      ragged_all_to_all_tma::mbarrier_init(&empty_mbar[s], 1);
    }
    ragged_all_to_all_tma::fence_barrier_init();
  }
  __syncthreads();

  const int64_t total_chunks =
      (byte_count + kTmaStageBytes - 1) / kTmaStageBytes;

  if (warp_id == kProducerWarp && lane == 0) {
    uint32_t empty_phase[kTmaStages] = {0};
    int prod_k = 0;
    for (int64_t c = my_cta_in_unit; c < total_chunks; c += ctas_per_unit) {
      const int s = prod_k & (kTmaStages - 1);
      if (prod_k >= kTmaStages) {
        ragged_all_to_all_tma::mbarrier_wait(&empty_mbar[s], empty_phase[s]);
      }
      const int64_t off = c * static_cast<int64_t>(kTmaStageBytes);
      const int64_t remaining = byte_count - off;
      const int cur_bytes = static_cast<int>(
          remaining < kTmaStageBytes ? remaining : kTmaStageBytes);
      ragged_all_to_all_tma::mbarrier_arrive_and_expect_tx(&full_mbar[s],
                                                           cur_bytes);
      ragged_all_to_all_tma::tma_load_1d(tma_smem[s],
                                         static_cast<const uint8_t*>(src) + off,
                                         &full_mbar[s], cur_bytes);
      ++prod_k;
    }
  } else if (warp_id == kConsumerWarp && lane == 0) {
    ncclCft<ncclCoopThread> cft_op{ncclCoopThread{}, cft_smem_storage};
    uint32_t full_phase[kTmaStages] = {0};
    int cons_k = 0;
    for (int64_t c = my_cta_in_unit; c < total_chunks; c += ctas_per_unit) {
      const int s = cons_k & (kTmaStages - 1);
      ragged_all_to_all_tma::mbarrier_wait(&full_mbar[s], full_phase[s]);

      const int64_t off = c * static_cast<int64_t>(kTmaStageBytes);
      const int64_t remaining = byte_count - off;
      const uint32_t cur_bytes = static_cast<uint32_t>(
          remaining < kTmaStageBytes ? remaining : kTmaStageBytes);
      cft_op.put(ncclCoopThread{}, leId, leOffset + off, &tma_smem[s][0],
                 cur_bytes);
      cft_op.submit(ncclCoopThread{});
      // flushSmem retires the smem side of the put so the ring slot is
      // free for the producer to refill; the fabric-side transfer is
      // drained by cft_op.flush() after the loop.
      cft_op.flushSmem(ncclCoopThread{});
      ragged_all_to_all_tma::mbarrier_arrive(&empty_mbar[s]);
      ++cons_k;
    }
    cft_op.flush(ncclCoopThread{});
    ragged_all_to_all_tma::fence_proxy_fabric_release();
  }
  __syncthreads();
}
#endif  // XLA_GPU_ENABLE_NCCL_CFT

#endif  // __CUDA_ARCH__ >= 900

template <int64_t kVectorSize>
__device__ void RaggedAllToAllCopy(
    const struct ncclDevComm& dev_comm, ncclWindow_t send_win,
    ncclWindow_t recv_win, const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes,
    int start_lsa, int lsa_size, int num_ranks, ncclGin* gin, ncclTeam world,
    unsigned int signal_index, [[maybe_unused]] bool cft_granted,
    [[maybe_unused]] RaggedAllToAllTmaSharedStorage* tma_storage) {
  (void)dev_comm;
  using T = DeviceVec<kVectorSize>;

  if (gin != nullptr) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nthreads = blockDim.x * gridDim.x;
    const int64_t total_updates = num_updates_per_replica * num_ranks;

    for (int64_t flat_idx = tid; flat_idx < total_updates;
         flat_idx += nthreads) {
      const int peer = flat_idx / num_updates_per_replica;
      if (peer >= start_lsa && peer < start_lsa + lsa_size) {
        continue;
      }

      RaggedAllToAllUpdateMetadata<kVectorSize> meta;
      if (!LoadRaggedAllToAllUpdateMetadata<kVectorSize>(
              flat_idx, num_updates_per_replica, num_row_elements,
              input_buffer_offset_bytes, output_buffer_offset_bytes,
              input_offsets_ptr, send_sizes_ptr, output_offsets_ptr, &meta)) {
        continue;
      }

      gin->put(world, meta.peer, recv_win, meta.dst_byte_offset, send_win,
               meta.src_byte_offset, meta.byte_count,
               ncclGin_SignalInc{signal_index});
    }
  }

  if (lsa_size > 0) {
    // Work-proportional CTA assignment: CTA k copies the global element range
    // [total*k/grid, total*(k+1)/grid), walking update boundaries as needed.
    // This keeps CTA work balanced regardless of how transferred bytes are
    // distributed across updates. Every CTA scans the update-size array,
    // then walks updates in peer-major order to locate its range.
    const int64_t grid = static_cast<int64_t>(gridDim.x);
    const int64_t num_lsa_updates =
        static_cast<int64_t>(lsa_size) * num_updates_per_replica;
    const int64_t meta_base =
        static_cast<int64_t>(start_lsa) * num_updates_per_replica;

    // Compute the CTA range once and publish it to all threads in the block.
    __shared__ int64_t shared_cta_begin;
    __shared__ int64_t shared_cta_end;
    if (threadIdx.x == 0) {
      int64_t total_elements = 0;
      for (int64_t update = 0; update < num_lsa_updates; ++update) {
        total_elements += send_sizes_ptr[meta_base + update] * num_row_elements;
      }
      shared_cta_begin =
          total_elements * static_cast<int64_t>(blockIdx.x) / grid;
      shared_cta_end =
          total_elements * (static_cast<int64_t>(blockIdx.x) + 1) / grid;
    }
    __syncthreads();
    const int64_t cta_begin = shared_cta_begin;
    const int64_t cta_end = shared_cta_end;

    // Element offset of the current update within the concatenated element
    // space that cta_begin/cta_end index into.
    int64_t update_begin = 0;
    for (int64_t update = 0; update < num_lsa_updates && update_begin < cta_end;
         ++update) {
      RaggedAllToAllUpdateMetadata<kVectorSize> meta;
      if (!LoadRaggedAllToAllUpdateMetadata<kVectorSize>(
              meta_base + update, num_updates_per_replica, num_row_elements,
              input_buffer_offset_bytes, output_buffer_offset_bytes,
              input_offsets_ptr, send_sizes_ptr, output_offsets_ptr, &meta)) {
        continue;
      }
      const int64_t update_end = update_begin + meta.byte_count / kVectorSize;
      if (update_end > cta_begin) {
        const int64_t lo =
            cta_begin > update_begin ? cta_begin - update_begin : 0;
        const int64_t hi =
            (cta_end < update_end ? cta_end : update_end) - update_begin;
        const int lsa_peer = static_cast<int>(update / num_updates_per_replica);
        const T* src = static_cast<const T*>(
            ncclGetLocalPointer(send_win, meta.src_byte_offset));
        T* dst = static_cast<T*>(
            ncclGetLsaPointer(recv_win, meta.dst_byte_offset, lsa_peer));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const T* slice_src = src + lo;
        T* slice_dst = dst + lo;
        const int64_t slice_bytes = (hi - lo) * kVectorSize;
        const bool tma_aligned =
            (slice_bytes & 15) == 0 &&
            (reinterpret_cast<uintptr_t>(slice_src) & 15) == 0 &&
            (reinterpret_cast<uintptr_t>(slice_dst) & 15) == 0;
        if (!tma_aligned) {
          for (int64_t i = lo + static_cast<int64_t>(threadIdx.x); i < hi;
               i += static_cast<int64_t>(blockDim.x)) {
            dst[i] = src[i];
          }
#if XLA_GPU_ENABLE_NCCL_CFT
        } else if (cft_granted) {
          if (lsa_peer == world.rank - start_lsa) {
            for (int64_t i = lo + static_cast<int64_t>(threadIdx.x); i < hi;
                 i += static_cast<int64_t>(blockDim.x)) {
              dst[i] = src[i];
            }
            __syncthreads();
          } else {
            ncclCftLeId leId;
            size_t leOffset;
            ncclGetPeerLeInfo(recv_win, meta.dst_byte_offset + lo * kVectorSize,
                              /*peerWorld=*/start_lsa + lsa_peer, dev_comm,
                              &leId, &leOffset);
            RaggedAllToAllCftUnitCopy(tma_storage, slice_src, leId, leOffset,
                                      slice_bytes, 0, 1);
          }
#endif  // XLA_GPU_ENABLE_NCCL_CFT
        } else {
          RaggedAllToAllTmaUnitCopy(tma_storage, slice_src, slice_dst,
                                    slice_bytes, 0, 1);
        }
#else
        for (int64_t i = lo + static_cast<int64_t>(threadIdx.x); i < hi;
             i += static_cast<int64_t>(blockDim.x)) {
          dst[i] = src[i];
        }
#endif  // __CUDA_ARCH__ >= 900
      }
      update_begin = update_end;
    }
  }
}

template <int64_t kVectorSize>
__global__ void __launch_bounds__(512) RaggedAllToAllDeviceKernelImpl(
    struct ncclDevComm dev_comm, ncclWindow_t send_win, ncclWindow_t recv_win,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes,
    bool cft_granted) {
  // NCCL device barrier/GIN APIs emit scope-qualified atomics that require
  // sm_60+. Lower architectures compile to an empty stub; the kernel is only
  // launched when the device supports NCCL device comms.
#if __CUDA_ARCH__ >= 600
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ RaggedAllToAllTmaSharedStorage tma_storage;
  RaggedAllToAllTmaSharedStorage* tma_storage_ptr = &tma_storage;
#else
  RaggedAllToAllTmaSharedStorage* tma_storage_ptr = nullptr;
#endif  // __CUDA_ARCH__ >= 900

  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclTeam lsa = ncclTeamLsa(dev_comm);
  const int start_lsa = world.rank - lsa.rank;
  const int lsa_size = lsa.nRanks;
  const int num_ranks = world.nRanks;
  const bool has_remote_peers = (lsa_size < num_ranks);

  if (has_remote_peers) {
    const int n_ctxs = static_cast<int>(dev_comm.ginContextCount);
    const int my_ctx = static_cast<int>(blockIdx.x) % n_ctxs;
    const unsigned int signal_index = 0;

    ncclGin gin{dev_comm, my_ctx};

    ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin,
                                        blockIdx.x};
    bar.sync(ncclCoopCta(), ::cuda::memory_order_acquire,
             ncclGinFenceLevel::Relaxed);

    RaggedAllToAllCopy<kVectorSize>(
        dev_comm, send_win, recv_win, input_offsets_ptr, send_sizes_ptr,
        output_offsets_ptr, num_updates_per_replica, num_row_elements,
        input_buffer_offset_bytes, output_buffer_offset_bytes, start_lsa,
        lsa_size, num_ranks, &gin, world, signal_index, cft_granted,
        tma_storage_ptr);

    gin.flush(ncclCoopCta());
    bar.sync(ncclCoopCta(), ::cuda::memory_order_release,
             ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
  } else {
    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dev_comm,
                                           ncclTeamTagLsa{}, blockIdx.x};
    bar.sync(ncclCoopCta(), ::cuda::memory_order_relaxed);

    RaggedAllToAllCopy<kVectorSize>(
        dev_comm, send_win, recv_win, input_offsets_ptr, send_sizes_ptr,
        output_offsets_ptr, num_updates_per_replica, num_row_elements,
        input_buffer_offset_bytes, output_buffer_offset_bytes, start_lsa,
        lsa_size, num_ranks, /*gin=*/nullptr, world, /*signal_index=*/0,
        cft_granted, tma_storage_ptr);

    bar.sync(ncclCoopCta(), ::cuda::memory_order_release);
  }
#endif  // __CUDA_ARCH__ >= 600
}

#else  // NCCL_VERSION_CODE < 22900

template <int64_t kVectorSize>
__global__ void RaggedAllToAllDeviceKernelImpl(
    void* dev_comm, void* send_win, void* recv_win,
    const int64_t* input_offsets_ptr, const int64_t* send_sizes_ptr,
    const int64_t* output_offsets_ptr, int64_t num_updates_per_replica,
    int64_t num_row_elements, int64_t input_buffer_offset_bytes,
    int64_t output_buffer_offset_bytes, bool cft_granted) {}

#endif  // NCCL_VERSION_CODE >= 22900

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_
