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

#include "xla/backends/cpu/collectives/gloo_reduce_scatter.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gloo/common/error.h"
#include "gloo/context.h"
#include "gloo/math.h"
#include "gloo/transport/unbound_buffer.h"
#include "gloo/types.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

// Halving-doubling reduce-scatter across `world_size` ranks, inspired by
// Gloo's `ReduceScatterHalvingDoubling` (`gloo/reduce_scatter.h`), adapted to
// Gloo's `UnboundBuffer` transport API with per-operation timeouts.
//
// Each rank starts with `total_count = world_size * count` elements and
// finishes with the sum of its own shard `[rank * count, (rank + 1) * count)`
// across all ranks.
//
// We divide the buffer into `chunks = 2^floor(log2(world_size))` chunks (the
// largest power of two <= `world_size`) and run three phases.
//
// Running example: `world_size = 6` (`= 4 + 2`), so `chunks = 4` and the
// buffer is divided into 4 chunks `[A, B, C, D]`.
//
// Phase 1 (Intra-block recursive halving):
//   Recursive halving requires a power-of-two number of ranks, so we split the
//   ranks into power-of-two blocks matching the binary bits of `world_size`.
//   For `world_size = 6`:
//     - Block 0 (size 4): ranks 0, 1, 2, 3
//     - Block 1 (size 2): ranks 4, 5
//
//   Within each block of size 2^k, ranks run k halving steps LSB-first
//   (`bitmask = 1, 2, ..., 2^{k-1}`):
//     - Step 0: pair ranks at within-block distance 1 (`rank_in_block ^ 1`).
//       Even ranks keep the first half of their active buffer; odd ranks keep
//       the second half. Each pair exchanges the half it is discarding and
//       reduces into the half it keeps.
//     - Step 1: pair ranks at within-block distance 2 (`rank_in_block ^ 2`)
//       and repeat on the half kept in step 0.
//     - ...
//     - Step i: pair ranks at within-block distance 2^i (`rank_in_block ^
//     2^i`).
//   Stepping LSB-first (rather than textbook MSB-first) ensures that smaller
//   and larger binary blocks share identical halving splits for their first
//   `log2(B_small)` steps, allowing each rank `r` in `B_large` to receive from
//   a single rank `r % B_small` in `B_small` in Phase 2.
//
//   In our 6-rank example:
//     - Block 0 (ranks 0..3) runs 2 steps:
//         Step 0: pair (0, 1) and (2, 3).
//                 Ranks 0, 2 keep [A, B]; ranks 1, 3 keep [C, D].
//         Step 1: pair (0, 2) and (1, 3).
//                 Rank 0 (00) keeps A (chunk 0 = 00)  \
//                 Rank 1 (01) keeps C (chunk 2 = 10)   | reduced across
//                 Rank 2 (10) keeps B (chunk 1 = 01)   | ranks 0..3
//                 Rank 3 (11) keeps D (chunk 3 = 11)  /
//       Notice that within a block of size 2^k, `rank_in_block` ends up holding
//       chunk `ReverseLastNBits(rank_in_block, k)`.
//     - Block 1 (ranks 4..5) runs 1 step:
//         Step 0: pair (4, 5).
//                 Rank 4 keeps [A, B]  \ reduced across
//                 Rank 5 keeps [C, D]  / ranks 4..5
//
// Phase 2 (Inter-block reduction, only when `world_size` is not a power of 2):
//   After Phase 1, each block only has the sum across its own members, and a
//   rank in a smaller block of size B_small holds `num_sends = B_large /
//   B_small` times as much data as each rank in the next larger block of size
//   B_large. So each rank in the smaller block splits its reduced slice into
//   `num_sends` pieces and sends each piece to the rank in the larger block
//   that holds that same chunk.
//
//   In our 6-rank example (`num_sends = 4 / 2 = 2`):
//     - Rank 4 holds [A, B]: it sends A to rank 0 and B to rank 2.
//     - Rank 5 holds [C, D]: it sends C to rank 1 and D to rank 3.
//     - Ranks 0..3 add those incoming chunks into their own chunks.
//   If there are more than two blocks (e.g. `world_size = 7 = 4 + 2 + 1`), this
//   cascades from smallest to largest (block of 1 -> block of 2 -> block of 4).
//   At the end of Phase 2, the largest block holds the sum across all ranks:
//     Rank 0 has A, Rank 1 has C, Rank 2 has B, Rank 3 has D.
//
// Phase 3 (Distribution):
//   Phase 3 serves two purposes:
//     1. Even when `world_size` is a power of two, Phase 1's LSB-first halving
//        leaves chunk `c` on rank `ReverseLastNBits(c, steps)`, so Phase 3
//        unscrambles that bit-reversal permutation (e.g. swapping ranks 1 and 2
//        when `world_size = 4`) and copies self-owned shards from `work` into
//        `recv_ptr`.
//     2. When `world_size` is not a power of two, it simultaneously re-slices
//        the `chunks = 2^steps` chunks of size
//        `chunk_size = ceil(total_count / chunks)` into `world_size` output
//        shards of size `count`.
//
//   In our 6-rank example, the 4 chunks `[A, B, C, D]` each have size
//   `chunk_size = ceil(total_count / 4)`, whereas the 6 output shards each have
//   size `count = total_count / 6`. Because the 4 chunks and the 6 output
//   shards divide the same array at different boundaries, each chunk covers
//   parts of one or more output shards.
//
//   For example, if `count = 2` (`total_count = 12` elements `0..11`), then
//   each chunk has `chunk_size = 3` elements and each rank wants `count = 2`:
//
//     Chunks held after Phase 2:        Output shards wanted by each rank:
//       A = [0, 1, 2]   (on rank 0)       rank 0 wants [0, 1]
//       B = [3, 4, 5]   (on rank 2)       rank 1 wants [2, 3]
//       C = [6, 7, 8]   (on rank 1)       rank 2 wants [4, 5]
//       D = [9, 10, 11] (on rank 3)       rank 3 wants [6, 7]
//                                         rank 4 wants [8, 9]
//                                         rank 5 wants [10, 11]
//
//   Each rank in the largest block sends each piece of its chunk to the rank
//   that wants those elements:
//     - Rank 0 (holds A = [0, 1, 2]):  keeps [0, 1], sends [2] to rank 1.
//     - Rank 2 (holds B = [3, 4, 5]):  sends [3] to rank 1, keeps [4, 5].
//     - Rank 1 (holds C = [6, 7, 8]):  sends [6, 7] to rank 3, [8] to rank 4.
//     - Rank 3 (holds D = [9, 10, 11]): sends [9] to rank 4, [10, 11] to
//     rank 5.

namespace xla::cpu {
namespace {

using ReduceFn = void (*)(void*, const void*, const void*, size_t);

template <typename T>
absl::StatusOr<ReduceFn> GetReduceFunction(ReductionKind reduction_kind) {
  switch (reduction_kind) {
    case ReductionKind::SUM:
      return static_cast<ReduceFn>(&gloo::sum<T>);
    case ReductionKind::PRODUCT:
      return static_cast<ReduceFn>(&gloo::product<T>);
    case ReductionKind::MIN:
      if constexpr (!is_complex_v<T>) {
        return static_cast<ReduceFn>(&gloo::min<T>);
      } else {
        return absl::InvalidArgumentError(
            "MIN reduction not supported for complex types");
      }
    case ReductionKind::MAX:
      if constexpr (!is_complex_v<T>) {
        return static_cast<ReduceFn>(&gloo::max<T>);
      } else {
        return absl::InvalidArgumentError(
            "MAX reduction not supported for complex types");
      }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported reduction kind: ", static_cast<int>(reduction_kind)));
}

}  // namespace

namespace internal {

uint32_t ReverseLastNBits(uint32_t ctr, uint32_t n) {
  if (n == 0) {
    return 0;
  }
#if defined(__has_builtin) && __has_builtin(__builtin_bitreverse32)
  return __builtin_bitreverse32(ctr) >> (32 - n);
#else
  ctr = ((ctr >> 1) & 0x55555555) | ((ctr & 0x55555555) << 1);
  ctr = ((ctr >> 2) & 0x33333333) | ((ctr & 0x33333333) << 2);
  ctr = ((ctr >> 4) & 0x0f0f0f0f) | ((ctr & 0x0f0f0f0f) << 4);
  ctr = ((ctr >> 8) & 0x00ff00ff) | ((ctr & 0x00ff00ff) << 8);
  return ((ctr >> 16) | (ctr << 16)) >> (32 - n);
#endif
}

}  // namespace internal

namespace {

using internal::ReverseLastNBits;

// Membership of a single rank in the power-of-two block decomposition of
// `world_size` (see Phase 1 above).
struct BlockMembership {
  // Starting rank ID of this rank's block (in ranks). This block covers ranks
  // `[offset_to_my_block, offset_to_my_block + my_block_size)`.
  uint32_t offset_to_my_block = 0;

  // Number of ranks in this block (always a power of two).
  uint32_t my_block_size = 0;

  // Number of ranks in the adjacent smaller block (at rank IDs immediately
  // above this block), or 0 if this is already the smallest block.
  uint32_t next_smaller_block_size = 0;

  // Number of ranks in the adjacent larger block (at rank IDs immediately
  // below this block), or 0 if this is already the largest block.
  uint32_t next_larger_block_size = 0;
};

BlockMembership GetBlockMembership(uint32_t world_size, uint32_t rank) {
  BlockMembership info;
  // Walk the set bits of `world_size` from largest block to smallest block,
  // assigning contiguous rank ranges starting at rank 0.
  // For example, for world_size = 7 (4 + 2 + 1):
  //   bit 4 -> ranks [0, 4)
  //   bit 2 -> ranks [4, 6)
  //   bit 1 -> rank  [6, 7)
  uint32_t offset = 0;
  uint32_t larger_block_size = 0;
  for (uint32_t block_size = uint32_t{1} << Log2Floor(world_size);
       block_size > 0; block_size >>= 1) {
    if ((world_size & block_size) == 0) {
      continue;
    }
    if (rank >= offset && rank < offset + block_size) {
      info.offset_to_my_block = offset;
      info.my_block_size = block_size;
      info.next_larger_block_size = larger_block_size;
      // The remaining ranks `world_size - (offset + block_size)` live to our
      // right; the next block to our right is the highest set bit of that
      // remainder (or 0 if no ranks remain).
      uint32_t rem = world_size - (offset + block_size);
      info.next_smaller_block_size =
          rem == 0 ? 0 : (uint32_t{1} << Log2Floor(rem));
      break;
    }
    offset += block_size;
    larger_block_size = block_size;
  }
  return info;
}

// Slice of `work` that holds this rank's reduced elements after Phase 1.
struct ReducedSlice {
  // Starting element index in `work`.
  int64_t offset = 0;

  // Number of valid elements starting at `offset` (`count >= 0`).
  int64_t count = 0;
};

// Phase 1: Recursive halving within each power-of-two block.
ReducedSlice ReduceWithinBlock(const std::shared_ptr<gloo::Context>& context,
                               const BlockMembership& block_info,
                               uint32_t steps, int64_t chunk_size,
                               int64_t total_count, size_t element_size,
                               ReduceFn reduce_fn, gloo::Slot base_slot,
                               std::chrono::system_clock::time_point deadline,
                               std::vector<char>& work) {
  const uint32_t steps_within_block = Log2Floor(block_info.my_block_size);
  int64_t step_chunk_size = chunk_size << (steps - 1);
  if (steps_within_block == 0) {
    return ReducedSlice{0, total_count};
  }

  const uint32_t rank = context->rank;
  // `[active_offset, active_offset + active_count)` is the slice of `work`
  // currently owned by this rank. Each step splits the active slice in half:
  // this rank abandons one half (sending it to `dest_rank`) and keeps the other
  // half (reducing `dest_rank`'s data into it), so after the final step
  // `[active_offset, active_offset + active_count)` is the single reduced slice
  // this rank holds for Phases 2 and 3.
  int64_t active_offset = 0;
  int64_t active_count = total_count;

  std::vector<char> scratch(step_chunk_size * element_size);
  auto work_buf = context->createUnboundBuffer(work.data(), work.size());
  auto scratch_buf =
      context->createUnboundBuffer(scratch.data(), scratch.size());

  for (uint32_t i = 0, bitmask = 1; i < steps_within_block;
       ++i, bitmask <<= 1) {
    // Because `offset_to_my_block` is a multiple of `my_block_size` and
    // `bitmask < my_block_size`, `rank ^ bitmask` equals
    // `offset_to_my_block + (rank_in_block ^ bitmask)`.
    const uint32_t dest_rank = rank ^ bitmask;

    // Element offsets into `work` for this step:
    //   - `step_send_offset`: start of the half this rank sends to `dest_rank`
    //     (`[step_send_offset, step_send_offset + send_count)`).
    //   - `step_recv_offset`: start of the half this rank keeps and reduces the
    //     received data into
    //     (`[step_recv_offset, step_recv_offset + recv_count)`).
    const int64_t step_send_offset =
        active_offset + ((dest_rank & bitmask) ? step_chunk_size : 0);
    const int64_t step_recv_offset =
        active_offset + ((rank & bitmask) ? step_chunk_size : 0);

    const int64_t send_count = std::max<int64_t>(
        0, std::min(step_chunk_size, total_count - step_send_offset));
    const int64_t recv_count = std::max<int64_t>(
        0, std::min(step_chunk_size, total_count - step_recv_offset));

    const auto step_slot = base_slot + static_cast<uint8_t>(i);
    if (send_count > 0) {
      work_buf->send(dest_rank, step_slot, step_send_offset * element_size,
                     send_count * element_size);
    }
    if (recv_count > 0) {
      scratch_buf->recv(dest_rank, step_slot, 0, recv_count * element_size);
    }
    if (send_count > 0) {
      work_buf->waitSend(deadline);
    }
    if (recv_count > 0) {
      scratch_buf->waitRecv(deadline);
      char* dst = work.data() + step_recv_offset * element_size;
      reduce_fn(dst, dst, scratch.data(), recv_count);
    }

    active_offset = step_recv_offset;
    active_count = recv_count;

    step_chunk_size >>= 1;
  }

  return ReducedSlice{active_offset, active_count};
}

// Phase 2: Fold smaller blocks into larger blocks (non-power-of-two sizes).
//
// Blocks fold from smallest up to largest (e.g. for 7 = 4 + 2 + 1, block 1
// folds into block 2, which then folds into block 4). A rank in a smaller
// block holds a slice `num_sends = larger_size / my_size` times larger than
// a rank in the next larger block, so it splits its slice into `num_sends`
// pieces and sends them to the corresponding ranks in the larger block.
void ReduceAcrossBlocks(const std::shared_ptr<gloo::Context>& context,
                        const BlockMembership& block_info, uint32_t steps,
                        int64_t chunk_size, int64_t total_count,
                        size_t element_size, ReduceFn reduce_fn,
                        gloo::Slot base_slot,
                        std::chrono::system_clock::time_point deadline,
                        const ReducedSlice& slice, std::vector<char>& work) {
  const uint32_t rank_in_block = context->rank - block_info.offset_to_my_block;
  const auto inter_block_slot = base_slot + static_cast<uint8_t>(steps);
  if (block_info.next_smaller_block_size != 0) {
    // The next smaller block starts immediately after this block's rank range
    // `[offset_to_my_block, offset_to_my_block + my_block_size)`.
    const uint32_t offset_to_smaller_block =
        block_info.offset_to_my_block + block_info.my_block_size;
    // Because Phase 1 runs halving steps LSB-first, ranks in this block share
    // the same first `log2(next_smaller_block_size)` halving choices as the
    // smaller-block rank with the same low bits (`rank_in_block %
    // next_smaller_block_size`), whose slice therefore contains our chunk.
    const uint32_t src_rank =
        offset_to_smaller_block +
        (rank_in_block % block_info.next_smaller_block_size);
    if (slice.count > 0) {
      std::vector<char> smaller_scratch(slice.count * element_size);
      auto recv_buf = context->createUnboundBuffer(smaller_scratch.data(),
                                                   smaller_scratch.size());
      recv_buf->recv(src_rank, inter_block_slot);
      recv_buf->waitRecv(deadline);
      char* dst = work.data() + slice.offset * element_size;
      reduce_fn(dst, dst, smaller_scratch.data(), slice.count);
    }
  }

  if (block_info.next_larger_block_size != 0) {
    // The next larger block immediately precedes this block's starting rank:
    // `[offset_to_my_block - next_larger_block_size, offset_to_my_block)`.
    const uint32_t offset_to_larger_block =
        block_info.offset_to_my_block - block_info.next_larger_block_size;
    // Our slice is `num_sends` times wider than a rank's slice in the larger
    // block, so we split our slice into `num_sends` pieces of size
    // `send_count_to_larger` (the chunk size of ranks in that larger block).
    const uint32_t num_sends =
        block_info.next_larger_block_size / block_info.my_block_size;
    const int64_t send_count_to_larger =
        chunk_size << (steps - Log2Floor(block_info.next_larger_block_size));
    // Map this sender rank to the destination ranks in the larger block:
    //   1. `src_ordinal` is which slice of our block (`0 .. my_block_size - 1`,
    //      in left-to-right array order) this rank holds after Phase 1.
    //      In our 6-rank example (`my_block_size = 2`):
    //        - rank 4 (`rank_in_block = 0`) -> `src_ordinal = 0` (`[A, B]`)
    //        - rank 5 (`rank_in_block = 1`) -> `src_ordinal = 1` (`[C, D]`)
    //   2. Splitting slice `src_ordinal` into `num_sends` pieces produces chunk
    //      indices `dest_ordinal + i` (`i = 0 .. num_sends - 1`) in the larger
    //      block:
    //        - `src_ordinal = 0` (`[A, B]`) -> chunks `0, 1` (`A`, `B`)
    //        - `src_ordinal = 1` (`[C, D]`) -> chunks `2, 3` (`C`, `D`)
    //   3. In the larger block, chunk `c` lives on within-block rank
    //      `ReverseLastNBits(c, log2(next_larger_block_size))`:
    //        - chunks `0, 1` (`A, B`) -> ranks `0, 2`
    //        - chunks `2, 3` (`C, D`) -> ranks `1, 3`.
    const uint32_t src_ordinal =
        ReverseLastNBits(rank_in_block, Log2Floor(block_info.my_block_size));
    uint32_t dest_ordinal = src_ordinal * num_sends;

    std::vector<std::unique_ptr<gloo::transport::UnboundBuffer>> send_bufs;
    for (uint32_t i = 0; i < num_sends; ++i) {
      // Global rank ID in the larger block that holds chunk `dest_ordinal + i`.
      const uint32_t dest_rank =
          offset_to_larger_block +
          ReverseLastNBits(dest_ordinal + i,
                           Log2Floor(block_info.next_larger_block_size));
      // Starting element index in `work` of sub-piece `i` within our slice.
      const int64_t my_offset = slice.offset + i * send_count_to_larger;
      const int64_t item_count = std::max<int64_t>(
          0, std::min(send_count_to_larger, total_count - my_offset));
      if (item_count > 0) {
        auto sb = context->createUnboundBuffer(
            work.data() + my_offset * element_size, item_count * element_size);
        sb->send(dest_rank, inter_block_slot);
        send_bufs.push_back(std::move(sb));
      }
    }
    for (auto& sb : send_bufs) {
      sb->waitSend(deadline);
    }
  }
}

// Describes one contiguous slice transfer in Phase 3.
//
// Phase 3 serves two purposes:
//   1. Even for power-of-two `world_size`, Phase 1's LSB-first halving leaves
//      chunk `c` on rank `ReverseLastNBits(c, steps)`, so Phase 3 unscrambles
//      that bit-reversal permutation and copies self-owned shards from `work`
//      into `recv_ptr`.
//   2. For non-power-of-two `world_size`, `chunk_size > count`, so source
//      chunks in the largest block and destination shards don't align 1-to-1.
//      For example, with world_size = 3 and count = 4 (N = 12, chunks = 2,
//      chunk_size = 6):
//        source chunks:      [   0 .. 6   )[   6 .. 12   )
//        destination shards: [ 0..4 )[ 4..8 )[ 8..12 )
//      Rank 1's output shard [4, 8) must be assembled from two pieces: [4, 6)
//      from source chunk 0 and [6, 8) from source chunk 1.
struct DistributionEntry {
  // Peer rank ID for this transfer:
  // - In `dist_map_for_send`: destination rank (`0 .. world_size - 1`).
  // - In `dist_map_for_recv`: source rank (`0 .. chunks - 1`).
  uint32_t rank;

  // Starting element index of this piece in the global `[0, N)` array (in
  // elements). On the sender, the data lives at `work + offset * element_size`.
  size_t offset;

  // Number of elements in this piece (`item_count * element_size` bytes).
  size_t item_count;
};

// Finds which peer ranks overlap with `[src_offset, src_offset + src_count)`
// given per-bucket sizes `recv_counts`.
//
// Used in two directions in Phase 3:
// - Sending (`bit_reverse_ranks = false`): a rank in the largest block has
//   reduced elements `[src_offset, src_offset + src_count)` and wants to know
//   which target ranks `t` need pieces of it (bucket `i` -> rank `i`).
// - Receiving (`bit_reverse_ranks = true`): target rank `t` wants elements
//   `[t * count, (t + 1) * count)` and needs to know which ranks in the largest
//   block hold them (bucket `i` -> rank `ReverseLastNBits(i, steps)`).
std::vector<DistributionEntry> GetDistributionMap(
    size_t src_offset, int64_t src_count,
    const std::vector<size_t>& recv_counts, uint32_t world_size,
    bool bit_reverse_ranks) {
  std::vector<DistributionEntry> dist_map;
  // Because `chunk_size = ceil(total_count / chunks)`, the last chunk(s) can
  // lie entirely beyond `total_count` (e.g. `world_size = 5, count = 1` gives
  // `chunk_size = 2`, so chunk 3 starts at offset 6 >= `total_count = 5`),
  // giving `src_count == 0`.
  if (src_count <= 0) {
    return dist_map;
  }

  const uint32_t steps = Log2Floor(world_size);
  const size_t size = recv_counts.size();

  // Advance `start` to the first bucket in `recv_counts` that overlaps with
  // `src_offset`, accumulating the starting element index of bucket `start` in
  // `dest_offset`.
  size_t dest_offset = 0;
  size_t start = 0;
  for (; start < size; ++start) {
    if (dest_offset + recv_counts[start] > src_offset) {
      break;
    }
    dest_offset += recv_counts[start];
  }
  // Convert `dest_offset` from bucket `start`'s global start offset into the
  // within-bucket offset where our slice begins (`0 <= dest_offset <
  // recv_counts[start]`).
  dest_offset = src_offset - dest_offset;

  // Walk consecutive buckets `i = start, start + 1, ...` and slice
  // `[src_offset, src_offset + src_count)` along bucket boundaries until all
  // `total_count` elements have been assigned to peer ranks.
  size_t total_count = src_count;
  for (size_t i = start; i < size; ++i) {
    // Number of elements available in bucket `i` starting at `src_offset`.
    // Only the first overlapping bucket (`i == start`) can start mid-bucket.
    size_t recv_count = recv_counts[i];
    if (dest_offset != 0) {
      recv_count -= dest_offset;
      dest_offset = 0;
    }
    // Peer rank that owns bucket `i`: either destination rank `i` (when
    // sending) or `ReverseLastNBits(i, steps)` in the largest block (when
    // receiving).
    const uint32_t rank = bit_reverse_ranks ? ReverseLastNBits(i, steps)
                                            : static_cast<uint32_t>(i);
    // Clamp to the remaining unassigned elements of our slice.
    recv_count = std::min(recv_count, total_count);
    if (recv_count > 0) {
      dist_map.push_back(DistributionEntry{rank, src_offset, recv_count});
      src_offset += recv_count;
      total_count -= recv_count;
    }
    if (total_count == 0) {
      break;
    }
  }
  return dist_map;
}

// Phase 3: Send final reduced shards from the largest block to destination
// ranks.
void DistributeShards(const std::shared_ptr<gloo::Context>& context,
                      const BlockMembership& block_info, uint32_t steps,
                      size_t chunks, size_t chunk_size, size_t count,
                      size_t element_size, gloo::Slot base_slot,
                      std::chrono::system_clock::time_point deadline,
                      const ReducedSlice& slice, std::vector<char>& work,
                      void* recv_ptr) {
  const uint32_t world_size = context->size;
  const uint32_t rank = context->rank;

  // Send map (only populated on ranks in the largest block, where
  // `next_larger_block_size == 0`): slice this rank's reduced chunk
  // `[slice.offset, slice.offset + slice.count)` against the `world_size`
  // output shards of size `count` to find which destination rank wants each
  // piece.
  std::vector<DistributionEntry> dist_map_for_send;
  if (block_info.next_larger_block_size == 0) {
    std::vector<size_t> recv_elems(world_size, count);
    dist_map_for_send =
        GetDistributionMap(slice.offset, slice.count, recv_elems, world_size,
                           /*bit_reverse_ranks=*/false);
  }

  // Receive map (populated on every rank): slice this rank's wanted output
  // shard `[rank * count, (rank + 1) * count)` against the `chunks` reduced
  // chunks of size `chunk_size` in the largest block to find which source rank
  // holds each piece (using `bit_reverse_ranks = true` because chunk `i` lives
  // on rank `ReverseLastNBits(i, steps)`).
  // Unlike Gloo, we pass uniform `chunk_size` for all `chunks` buckets without
  // clamping tail buckets to `total_count`; these are equivalent because the
  // clamped and unclamped prefix sums agree up through element `total_count -
  // 1` and no shard offset exceeds `total_count`.
  std::vector<size_t> recv_counts_largest_block(chunks, chunk_size);
  const size_t my_dst_offset = rank * count;
  const std::vector<DistributionEntry> dist_map_for_recv =
      GetDistributionMap(my_dst_offset, count, recv_counts_largest_block,
                         world_size, /*bit_reverse_ranks=*/true);

  const auto dist_slot = base_slot + static_cast<uint8_t>(steps + 1);
  std::vector<std::unique_ptr<gloo::transport::UnboundBuffer>> dist_sends;
  std::vector<std::unique_ptr<gloo::transport::UnboundBuffer>> dist_recvs;
  char* out_bytes = static_cast<char*>(recv_ptr);

  // Post receives for all remote pieces of our output shard directly into their
  // contiguous positions in `recv_ptr`. Self-pieces (`entry.rank == rank`) are
  // skipped here and copied locally below, while still advancing `out_offset`.
  size_t out_offset = 0;
  for (const auto& entry : dist_map_for_recv) {
    if (entry.rank != rank) {
      auto rb =
          context->createUnboundBuffer(out_bytes + out_offset * element_size,
                                       entry.item_count * element_size);
      rb->recv(entry.rank, dist_slot);
      dist_recvs.push_back(std::move(rb));
    }
    out_offset += entry.item_count;
  }

  // Send each piece of our reduced chunk to its destination rank (or memcpy
  // directly into `recv_ptr` if the destination is this rank itself).
  for (const auto& entry : dist_map_for_send) {
    if (entry.rank == rank) {
      // Find the element offset within `recv_ptr` where our self-piece belongs.
      size_t local_recv_offset = 0;
      for (const auto& recv_entry : dist_map_for_recv) {
        if (recv_entry.rank == rank) {
          break;
        }
        local_recv_offset += recv_entry.item_count;
      }
      std::memcpy(out_bytes + local_recv_offset * element_size,
                  work.data() + entry.offset * element_size,
                  entry.item_count * element_size);
    } else {
      auto sb = context->createUnboundBuffer(
          work.data() + entry.offset * element_size,
          entry.item_count * element_size);
      sb->send(entry.rank, dist_slot);
      dist_sends.push_back(std::move(sb));
    }
  }

  // Wait for all distribution sends and receives to complete.
  for (auto& rb : dist_recvs) {
    rb->waitRecv(deadline);
  }
  for (auto& sb : dist_sends) {
    sb->waitSend(deadline);
  }
}

absl::Status RunReduceScatterHalvingDoubling(
    const std::shared_ptr<gloo::Context>& context, const void* send_ptr,
    void* recv_ptr, size_t count, size_t element_size, ReduceFn reduce_fn,
    absl::Duration timeout, uint32_t tag) {
  const uint32_t world_size = context->size;
  const uint32_t rank = context->rank;
  const size_t total_count = count * world_size;
  const size_t chunk_bytes = count * element_size;

  // All ranks in the collective pass the same `count` and `world_size`, so all
  // ranks take this early exit together without any communication.
  if (count == 0 || world_size == 0) {
    return absl::OkStatus();
  }
  if (world_size == 1) {
    if (recv_ptr != send_ptr) {
      std::memmove(recv_ptr, send_ptr, chunk_bytes);
    }
    return absl::OkStatus();
  }

  const uint32_t steps = Log2Floor(world_size);
  const size_t chunks = size_t{1} << steps;
  const size_t chunk_size = (total_count + chunks - 1) / chunks;
  const BlockMembership block_info = GetBlockMembership(world_size, rank);

  // Every send/recv/reduce offset in Phases 1, 2, and 3 is clamped against
  // `total_count`, so `work` is never accessed at or beyond `total_count` and
  // needs no tail padding.
  // TODO(phawkins): Reduce memory usage for large arrays by allocating `work`
  // for `total_count / 2` elements instead of `total_count`: step 0 can read
  // inputs directly from `send_ptr`, receive into `work`, and reduce in-place
  // into `work`, while subsequent steps can receive into the unused half of
  // `work` instead of allocating a separate `scratch` buffer.
  const char* send_bytes = static_cast<const char*>(send_ptr);
  std::vector<char> work(send_bytes, send_bytes + total_count * element_size);

  const auto base_slot = gloo::Slot::build(kReduceScatterSlotPrefix, tag);
  const auto deadline = absl::ToChronoTime(absl::Now() + timeout);

  // Note on `UnboundBuffer::waitSend` / `waitRecv`: Gloo throws
  // `gloo::IoException` when `deadline` expires. Their `bool` return value is
  // `false` only when explicitly aborted via `abortWaitSend` / `abortWaitRecv`,
  // which XLA never calls.
  try {
    const ReducedSlice slice =
        ReduceWithinBlock(context, block_info, steps, chunk_size, total_count,
                          element_size, reduce_fn, base_slot, deadline, work);

    ReduceAcrossBlocks(context, block_info, steps, chunk_size, total_count,
                       element_size, reduce_fn, base_slot, deadline, slice,
                       work);

    DistributeShards(context, block_info, steps, chunks, chunk_size, count,
                     element_size, base_slot, deadline, slice, work, recv_ptr);
  } catch (const gloo::IoException& e) {
    return absl::DeadlineExceededError(
        absl::StrCat("Gloo ReduceScatter timed out: ", e.what()));
  } catch (const std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo ReduceScatter failed: ", e.what()));
  }

  return absl::OkStatus();
}

absl::StatusOr<ReduceFn> GetReduceFunctionForType(
    PrimitiveType dtype, ReductionKind reduction_kind) {
  switch (dtype) {
    case S8:
      return GetReduceFunction<int8_t>(reduction_kind);
    case PRED:
    case U8:
      return GetReduceFunction<uint8_t>(reduction_kind);
    case S16:
      return GetReduceFunction<int16_t>(reduction_kind);
    case U16:
      return GetReduceFunction<uint16_t>(reduction_kind);
    case S32:
      return GetReduceFunction<int32_t>(reduction_kind);
    case U32:
      return GetReduceFunction<uint32_t>(reduction_kind);
    case S64:
      return GetReduceFunction<int64_t>(reduction_kind);
    case U64:
      return GetReduceFunction<uint64_t>(reduction_kind);
    case BF16:
      return GetReduceFunction<bfloat16>(reduction_kind);
    case F16:
      return GetReduceFunction<gloo::float16>(reduction_kind);
    case F32:
      return GetReduceFunction<float>(reduction_kind);
    case F64:
      return GetReduceFunction<double>(reduction_kind);
    case C64:
      return GetReduceFunction<std::complex<float>>(reduction_kind);
    case C128:
      return GetReduceFunction<std::complex<double>>(reduction_kind);
    default:
      return absl::InvalidArgumentError("Unknown datatype in reducescatter");
  }
}

}  // namespace

absl::Status GlooReduceScatter(const std::shared_ptr<gloo::Context>& context,
                               se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               absl::Duration timeout, uint32_t tag) {
  ABSL_ASSIGN_OR_RETURN(ReduceFn reduce_fn,
                   GetReduceFunctionForType(dtype, reduction_kind));
  return RunReduceScatterHalvingDoubling(
      context, send_buffer.opaque(), recv_buffer.opaque(), count,
      primitive_util::ByteWidth(dtype), reduce_fn, timeout, tag);
}

}  // namespace xla::cpu
