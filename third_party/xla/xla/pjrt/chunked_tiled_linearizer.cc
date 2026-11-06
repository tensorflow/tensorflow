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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/tiled_linearizer.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace {

struct ChunkedPlanCache {
  ChunkedPlanCache() : cache(1024) {}
  absl::Mutex mu;
  TransposePlanCache cache ABSL_GUARDED_BY(mu);
};

absl::NoDestructor<ChunkedPlanCache> chunked_plan_cache;

}  // namespace

/*static*/ std::shared_ptr<TransposePlan>
TiledLinearizer::CreateChunkedTransposePlan(
    bool is_h2d, const Shape& host_shape,
    std::optional<absl::Span<int64_t const>> host_byte_strides,
    const Shape& device_shape, int64_t target_chunk_bytes,
    int64_t offset_alignment, int64_t size_alignment) {
  if (target_chunk_bytes <= 0 || !device_shape.IsArray() ||
      !device_shape.is_static() || !device_shape.has_layout() ||
      !host_shape.IsArray() || !host_shape.is_static() ||
      host_shape.element_type() != device_shape.element_type() ||
      host_shape.dimensions() != device_shape.dimensions()) {
    return nullptr;
  }
  const Layout& device_layout = device_shape.layout();
  if (device_layout.tail_padding_alignment_in_elements() != 1 ||
      !device_layout.split_configs().empty() ||
      device_layout.dynamic_shape_metadata_prefix_bytes() != 0 ||
      device_layout.has_physical_shape()) {
    return nullptr;
  }
  absl::Span<int64_t const> dims = device_shape.dimensions();
  absl::Span<int64_t const> minor_to_major = device_layout.minor_to_major();
  if (dims.size() < 2 || minor_to_major.size() != dims.size()) {
    return nullptr;
  }
  absl::Span<const Tile> tiles = device_layout.tiles();
  if (tiles.empty() || tiles.size() > 2 || tiles[0].dimensions().size() != 2 ||
      tiles[0].dimension(0) == Tile::kMatchDimension ||
      tiles[0].dimension(1) == Tile::kMatchDimension) {
    return nullptr;
  }
  // Second-level tiling `tiles[1] = (packing_factor, 1)` packs `< 32`-bit
  // elements along the 2nd-minor dimension into 32-bit words.
  int64_t packing_factor = 1;
  if (tiles.size() == 2) {
    if (tiles[1].dimensions().size() != 2 || tiles[1].dimension(0) < 1 ||
        tiles[1].dimension(1) != 1) {
      return nullptr;
    }
    packing_factor = tiles[1].dimension(0);
  }
  if (tiles[0].dimension(0) % packing_factor != 0) {
    return nullptr;
  }
  const int64_t total_device_bytes = ShapeUtil::ArraySize(device_shape);
  if (total_device_bytes <= target_chunk_bytes) {
    return nullptr;
  }
  const int64_t elem_bits = ShapeUtil::ElementSizeInBits(device_shape);
  // Rejects:
  // - Sub-byte types (`elem_bits < 8`, e.g., `S4`/`U4`): `kPackSubbyte` in
  //   `TransposePlan` is currently scalar (`PackIntNStridedDst`) and slower
  //   than unchunked H2D linearization, and sub-byte unpacking on D2H is not
  //   yet implemented.
  //   TODO(b/575488723): Vectorize sub-byte packing (`kPackSubbyte`) for H2D
  //   and support sub-byte unpacking for D2H in `TransposePlan`.
  // - On-device element bit widths that differ from the primitive storage width
  //   (`elem_bits != primitive_util::StorageBitWidth(...)`, e.g., `E(16)` or
  //   `E(32)` padding on 8/16-bit types, or sub-byte types with `E(0)`).
  // - Packed D2H (`!is_h2d && packing_factor > 1`): 2-level packed D2H tiles
  //   `{T0/pf, 128, pf}` currently tie on outer stride in
  //   `TransposePlan::Create` and fall back to scalar `memcpy`.
  //   TODO(b/575488723): Break outer-stride ties using in-tile stride in
  //   `TransposePlan` for packed D2H.
  // - `> 32`-bit types (`elem_bits > 32`): TPU splits 64-bit elements into
  //   separate high/low 32-bit words across tiles.
  // - Non-byte-multiple widths or non-32-bit packed words.
  if (elem_bits < 8 || elem_bits > 32 || elem_bits % 8 != 0 ||
      elem_bits !=
          primitive_util::StorageBitWidth(device_shape.element_type()) ||
      (!is_h2d && packing_factor > 1) || elem_bits * packing_factor != 32) {
    return nullptr;
  }
  const int64_t second_minor_dim = minor_to_major[1];
  if (dims[second_minor_dim] % packing_factor != 0) {
    return nullptr;
  }

  // TransposePlan models second-level packing (sub-tile (packing_factor, 1)) as
  // an extra minor dimension of size `packing_factor` split off
  // `second_minor_dim`, with 3D device tiling
  // {tiles[0].dimension(0) / packing_factor, tiles[0].dimension(1),
  //  packing_factor}.
  constexpr int64_t kMaxChunks = 1024;
  TransposePlan::Options options;
  options.elem_size_in_bytes = elem_bits / 8;
  options.num_threads = static_cast<int>(
      std::min(kMaxChunks,
               CeilOfRatio<int64_t>(total_device_bytes, target_chunk_bytes)));
  absl::InlinedVector<int64_t, 5> plan_dims;
  absl::InlinedVector<int64_t, 5> permutation;
  absl::InlinedVector<int64_t, 5> input_strides;
  absl::InlinedVector<int64_t, 3> device_tiling = {
      tiles[0].dimension(0) / packing_factor, tiles[0].dimension(1)};
  if (packing_factor > 1) {
    device_tiling.push_back(packing_factor);
  }
  auto has_unpacked_host_layout = [&](const Shape& s) {
    if (!s.has_layout()) {
      return false;
    }
    const Layout& l = s.layout();
    return l.minor_to_major().size() == dims.size() && l.tiles().empty() &&
           l.element_size_in_bits() == 0 &&
           l.tail_padding_alignment_in_elements() == 1 &&
           l.split_configs().empty() &&
           l.dynamic_shape_metadata_prefix_bytes() == 0 &&
           !l.has_physical_shape();
  };

  if (is_h2d) {
    input_strides.resize(dims.size());
    if (host_byte_strides && !host_byte_strides->empty()) {
      if (host_byte_strides->size() != dims.size()) {
        return nullptr;
      }
      absl::c_copy(*host_byte_strides, input_strides.begin());
    } else if (!has_unpacked_host_layout(host_shape) ||
               !ShapeUtil::UnpackedByteStrides(host_shape,
                                               absl::MakeSpan(input_strides))
                    .ok()) {
      return nullptr;
    }
    plan_dims.assign(dims.begin(), dims.end());
    permutation.assign(minor_to_major.rbegin(), minor_to_major.rend());
    if (packing_factor > 1) {
      const int64_t stride = input_strides[second_minor_dim];
      plan_dims[second_minor_dim] /= packing_factor;
      input_strides[second_minor_dim] = stride * packing_factor;
      permutation.push_back(plan_dims.size());
      plan_dims.push_back(packing_factor);
      input_strides.push_back(stride);
    }
    options.input_striding = TransposePlan::Striding{input_strides};
    options.output_tiling = TransposePlan::Tiling{device_tiling};
    options.chunk_contiguity = TransposePlan::ChunkContiguity::kOutput;
  } else {
    DCHECK(!host_byte_strides.has_value());
    DCHECK_EQ(packing_factor, 1);
    if (!has_unpacked_host_layout(host_shape)) {
      return nullptr;
    }
    absl::InlinedVector<int64_t, 4> logical_to_phys(dims.size());
    for (auto it = minor_to_major.rbegin(); it != minor_to_major.rend(); ++it) {
      logical_to_phys[*it] = plan_dims.size();
      plan_dims.push_back(dims[*it]);
    }
    absl::Span<int64_t const> host_m2m = host_shape.layout().minor_to_major();
    for (auto it = host_m2m.rbegin(); it != host_m2m.rend(); ++it) {
      permutation.push_back(logical_to_phys[*it]);
    }
    options.input_tiling = TransposePlan::Tiling{device_tiling};
    options.chunk_contiguity = TransposePlan::ChunkContiguity::kInput;
  }
  options.dims = plan_dims;
  options.permutation = permutation;

  std::shared_ptr<TransposePlan> plan;
  {
    absl::MutexLock lock(chunked_plan_cache->mu);
    absl::StatusOr<std::shared_ptr<TransposePlan>> cached =
        chunked_plan_cache->cache.GetOrCreate(options);
    if (!cached.ok() || (*cached)->Parallelism() <= 1) {
      return nullptr;
    }
    plan = *std::move(cached);
  }
  offset_alignment = std::max<int64_t>(1, offset_alignment);
  size_alignment =
      std::max(offset_alignment, std::max<int64_t>(1, size_alignment));
  if (offset_alignment > 1 || size_alignment > 1) {
    for (int i = 0; i < plan->Parallelism(); ++i) {
      const int64_t offset = is_h2d ? plan->OutputChunkOffsetBytes(i)
                                    : plan->InputChunkOffsetBytes(i);
      const int64_t size =
          is_h2d ? plan->OutputChunkSizeBytes(i) : plan->InputChunkSizeBytes(i);
      if (offset % offset_alignment != 0 || size % size_alignment != 0) {
        return nullptr;
      }
    }
  }
  return plan;
}

}  // namespace xla
