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

#include "xla/pjrt/strided_gather.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/pjrt/transpose.h"
#include "xla/util.h"

namespace xla {

namespace {

bool NextIndex(absl::Span<int64_t> indices, absl::Span<const int64_t> limits) {
  CHECK_EQ(indices.size(), limits.size());

  for (int i = indices.size() - 1; i >= 0; --i) {
    int64_t& index = indices[i];
    if (index + 1 < limits[i]) {
      ++index;
      std::fill(indices.begin() + i + 1, indices.end(), 0);
      return true;
    }
  }
  return false;
}

// Canonicalizes `spec` by dropping size-1 stride levels, folding contiguous
// inner stride levels into `element_size`, and merging adjacent stride levels
// that are contiguous in the source buffer.
StridedGather::Spec CanonicalizeSpec(StridedGather::Spec spec) {
  CHECK_EQ(spec.counts.size(), spec.src_byte_strides.size());

  if (spec.element_size == 0 || absl::c_linear_search(spec.counts, 0)) {
    spec.element_size = 0;
    spec.counts = {0};
    spec.src_byte_offset = 0;
    spec.src_byte_size = spec.src_byte_size == 0 ? 0 : -1;
    spec.src_byte_strides = {0};
    return spec;
  }

  // Drop size-1 stride levels and merge adjacent stride levels that are
  // contiguous in the source buffer, from innermost to outermost. They are
  // always contiguous in the output since it is compact.
  DimensionVector counts;
  std::vector<int64_t> src_byte_strides;
  for (int i = spec.counts.size() - 1; i >= 0; --i) {
    if (spec.counts[i] == 1) {
      continue;
    }
    if (counts.empty() && spec.src_byte_strides[i] == spec.element_size) {
      // All inner stride levels so far are contiguous in the source buffer;
      // fold this stride level into the contiguous chunk size `element_size`.
      spec.element_size *= spec.counts[i];
    } else if (!counts.empty() && spec.src_byte_strides[i] ==
                                      src_byte_strides.back() * counts.back()) {
      // An inner stride level was strided, but this stride level's source
      // stride continues the immediately more-inner stride level
      // (`counts.back()`) without a gap, so collapse them into a single stride
      // level.
      counts.back() *= spec.counts[i];
    } else {
      counts.push_back(spec.counts[i]);
      src_byte_strides.push_back(spec.src_byte_strides[i]);
    }
  }
  absl::c_reverse(counts);
  absl::c_reverse(src_byte_strides);

  spec.counts = std::move(counts);
  spec.src_byte_strides = std::move(src_byte_strides);
  return spec;
}

}  // namespace

StridedGather::StridedGather(Spec spec)
    : spec_(CanonicalizeSpec(std::move(spec))) {}

int64_t StridedGather::num_chunks() const {
  return absl::c_accumulate(spec_.counts, int64_t{1}, std::multiplies<>());
}

void StridedGather::CreateTransposePlan(
    absl::FunctionRef<void(const TransposePlan::Options&)> create_fn) const {
  // `xla::TransposePlan` only supports power-of-two `elem_size_in_bytes` up to
  // 16 bytes, so decompose larger chunks by adding a minormost dimension.
  const int64_t element_size = std::gcd<int64_t>(spec_.element_size, 16);

  DimensionVector dims = spec_.counts;
  std::vector<int64_t> strides = spec_.src_byte_strides;

  if (spec_.element_size > element_size) {
    dims.push_back(spec_.element_size / element_size);
    strides.push_back(element_size);
  }

  std::vector<int64_t> permutation(dims.size());
  absl::c_iota(permutation, 0);

  TransposePlan::Options options;
  options.elem_size_in_bytes = element_size;
  options.dims = dims;
  options.permutation = permutation;
  options.input_striding =
      TransposePlan::Striding{/*strides_in_bytes=*/strides};
  options.output_tiling = TransposePlan::Tiling{};

  create_fn(options);
}

StridedGather::ChunkRange::ChunkIterator::ChunkIterator(
    const StridedGather* absl_nonnull gather)
    : gather_(gather),
      index_(gather->spec_.counts.size(), 0),
      chunk_({
          /*offset=*/gather->spec_.src_byte_offset,
          /*size=*/gather->chunk_size(),
      }) {}

StridedGather::ChunkRange::ChunkIterator&
StridedGather::ChunkRange::ChunkIterator::operator++() {
  CHECK_NE(gather_, nullptr);
  if (const absl::Span<const int64_t> counts = gather_->spec_.counts;
      NextIndex(absl::MakeSpan(index_), counts)) {
    int64_t offset = gather_->spec_.src_byte_offset;
    for (int i = 0; i < counts.size(); ++i) {
      offset += index_[i] * gather_->spec_.src_byte_strides[i];
    }
    chunk_ = Chunk{/*offset=*/offset, /*size=*/gather_->chunk_size()};
  } else {
    gather_ = nullptr;
    index_.clear();
    chunk_ = Chunk{/*offset=*/0, /*size=*/0};
  }
  return *this;
}

StridedGather::ChunkRange::ChunkIterator StridedGather::ChunkRange::begin()
    const {
  return gather_->num_chunks() > 0 ? ChunkIterator(gather_) : ChunkIterator();
}

StridedGather::ChunkRange::ChunkIterator StridedGather::ChunkRange::end()
    const {
  return ChunkIterator();
}

StridedGather::ChunkRange StridedGather::chunks() const {
  return ChunkRange(this);
}

}  // namespace xla
