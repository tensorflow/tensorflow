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

#ifndef XLA_PJRT_STRIDED_GATHER_H_
#define XLA_PJRT_STRIDED_GATHER_H_

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/pjrt/transpose.h"
#include "xla/util.h"

namespace xla {

// Computes the sequence of contiguous byte ranges in a source buffer that form
// a sliced or reassembled destination buffer.
//
// `StridedGather` canonicalizes a `Spec` describing a strided gather of
// elements by dropping size-1 stride levels, folding contiguous inner stride
// levels into `element_size`, and merging adjacent stride levels that are
// contiguous in the source buffer, and lazily yields the resulting contiguous
// `(offset, size)` byte chunks.
class StridedGather {
 public:
  // Specification of a strided gather of elements from a source buffer into a
  // compact destination buffer.
  struct Spec {
    // Size of one element in bytes. An element is the unit that is never split,
    // e.g., one tile, one packed byte, or one contiguous chunk after
    // canonicalization.
    int64_t element_size;

    // Number of steps along each stride level, ordered from outermost to
    // innermost.
    DimensionVector counts;

    // Byte offset of the first element in the source buffer.
    int64_t src_byte_offset;

    // Total size of the source buffer in bytes, or -1 if the output is empty
    // and the source buffer is non-empty.
    int64_t src_byte_size;

    // Byte stride of each stride level in the source buffer.
    std::vector<int64_t> src_byte_strides;

    bool operator==(const Spec& other) const {
      return element_size == other.element_size && counts == other.counts &&
             src_byte_offset == other.src_byte_offset &&
             src_byte_size == other.src_byte_size &&
             src_byte_strides == other.src_byte_strides;
    }

    bool operator!=(const Spec& other) const { return !(*this == other); }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Spec& spec) {
      absl::Format(&sink,
                   "{element_size: %d, counts: [%s], src_byte_offset: %d, "
                   "src_byte_size: %d, src_byte_strides: [%s]}",
                   spec.element_size, absl::StrJoin(spec.counts, ", "),
                   spec.src_byte_offset, spec.src_byte_size,
                   absl::StrJoin(spec.src_byte_strides, ", "));
    }
  };

  // A contiguous byte range `[offset, offset + size)` in the source buffer.
  struct Chunk {
    // Byte offset of the chunk in the source buffer.
    int64_t offset;

    // Size of the chunk in bytes.
    int64_t size;

    bool operator==(const Chunk& other) const {
      return offset == other.offset && size == other.size;
    }

    bool operator!=(const Chunk& other) const { return !(*this == other); }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Chunk& chunk) {
      absl::Format(&sink, "{offset: %d, size: %d}", chunk.offset, chunk.size);
    }
  };

  class ChunkRange;

  explicit StridedGather(Spec spec);

  // Returns the specification of the strided gather.
  const Spec& spec() const { return spec_; }

  // Returns the number of contiguous chunks in the indexed sequence. Returns 0
  // if the output buffer is empty.
  int64_t num_chunks() const;

  // Returns the size of each contiguous chunk in bytes. Returns 0 if the output
  // buffer is empty.
  int64_t chunk_size() const { return spec_.element_size; }

  // Returns true if the output covers the entire source buffer in the same
  // order without any slicing or reordering.
  bool IsIdentity() const {
    return num_chunks() <= 1 && spec_.src_byte_offset == 0 &&
           chunk_size() == spec_.src_byte_size;
  }

  // Returns a range view over the contiguous byte chunks of the source buffer.
  ChunkRange chunks() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Creates a transpose plan that gathers the elements from the source buffer
  // into a compact destination buffer. The input pointer passed to
  // `TransposePlan::Execute` must be offset by `spec().src_byte_offset`.
  template <typename CreateFn>
  auto CreateTransposePlan(CreateFn&& create_fn) const {
    std::optional<decltype(create_fn(
        std::declval<const TransposePlan::Options&>()))>
        result;
    CreateTransposePlan(absl::FunctionRef<void(const TransposePlan::Options&)>(
        [&](const TransposePlan::Options& options) {
          result.emplace(std::forward<CreateFn>(create_fn)(options));
        }));
    return *std::move(result);
  }

 private:
  friend class ChunkRange;

  void CreateTransposePlan(
      absl::FunctionRef<void(const TransposePlan::Options&)> create_fn) const;

  // Specification of the strided gather.
  Spec spec_;
};

// Range view over the contiguous byte chunks of a `StridedGather`.
class StridedGather::ChunkRange {
 public:
  class ChunkIterator;

  using value_type = Chunk;
  using const_iterator = ChunkIterator;

  ChunkIterator begin() const;
  ChunkIterator end() const;

 private:
  friend class StridedGather;

  explicit ChunkRange(const StridedGather* absl_nonnull gather)
      : gather_(gather) {}

  // Parent gather.
  const StridedGather* absl_nonnull gather_;
};

// Input iterator over the chunks of a `StridedGather`.
class StridedGather::ChunkRange::ChunkIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Chunk;
  using difference_type = std::ptrdiff_t;
  using pointer = const Chunk*;
  using reference = Chunk;

  ChunkIterator() = default;

  reference operator*() const { return chunk_; }
  pointer operator->() const { return &chunk_; }

  ChunkIterator& operator++();
  ChunkIterator operator++(int) {
    ChunkIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool operator==(const ChunkIterator& other) const {
    return gather_ == other.gather_ &&
           (gather_ == nullptr || index_ == other.index_);
  }

  bool operator!=(const ChunkIterator& other) const {
    return !(*this == other);
  }

 private:
  friend class ChunkRange;

  explicit ChunkIterator(const StridedGather* absl_nonnull gather);

  // Parent gather, or nullptr for the past-the-end sentinel iterator.
  const StridedGather* absl_nullable gather_ = nullptr;

  // Current multi-dimensional index within gather_->spec_.counts.
  DimensionVector index_;

  // Current byte chunk corresponding to `index_`.
  Chunk chunk_ = {/*offset=*/0, /*size=*/0};
};

}  // namespace xla

#endif  // XLA_PJRT_STRIDED_GATHER_H_
