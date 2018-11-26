/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/sparse_index_array.h"

#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

SparseIndexArray::SparseIndexArray() : rank_(0), max_indices_(0) {}

SparseIndexArray::SparseIndexArray(int64 max_indices, int64 rank,
                                   std::vector<int64> indices)
    : indices_(std::move(indices)), rank_(rank), max_indices_(max_indices) {
  CHECK_GT(rank_, 0);
  CHECK_EQ(indices_.size() % rank_, 0)
      << "indices_.size(): " << indices_.size() << ", rank_: " << rank_;
  CHECK_LE(index_count(), max_indices_);
}

SparseIndexArray::SparseIndexArray(int64 max_indices, int64 rank,
                                   absl::Span<const int64> indices)
    : SparseIndexArray(max_indices, rank,
                       std::vector<int64>(indices.begin(), indices.end())) {}

SparseIndexArray::SparseIndexArray(int64 max_indices,
                                   const Array2D<int64>& indices)
    : SparseIndexArray(max_indices, indices.n2(),
                       std::vector<int64>(indices.begin(), indices.end())) {}

int64 SparseIndexArray::index_count() const {
  CHECK_GT(rank_, 0);
  CHECK_EQ(indices_.size() % rank_, 0);
  return indices_.size() / rank_;
}

absl::Span<const int64> SparseIndexArray::At(
    int64 sparse_element_number) const {
  CHECK_GT(rank_, 0);
  CHECK_GE(sparse_element_number, 0);
  CHECK_LE(rank_ * sparse_element_number + rank_, indices_.size());
  return absl::Span<const int64>(
      indices_.data() + rank_ * sparse_element_number, rank_);
}

absl::Span<int64> SparseIndexArray::At(int64 sparse_element_number) {
  CHECK_GT(rank_, 0);
  CHECK_GE(sparse_element_number, 0);
  CHECK_LE(rank_ * sparse_element_number + rank_, indices_.size());
  return absl::Span<int64>(indices_.data() + rank_ * sparse_element_number,
                           rank_);
}

void SparseIndexArray::Append(absl::Span<const int64> index) {
  CHECK_GT(rank_, 0);
  CHECK_EQ(index.size(), rank_);
  indices_.insert(indices_.end(), index.begin(), index.end());
}

void SparseIndexArray::Clear() { indices_.clear(); }

void SparseIndexArray::Resize(int64 num_indices) {
  CHECK_GT(rank_, 0);
  indices_.resize(rank_ * num_indices);
}

bool SparseIndexArray::Validate(const Shape& shape) const {
  if (rank_ == 0 || rank_ != ShapeUtil::Rank(shape)) {
    return false;
  }
  int64 num_indices = index_count();
  if (num_indices > LayoutUtil::MaxSparseElements(shape.layout())) {
    return false;
  }
  if (num_indices < 2) {
    return true;
  }
  absl::Span<const int64> last = At(0);
  if (!IndexUtil::IndexInBounds(shape, last)) {
    return false;
  }
  for (int64 n = 1; n < num_indices; ++n) {
    absl::Span<const int64> next = At(n);
    if (!IndexUtil::IndexInBounds(shape, next)) {
      return false;
    }
    if (IndexUtil::CompareIndices(last, next) >= 0) {
      return false;
    }
    last = next;
  }
  return true;
}

}  // namespace xla
