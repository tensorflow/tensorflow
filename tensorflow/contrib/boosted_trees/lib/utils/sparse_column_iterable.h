// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_SPARSE_COLUMN_ITERABLE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_SPARSE_COLUMN_ITERABLE_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

// Enables row-wise iteration through examples on sparse feature columns.
class SparseColumnIterable {
 public:
  // Indicates a contiguous range for an example: [start, end).
  struct ExampleRowRange {
    int64 example_idx;
    int64 start;
    int64 end;
  };

  // Helper class to iterate through examples and return the corresponding
  // indices row range. Note that the row range can be empty in case a given
  // example has no corresponding indices.
  // An Iterator can be initialized from any example start offset, the
  // corresponding range indicators will be initialized in log time.
  class Iterator {
   public:
    Iterator(SparseColumnIterable* iter, int64 example_idx);

    Iterator& operator++() {
      ++example_idx_;
      if (cur_ < end_ && iter_->ix()(cur_, 0) < example_idx_) {
        cur_ = next_;
        UpdateNext();
      }
      return (*this);
    }

    Iterator operator++(int) {
      Iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    bool operator!=(const Iterator& other) const {
      QCHECK_EQ(iter_, other.iter_);
      return (example_idx_ != other.example_idx_);
    }

    bool operator==(const Iterator& other) const {
      QCHECK_EQ(iter_, other.iter_);
      return (example_idx_ == other.example_idx_);
    }

    const ExampleRowRange& operator*() {
      range_.example_idx = example_idx_;
      if (cur_ < end_ && iter_->ix()(cur_, 0) == example_idx_) {
        range_.start = cur_;
        range_.end = next_;
      } else {
        range_.start = 0;
        range_.end = 0;
      }
      return range_;
    }

   private:
    void UpdateNext() {
      next_ = std::min(next_ + 1, end_);
      while (next_ < end_ && iter_->ix()(cur_, 0) == iter_->ix()(next_, 0)) {
        ++next_;
      }
    }

    const SparseColumnIterable* iter_;
    int64 example_idx_;
    int64 cur_;
    int64 next_;
    const int64 end_;
    ExampleRowRange range_;
  };

  // Constructs an iterable given the desired examples slice and corresponding
  // feature columns.
  SparseColumnIterable(TTypes<int64>::ConstMatrix ix, int64 example_start,
                       int64 example_end)
      : ix_(ix), example_start_(example_start), example_end_(example_end) {
    QCHECK(example_start >= 0 && example_end >= 0);
  }

  Iterator begin() { return Iterator(this, example_start_); }
  Iterator end() { return Iterator(this, example_end_); }

  const TTypes<int64>::ConstMatrix& ix() const { return ix_; }
  int64 example_start() const { return example_start_; }
  int64 example_end() const { return example_end_; }

  const TTypes<int64>::ConstMatrix& sparse_indices() const { return ix_; }

 private:
  // Sparse indices matrix.
  TTypes<int64>::ConstMatrix ix_;

  // Example slice spec.
  const int64 example_start_;
  const int64 example_end_;
};

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_SPARSE_COLUMN_ITERABLE_H_
