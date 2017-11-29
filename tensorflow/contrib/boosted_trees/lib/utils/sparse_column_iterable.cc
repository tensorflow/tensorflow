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

#include "tensorflow/contrib/boosted_trees/lib/utils/sparse_column_iterable.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

using ExampleRowRange = SparseColumnIterable::ExampleRowRange;
using Iterator = SparseColumnIterable::Iterator;

namespace {

// Iterator over indices matrix rows.
class IndicesRowIterator
    : public std::iterator<std::random_access_iterator_tag, const int64> {
 public:
  IndicesRowIterator() : iter_(nullptr), row_idx_(-1) {}
  IndicesRowIterator(SparseColumnIterable* iter, int row_idx)
      : iter_(iter), row_idx_(row_idx) {}
  IndicesRowIterator(const IndicesRowIterator& other)
      : iter_(other.iter_), row_idx_(other.row_idx_) {}

  IndicesRowIterator& operator=(const IndicesRowIterator& other) {
    iter_ = other.iter_;
    row_idx_ = other.row_idx_;
    return (*this);
  }

  IndicesRowIterator& operator++() {
    ++row_idx_;
    return (*this);
  }

  IndicesRowIterator operator++(int) {
    IndicesRowIterator tmp(*this);
    ++row_idx_;
    return tmp;
  }

  reference operator*() { return iter_->ix()(row_idx_, 0); }

  pointer operator->() { return &iter_->ix()(row_idx_, 0); }

  IndicesRowIterator& operator--() {
    --row_idx_;
    return (*this);
  }

  IndicesRowIterator operator--(int) {
    IndicesRowIterator tmp(*this);
    --row_idx_;
    return tmp;
  }

  IndicesRowIterator& operator+=(const difference_type& step) {
    row_idx_ += step;
    return (*this);
  }
  IndicesRowIterator& operator-=(const difference_type& step) {
    row_idx_ -= step;
    return (*this);
  }

  IndicesRowIterator operator+(const difference_type& step) const {
    IndicesRowIterator tmp(*this);
    tmp += step;
    return tmp;
  }

  IndicesRowIterator operator-(const difference_type& step) const {
    IndicesRowIterator tmp(*this);
    tmp -= step;
    return tmp;
  }

  difference_type operator-(const IndicesRowIterator& other) const {
    return row_idx_ - other.row_idx_;
  }

  bool operator!=(const IndicesRowIterator& other) const {
    QCHECK_EQ(iter_, other.iter_);
    return (row_idx_ != other.row_idx_);
  }

  bool operator==(const IndicesRowIterator& other) const {
    QCHECK_EQ(iter_, other.iter_);
    return (row_idx_ == other.row_idx_);
  }

  Eigen::Index row_idx() const { return row_idx_; }

 private:
  SparseColumnIterable* iter_;
  Eigen::Index row_idx_;
};
}  // namespace

Iterator::Iterator(SparseColumnIterable* iter, int64 example_idx)
    : iter_(iter), example_idx_(example_idx), end_(iter->ix_.dimension(0)) {
  cur_ = next_ = std::lower_bound(IndicesRowIterator(iter, 0),
                                  IndicesRowIterator(iter, end_), example_idx_)
                     .row_idx();
  UpdateNext();
}

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
