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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLES_ITERABLE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLES_ITERABLE_H_

#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/utils/example.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/sparse_column_iterable.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

// Enables row-wise iteration through examples from feature columns.
class ExamplesIterable {
 public:
  // Constructs an iterable given the desired examples slice and corresponding
  // feature columns.
  ExamplesIterable(
      const std::vector<Tensor>& dense_float_feature_columns,
      const std::vector<sparse::SparseTensor>& sparse_float_feature_columns,
      const std::vector<sparse::SparseTensor>& sparse_int_feature_columns,
      int64 example_start, int64 example_end);

  // Helper class to iterate through examples.
  class Iterator {
   public:
    Iterator(ExamplesIterable* iter, int64 example_idx);

    Iterator& operator++() {
      // Advance to next example.
      ++example_idx_;

      // Update sparse column iterables.
      for (auto& it : sparse_float_column_iterators_) {
        ++it;
      }
      for (auto& it : sparse_int_column_iterators_) {
        ++it;
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

    const Example& operator*() {
      // Set example index based on iterator.
      example_.example_idx = example_idx_;

      // Get dense float values per column.
      auto& dense_float_features = example_.dense_float_features;
      for (size_t dense_float_idx = 0;
           dense_float_idx < dense_float_features.size(); ++dense_float_idx) {
        dense_float_features[dense_float_idx] =
            iter_->dense_float_column_values_[dense_float_idx](example_idx_, 0);
      }

      // Get sparse float values per column.
      auto& sparse_float_features = example_.sparse_float_features;
      sparse_float_features.clear();
      // Iterate through each sparse float feature column.
      for (size_t sparse_float_idx = 0;
           sparse_float_idx < iter_->sparse_float_column_iterables_.size();
           ++sparse_float_idx) {
        // Get range for values tensor.
        const auto& row_range =
            (*sparse_float_column_iterators_[sparse_float_idx]);
        DCHECK_EQ(example_idx_, row_range.example_idx);
        // If the example has this feature column.
        if (row_range.start < row_range.end) {
          // Retrieve original indices tensor.
          const TTypes<int64>::ConstMatrix& indices =
              iter_->sparse_float_column_iterables_[sparse_float_idx]
                  .sparse_indices();

          // For each value.
          for (int64 row_idx = row_range.start; row_idx < row_range.end;
               ++row_idx) {
            // Get the feature id for the feature column and the value.
            const int32 feature_id = indices(row_idx, 1);
            DCHECK_EQ(example_idx_, indices(row_idx, 0));

            // Save the value to our sparse matrix.
            sparse_float_features.addElement(
                sparse_float_idx, feature_id,
                iter_->sparse_float_column_values_[sparse_float_idx](row_idx));
          }
        }
      }

      // Get sparse int values per column.
      auto& sparse_int_features = example_.sparse_int_features;
      for (size_t sparse_int_idx = 0;
           sparse_int_idx < sparse_int_features.size(); ++sparse_int_idx) {
        const auto& row_range = (*sparse_int_column_iterators_[sparse_int_idx]);
        DCHECK_EQ(example_idx_, row_range.example_idx);
        sparse_int_features[sparse_int_idx].clear();
        if (row_range.start < row_range.end) {
          sparse_int_features[sparse_int_idx].reserve(row_range.end -
                                                      row_range.start);
          for (int64 row_idx = row_range.start; row_idx < row_range.end;
               ++row_idx) {
            sparse_int_features[sparse_int_idx].insert(
                iter_->sparse_int_column_values_[sparse_int_idx](row_idx));
          }
        }
      }

      return example_;
    }

   private:
    // Examples iterable (not owned).
    const ExamplesIterable* iter_;

    // Example index.
    int64 example_idx_;

    // Sparse float column iterators.
    std::vector<SparseColumnIterable::Iterator> sparse_float_column_iterators_;

    // Sparse int column iterators.
    std::vector<SparseColumnIterable::Iterator> sparse_int_column_iterators_;

    // Example placeholder.
    Example example_;
  };

  Iterator begin() { return Iterator(this, example_start_); }
  Iterator end() { return Iterator(this, example_end_); }

 private:
  // Example slice spec.
  const int64 example_start_;
  const int64 example_end_;

  // Dense float column values.
  std::vector<TTypes<float>::ConstMatrix> dense_float_column_values_;

  // Sparse float column iterables.
  std::vector<SparseColumnIterable> sparse_float_column_iterables_;

  // Sparse float column values.
  std::vector<TTypes<float>::ConstVec> sparse_float_column_values_;

  // Sparse int column iterables.
  std::vector<SparseColumnIterable> sparse_int_column_iterables_;

  // Sparse int column values.
  std::vector<TTypes<int64>::ConstVec> sparse_int_column_values_;
};

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLES_ITERABLE_H_
