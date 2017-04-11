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
#include "tensorflow/contrib/boosted_trees/lib/utils/examples_iterable.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

using Iterator = ExamplesIterable::Iterator;

ExamplesIterable::ExamplesIterable(
    const std::vector<Tensor>& dense_float_feature_columns,
    const std::vector<sparse::SparseTensor>& sparse_float_feature_columns,
    const std::vector<sparse::SparseTensor>& sparse_int_feature_columns,
    int64 example_start, int64 example_end)
    : example_start_(example_start), example_end_(example_end) {
  // Create dense float column values.
  dense_float_column_values_.reserve(dense_float_feature_columns.size());
  for (auto& dense_float_column : dense_float_feature_columns) {
    dense_float_column_values_.emplace_back(
        dense_float_column.template matrix<float>());
  }

  // Create sparse float column iterables and values.
  sparse_float_column_iterables_.reserve(sparse_float_feature_columns.size());
  sparse_float_column_values_.reserve(sparse_float_feature_columns.size());
  for (auto& sparse_float_column : sparse_float_feature_columns) {
    sparse_float_column_iterables_.emplace_back(
        sparse_float_column.indices().template matrix<int64>(), example_start,
        example_end);
    sparse_float_column_values_.emplace_back(
        sparse_float_column.values().template vec<float>());
  }

  // Create sparse int column iterables and values.
  sparse_int_column_iterables_.reserve(sparse_int_feature_columns.size());
  sparse_int_column_values_.reserve(sparse_int_feature_columns.size());
  for (auto& sparse_int_column : sparse_int_feature_columns) {
    sparse_int_column_iterables_.emplace_back(
        sparse_int_column.indices().template matrix<int64>(), example_start,
        example_end);
    sparse_int_column_values_.emplace_back(
        sparse_int_column.values().template vec<int64>());
  }
}

Iterator::Iterator(ExamplesIterable* iter, int64 example_idx)
    : iter_(iter), example_idx_(example_idx) {
  // Create sparse iterators.
  sparse_float_column_iterators_.reserve(
      iter->sparse_float_column_iterables_.size());
  for (auto& iterable : iter->sparse_float_column_iterables_) {
    sparse_float_column_iterators_.emplace_back(iterable.begin());
  }
  sparse_int_column_iterators_.reserve(
      iter->sparse_int_column_iterables_.size());
  for (auto& iterable : iter->sparse_int_column_iterables_) {
    sparse_int_column_iterators_.emplace_back(iterable.begin());
  }

  // Pre-size example features.
  example_.dense_float_features.resize(
      iter_->dense_float_column_values_.size());
  example_.sparse_float_features.resize(
      iter_->sparse_float_column_values_.size());
  example_.sparse_int_features.resize(iter_->sparse_int_column_values_.size());
}

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
