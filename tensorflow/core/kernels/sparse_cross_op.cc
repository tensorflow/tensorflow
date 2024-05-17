/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Contains OP to generate sparse crosses.
#include <assert.h>

#include <limits>
#include <string>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/strong_hash.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {
// An interface that represents a column with batches.
template <typename InternalType>
class ColumnInterface {
 public:
  // Returns the number of features in the specified batch.
  virtual int64_t FeatureCount(int64_t batch) const = 0;

  // Returns the fingerprint of nth feature from the specified batch.
  virtual InternalType Feature(int64_t batch, int64_t n,
                               bool strong_hash) const = 0;

  virtual ~ColumnInterface() {}
};

// A column that is backed by a sparse tensor.
template <typename InternalType>
class SparseTensorColumn : public ColumnInterface<InternalType> {
 public:
  SparseTensorColumn(const Tensor& values, std::vector<int64_t> feature_counts,
                     std::vector<int64_t> feature_start_indices)
      : values_(values),
        feature_counts_(std::move(feature_counts)),
        feature_start_indices_(std::move(feature_start_indices)) {
    CHECK_EQ(feature_counts_.size(), feature_start_indices_.size());
  }

  int64_t FeatureCount(int64_t batch) const override {
    return feature_counts_[batch];
  }

  InternalType Feature(int64_t batch, int64_t n,
                       bool strong_hash) const override;

  ~SparseTensorColumn() override {}

 private:
  const Tensor& values_;
  std::vector<int64_t> feature_counts_;
  std::vector<int64_t> feature_start_indices_;
};

// A column that is backed by a sparse tensor.
template <typename InternalType>
class KeyedSparseTensorColumn : public ColumnInterface<InternalType> {
 public:
  KeyedSparseTensorColumn(const Tensor& values,
                          std::vector<int64_t> feature_counts,
                          std::vector<int64_t> feature_start_indices,
                          std::vector<int64_t> key)
      : values_(values),
        feature_counts_(std::move(feature_counts)),
        feature_start_indices_(std::move(feature_start_indices)) {
    DCHECK_EQ(feature_counts_.size(), feature_start_indices_.size());
    std::memcpy(key_, key.data(), sizeof(key_));
  }

  int64_t FeatureCount(int64_t batch) const override {
    return feature_counts_[batch];
  }

  InternalType Feature(int64_t batch, int64_t n,
                       bool strong_hash) const override;

  ~KeyedSparseTensorColumn() override {}

 private:
  const Tensor& values_;
  tensorflow::uint64 key_[2];
  std::vector<int64_t> feature_counts_;
  std::vector<int64_t> feature_start_indices_;
};

// InternalType is int64 only when using HashCrosser.
template <>
int64_t SparseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n,
                                             bool strong_hash) const {
  const int64_t start = feature_start_indices_[batch];
  if (DT_STRING == values_.dtype())
    return Fingerprint64(values_.vec<tstring>().data()[start + n]);
  return values_.vec<int64_t>().data()[start + n];
}

template <>
int64_t KeyedSparseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n,
                                                  bool strong_hash) const {
  const int64_t start = feature_start_indices_[batch];
  if (strong_hash) {
    if (DT_STRING == values_.dtype()) {
      return StrongKeyedHash(key_, values_.vec<tstring>()(start + n));
    }
    return StrongKeyedHash(
        key_,
        {reinterpret_cast<const char*>(&values_.vec<int64_t>()(start + n)),
         sizeof(values_.dtype())});
  }
  if (DT_STRING == values_.dtype())
    return Fingerprint64(values_.vec<tstring>()(start + n));
  return Fingerprint64(
      {reinterpret_cast<const char*>(&values_.vec<int64_t>()(start + n)),
       sizeof(values_.dtype())});
}

// InternalType is string or StringPiece when using StringCrosser.
template <>
tstring SparseTensorColumn<tstring>::Feature(int64_t batch, int64_t n,
                                             bool strong_hash) const {
  const int64_t start = feature_start_indices_[batch];
  if (DT_STRING == values_.dtype())
    return values_.vec<tstring>().data()[start + n];
  return std::to_string(values_.vec<int64_t>().data()[start + n]);
}

template <>
tstring KeyedSparseTensorColumn<tstring>::Feature(int64_t batch, int64_t n,
                                                  bool strong_hash) const {
  const int64_t start = feature_start_indices_[batch];
  if (DT_STRING == values_.dtype())
    return values_.vec<tstring>().data()[start + n];
  return std::to_string(values_.vec<int64_t>().data()[start + n]);
}

template <>
StringPiece SparseTensorColumn<StringPiece>::Feature(int64_t batch, int64_t n,
                                                     bool strong_hash) const {
  const int64_t start = feature_start_indices_[batch];
  return values_.vec<tstring>().data()[start + n];
}

template <>
StringPiece KeyedSparseTensorColumn<StringPiece>::Feature(
    int64_t batch, int64_t n, bool strong_hash) const {
  const int64_t start = feature_start_indices_[batch];
  return values_.vec<tstring>().data()[start + n];
}

// A column that is backed by a dense tensor.
template <typename InternalType>
class DenseTensorColumn : public ColumnInterface<InternalType> {
 public:
  explicit DenseTensorColumn(const Tensor& tensor) : tensor_(tensor) {}

  int64_t FeatureCount(int64_t batch) const override {
    return tensor_.dim_size(1);
  }

  InternalType Feature(int64_t batch, int64_t n,
                       bool strong_hash) const override;

  ~DenseTensorColumn() override {}

 private:
  const Tensor& tensor_;
};

// A column that is backed by a dense tensor.
template <typename InternalType>
class KeyedDenseTensorColumn : public ColumnInterface<InternalType> {
 public:
  explicit KeyedDenseTensorColumn(const Tensor& tensor,
                                  std::vector<int64_t> key)
      : tensor_(tensor) {
    std::memcpy(key_, key.data(), sizeof(key_));
  }

  int64_t FeatureCount(int64_t batch) const override {
    return tensor_.dim_size(1);
  }

  InternalType Feature(int64_t batch, int64_t n,
                       bool strong_hash) const override;

  ~KeyedDenseTensorColumn() override {}

 private:
  const Tensor& tensor_;
  tensorflow::uint64 key_[2];
};

// InternalType is int64 only when using HashCrosser.
template <>
int64_t DenseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n,
                                            bool strong_hash) const {
  if (DT_STRING == tensor_.dtype())
    return Fingerprint64(tensor_.matrix<tstring>()(batch, n));
  return tensor_.matrix<int64_t>()(batch, n);
}

template <>
int64_t KeyedDenseTensorColumn<int64_t>::Feature(int64_t batch, int64_t n,
                                                 bool strong_hash) const {
  if (strong_hash) {
    if (DT_STRING == tensor_.dtype()) {
      return StrongKeyedHash(key_, tensor_.matrix<tstring>()(batch, n));
    }
    return StrongKeyedHash(
        key_,
        {reinterpret_cast<const char*>(tensor_.matrix<int64_t>()(batch, n)),
         sizeof(tensor_.dtype())});
  }
  if (DT_STRING == tensor_.dtype())
    return Fingerprint64(tensor_.matrix<tstring>()(batch, n));
  return tensor_.matrix<int64_t>()(batch, n);
}

// Internal type is string or StringPiece when using StringCrosser.
template <>
tstring DenseTensorColumn<tstring>::Feature(int64_t batch, int64_t n,
                                            bool strong_hash) const {
  if (DT_STRING == tensor_.dtype()) return tensor_.matrix<tstring>()(batch, n);
  return std::to_string(tensor_.matrix<int64_t>()(batch, n));
}

template <>
tstring KeyedDenseTensorColumn<tstring>::Feature(int64_t batch, int64_t n,
                                                 bool strong_hash) const {
  if (DT_STRING == tensor_.dtype()) return tensor_.matrix<tstring>()(batch, n);
  return std::to_string(tensor_.matrix<int64_t>()(batch, n));
}

template <>
StringPiece DenseTensorColumn<StringPiece>::Feature(int64_t batch, int64_t n,
                                                    bool strong_hash) const {
  return tensor_.matrix<tstring>()(batch, n);
}

template <>
StringPiece KeyedDenseTensorColumn<StringPiece>::Feature(
    int64_t batch, int64_t n, bool strong_hash) const {
  return tensor_.matrix<tstring>()(batch, n);
}

// Updates Output tensors with sparse crosses.
template <typename OutType>
class OutputUpdater {
 public:
  OutputUpdater(const std::vector<int64_t>& output_start_indices,
                Tensor* indices_out, Tensor* values_out)
      : output_start_indices_(output_start_indices),
        indices_out_(indices_out),
        values_out_(values_out) {}

  void Update(const int64_t batch_index, const int64_t cross_count,
              const OutType& cross) const {
    const int64_t output_index =
        output_start_indices_[batch_index] + cross_count;

    auto indices_matrix = indices_out_->matrix<int64_t>();
    indices_matrix(output_index, 0) = batch_index;
    indices_matrix(output_index, 1) = cross_count;

    auto value_vec = values_out_->vec<OutType>();
    value_vec(output_index) = cross;
  }

 private:
  const std::vector<int64_t>& output_start_indices_;
  Tensor* indices_out_;
  Tensor* values_out_;
};

// Generates the sparse crosses as concatenation of strings.
template <typename InternalType>
class StringCrosser {
 public:
  StringCrosser(const std::vector<
                    std::unique_ptr<ColumnInterface<InternalType>>>& columns,
                const int64_t num_buckets_unused, const uint64 hash_key_unused,
                const tstring k_feature_separator)
      : columns_(columns), k_feature_separator_(k_feature_separator) {}

  string Generate(const int64_t batch_index,
                  const std::vector<int>& permutation,
                  bool unused_strong_hash) const {
    gtl::InlinedVector<InternalType, 6> cross_vec(columns_.size());
    for (int i = 0; i < permutation.size(); i++) {
      cross_vec[i] = columns_[i]->Feature(batch_index, permutation[i], false);
    }
    // TODO(zakaria): this will copy the string twice, might effect
    // performance.
    return absl::StrJoin(cross_vec, k_feature_separator_);
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>& columns_;
  const tstring k_feature_separator_;
};

// Generates the sparse crosses as nested hash to avoid string manipulations.
class HashCrosser {
 public:
  HashCrosser(
      const std::vector<std::unique_ptr<ColumnInterface<int64_t>>>& columns,
      const int64_t num_buckets, const uint64 hash_key,
      const tstring k_feature_separator_unused)
      : columns_(columns), num_buckets_(num_buckets), hash_key_(hash_key) {}

  int64_t Generate(const int64_t batch_index,
                   const std::vector<int>& permutation,
                   bool unused_strong_hash) const {
    // Do the fingerprint concatenation on uint64.
    uint64 hashed_output = hash_key_;
    for (size_t i = 0; i < permutation.size(); ++i) {
      uint64 hash_i = columns_[i]->Feature(batch_index, permutation[i], false);
      hashed_output = FingerprintCat64(hashed_output, hash_i);
    }
    // The return value is int64 based on the number of buckets.
    if (num_buckets_ > 0) {
      return hashed_output % num_buckets_;
    } else {
      // To prevent negative output we take modulo to max int64.
      return hashed_output % std::numeric_limits<int64_t>::max();
    }
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<int64_t>>>& columns_;
  const int64_t num_buckets_;
  const uint64 hash_key_;
};

// Generates the sparse crosses as nested hash to avoid string manipulations.
class HashCrosserV2 {
 public:
  HashCrosserV2(
      const std::vector<std::unique_ptr<ColumnInterface<int64_t>>>& columns,
      const int64_t num_buckets, const uint64 hash_key_unused,
      const tstring k_feature_separator_unused)
      : columns_(columns), num_buckets_(num_buckets) {}

  int64_t Generate(const int64_t batch_index,
                   const std::vector<int>& permutation,
                   bool strong_hash) const {
    // Do the fingerprint concatenation on uint64.
    uint64 hashed_output =
        columns_[0]->Feature(batch_index, permutation[0], strong_hash);
    for (size_t i = 1; i < permutation.size(); ++i) {
      uint64 hash_i =
          columns_[i]->Feature(batch_index, permutation[i], strong_hash);
      hashed_output = FingerprintCat64(hashed_output, hash_i);
    }
    // The return value is int64 based on the number of buckets.
    if (num_buckets_ > 0) {
      return hashed_output % num_buckets_;
    } else {
      // To prevent negative output we take modulo to max int64.
      return hashed_output % std::numeric_limits<int64_t>::max();
    }
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<int64_t>>>& columns_;
  const int64_t num_buckets_;
};

// ProductIterator generates cartesian products based on indices.
template <typename InternalType>
class ProductIterator {
 public:
  explicit ProductIterator(
      const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>&
          columns,
      int64_t batch_index)
      : columns_(columns), batch_index_(batch_index) {
    next_permutation_.resize(columns_.size(), 0);
    // Sets has_next_ to false if any feature column has 0 features.
    has_next_ = true;
    for (int i = 0; i < columns_.size(); i++) {
      if (columns_[i]->FeatureCount(batch_index_) == 0) {
        has_next_ = false;
        break;
      }
    }
  }

  std::vector<int> Next() {
    std::vector<int> permutation(next_permutation_);

    // Generates next permutation, if available.
    bool carry = true;
    for (int i = next_permutation_.size() - 1; i >= 0; i--) {
      if (carry) {
        next_permutation_[i] = next_permutation_[i] + 1;
      }
      if (next_permutation_[i] == columns_[i]->FeatureCount(batch_index_)) {
        next_permutation_[i] = 0;
      } else {
        carry = false;
        break;
      }
    }
    has_next_ = !carry;
    return permutation;
  }

  bool HasNext() { return has_next_; }

 private:
  bool has_next_;
  const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>& columns_;
  const int64_t batch_index_;
  std::vector<int> next_permutation_;
};

template <bool HASHED_OUTPUT, typename InternalType>
struct CrossTraits;

template <typename InternalType>
struct CrossTraits<false, InternalType> {
  typedef StringCrosser<InternalType> Crosser;
  typedef StringCrosser<InternalType> CrosserV2;
  typedef OutputUpdater<tstring> Updater;
};

template <>
struct CrossTraits<true, int64_t> {
  typedef HashCrosser Crosser;
  typedef HashCrosserV2 CrosserV2;
  typedef OutputUpdater<int64_t> Updater;
};
}  // namespace

// Calculate the batch size from either the shapes input or the dense input.
int64_t CalculateBatchSize(const OpInputList& shapes_list_in,
                           const OpInputList& dense_list_in) {
  if (shapes_list_in.size() > 0) {
    return shapes_list_in[0].vec<int64_t>()(0);
  }

  if (dense_list_in.size() > 0) {
    return dense_list_in[0].dim_size(0);
  }

  return 0;
}

// Validates input tensors.
Status ValidateInput(const OpInputList& indices_list_in,
                     const OpInputList& values_list_in,
                     const OpInputList& shapes_list_in,
                     const OpInputList& dense_list_in,
                     const DataType& internal_type) {
  const auto size = indices_list_in.size();
  // Only perform internal_type check for SparseCrossOp.
  // Check if the internal_type is not invalid before doing so.
  bool check_type = internal_type != DT_INVALID;
  // Validates indices_list_in OpInputList.
  for (int i = 0; i < size; i++) {
    if (check_type && indices_list_in[i].dtype() != DT_INT64) {
      return errors::InvalidArgument("Input indices should be of type ",
                                     DT_INT64, " but received ",
                                     indices_list_in[i].dtype());
    }
    if (!TensorShapeUtils::IsMatrix(indices_list_in[i].shape())) {
      return errors::InvalidArgument(
          "Input indices should be a matrix but received shape ",
          indices_list_in[i].shape().DebugString(), " at position ", i);
    }
    if (indices_list_in[i].shape().dim_size(1) != 2) {
      return errors::InvalidArgument("Expected D2 of index to be 2 got ",
                                     indices_list_in[i].shape().dim_size(1),
                                     " at position ", i);
    }
  }

  // Validates values_list_in OpInputList.
  if (values_list_in.size() != size) {
    return errors::InvalidArgument("Expected ", size, " input values, got ",
                                   values_list_in.size());
  }
  for (int i = 0; i < size; i++) {
    // Make sure to avoid the expected type to be string, but input values to be
    // int64.
    if (check_type && internal_type == DT_STRING &&
        values_list_in[i].dtype() == DT_INT64) {
      return errors::InvalidArgument("Input values should be of internal type ",
                                     internal_type, " but received ",
                                     values_list_in[i].dtype());
    }
    if (!TensorShapeUtils::IsVector(values_list_in[i].shape())) {
      return errors::InvalidArgument(
          "Input values should be a vector but received shape ",
          values_list_in[i].shape().DebugString(), " at position ", i);
    }
    if (indices_list_in[i].shape().dim_size(0) !=
        values_list_in[i].shape().dim_size(0)) {
      return errors::InvalidArgument(
          "Expected size of values to be ",
          indices_list_in[i].shape().dim_size(0), " got ",
          values_list_in[i].shape().dim_size(0), " at position ", i);
    }
  }

  // Validates shapes_list_in OpInputList
  if (shapes_list_in.size() != size) {
    return errors::InvalidArgument("Expected ", size, " input shapes, got ",
                                   shapes_list_in.size());
  }
  for (int i = 0; i < size; i++) {
    if (check_type && shapes_list_in[i].dtype() != DT_INT64) {
      return errors::InvalidArgument("Input shape should be of type ", DT_INT64,
                                     " but received ",
                                     shapes_list_in[i].dtype());
    }
    if (!TensorShapeUtils::IsVector(shapes_list_in[i].shape())) {
      return errors::InvalidArgument(
          "Input shapes should be a vector but received shape ",
          shapes_list_in[i].shape().DebugString(), " at position ", i);
    }

    if (shapes_list_in[i].vec<int64_t>().size() != 2) {
      return errors::InvalidArgument("shape should imply a 2D tensor, but got ",
                                     shapes_list_in[i].shape().DebugString(),
                                     " at position ", i);
    }
  }

  // Validates dense_list_in OpInputList
  for (int i = 0; i < dense_list_in.size(); ++i) {
    // Make sure to avoid the expected type to be string, but input values to be
    // int64.
    if (check_type && internal_type == DT_STRING &&
        dense_list_in[i].dtype() == DT_INT64) {
      return errors::InvalidArgument("Dense inputs should be of internal type ",
                                     internal_type, " but received ",
                                     dense_list_in[i].dtype());
    }
    if (!TensorShapeUtils::IsMatrix(dense_list_in[i].shape())) {
      return errors::InvalidArgument(
          "Dense inputs should be a matrix but received shape ",
          dense_list_in[i].shape().DebugString(), " at position ", i);
    }
  }

  // Validates batch sizes.  (Note: we do this after validating the input
  // shapes, because CalculateBatchSize() depends on inputs having valid
  // shapes).
  const auto batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
  for (int i = 0; i < size; i++) {
    if (shapes_list_in[i].vec<int64_t>()(0) != batch_size) {
      return errors::InvalidArgument(
          "Expected batch size ", batch_size, " got ",
          shapes_list_in[i].vec<int64_t>()(0), " at position ", i);
    }
  }
  for (int i = 0; i < dense_list_in.size(); ++i) {
    if (dense_list_in[i].dim_size(0) != batch_size) {
      return errors::InvalidArgument("Expected batch size ", batch_size,
                                     " got ", dense_list_in[i].dim_size(0),
                                     " at dense tensor ", i);
    }
  }

  return absl::OkStatus();
}

// Extracts data about the features and populates feature data.
void ExtractFeatureData(
    const OpInputList& indices_list_in, int64_t batch_size,
    std::vector<std::vector<int64_t>>* feature_counts,
    std::vector<std::vector<int64_t>>* feature_start_indices) {
  gtl::InlinedVector<int64_t, 8> current_row(indices_list_in.size(), 0);
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < indices_list_in.size(); i++) {
      const auto indices = indices_list_in[i].matrix<int64_t>();
      int64_t feature_count = 0;
      int64_t start_index = current_row[i];
      // Loops until we reach next batch index for current feature column.
      while (current_row[i] < indices_list_in[i].dim_size(0) &&
             indices(current_row[i], 0) == b) {
        feature_count++;
        current_row[i]++;
      }
      (*feature_counts)[i].push_back(feature_count);
      (*feature_start_indices)[i].push_back(start_index);
    }
  }
}

// Returns number of crosses for a given batch_index
template <typename InternalType>
int64_t CrossCountByBatchIndex(
    const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>& columns,
    int batch_index) {
  int64_t cross_count = 1;
  for (int i = 0; i < columns.size(); i++) {
    const auto feature_count = columns[i]->FeatureCount(batch_index);
    // If one column is missing any feature, there won't be any cross.
    if (feature_count == 0) {
      return 0;
    }
    cross_count *= feature_count;
  }
  return cross_count;
}

// Generate the columns given the sparse and dense inputs.
template <typename InternalType>
std::vector<std::unique_ptr<ColumnInterface<InternalType>>>
GenerateColumnsFromInput(const OpInputList& indices_list_in,
                         const OpInputList& values_list_in,
                         const OpInputList& shapes_list_in,
                         const OpInputList& dense_list_in) {
  std::vector<std::unique_ptr<ColumnInterface<InternalType>>> columns;
  const int64_t batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
  const int64_t number_of_columns = shapes_list_in.size();

  std::vector<std::vector<int64_t>> feature_counts(number_of_columns,
                                                   std::vector<int64_t>());
  std::vector<std::vector<int64_t>> feature_start_indices(
      number_of_columns, std::vector<int64_t>());

  ExtractFeatureData(indices_list_in, batch_size, &feature_counts,
                     &feature_start_indices);

  columns.reserve(values_list_in.size());
  for (int i = 0; i < values_list_in.size(); ++i) {
    columns.emplace_back(new SparseTensorColumn<InternalType>(
        values_list_in[i], std::move(feature_counts[i]),
        std::move(feature_start_indices[i])));
  }
  for (int i = 0; i < dense_list_in.size(); ++i) {
    columns.emplace_back(new DenseTensorColumn<InternalType>(dense_list_in[i]));
  }

  return columns;
}

// Generate the columns given the sparse and dense inputs.
template <typename InternalType>
std::vector<std::unique_ptr<ColumnInterface<InternalType>>>
GenerateKeyedColumnsFromInput(const OpInputList& indices_list_in,
                              const OpInputList& values_list_in,
                              const OpInputList& shapes_list_in,
                              const OpInputList& dense_list_in,
                              std::vector<int64_t> keys) {
  std::vector<std::unique_ptr<ColumnInterface<InternalType>>> columns;
  const int64_t batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
  const int64_t number_of_columns = shapes_list_in.size();

  std::vector<std::vector<int64_t>> feature_counts(number_of_columns,
                                                   std::vector<int64_t>());
  std::vector<std::vector<int64_t>> feature_start_indices(
      number_of_columns, std::vector<int64_t>());

  ExtractFeatureData(indices_list_in, batch_size, &feature_counts,
                     &feature_start_indices);

  columns.reserve(values_list_in.size());
  for (int i = 0; i < values_list_in.size(); ++i) {
    columns.emplace_back(new KeyedSparseTensorColumn<InternalType>(
        values_list_in[i], std::move(feature_counts[i]),
        std::move(feature_start_indices[i]), keys));
  }
  for (int i = 0; i < dense_list_in.size(); ++i) {
    columns.emplace_back(
        new KeyedDenseTensorColumn<InternalType>(dense_list_in[i], keys));
  }

  return columns;
}

// Allocates output tensors with proper size and sets the shape tensor of
// the output SparseTensor.
// It also output_start_indices which contains the start indices for each
// input in the output SparseTensor.
template <typename InternalType>
Status CreateOutputTensors(
    const std::vector<std::unique_ptr<ColumnInterface<InternalType>>>& columns,
    int64_t batch_size, OpKernelContext* context, Tensor** indices_out,
    Tensor** values_out, Tensor** shape_out,
    std::vector<int64_t>* output_start_indices) {
  // Calculates dimensions for output tensors.
  int64_t cross_count_total = 0;
  int64_t max_cross_count = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    // For each input, sets starting indices in output SparseTensor
    (*output_start_indices)[b] = cross_count_total;
    const auto cross_count = CrossCountByBatchIndex(columns, b);
    max_cross_count = std::max(max_cross_count, cross_count);
    cross_count_total += cross_count;
  }

  // Allocates tensors.
  TF_RETURN_IF_ERROR(context->allocate_output(
      0, TensorShape({cross_count_total, 2}), indices_out));
  TF_RETURN_IF_ERROR(context->allocate_output(
      1, TensorShape({cross_count_total}), values_out));
  TF_RETURN_IF_ERROR(context->allocate_output(2, TensorShape({2}), shape_out));

  // Sets shape.
  auto shape_vec = (*shape_out)->vec<int64_t>();
  shape_vec(0) = batch_size;
  shape_vec(1) = max_cross_count;

  return absl::OkStatus();
}

template <bool HASHED_OUTPUT, typename InternalType>
class SparseCrossOp : public OpKernel {
 public:
  explicit SparseCrossOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
    // Read signed_hash_key_ as int64 since uint64 attributes are not
    // supported by REGISTER_OP.
    int64_t signed_hash_key_;
    OP_REQUIRES_OK(context, context->GetAttr("hash_key", &signed_hash_key_));
    hash_key_ = static_cast<uint64>(signed_hash_key_);
    OP_REQUIRES_OK(context, context->GetAttr("internal_type", &internal_type_));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList indices_list_in;
    OP_REQUIRES_OK(context, context->input_list("indices", &indices_list_in));
    OpInputList values_list_in;
    OP_REQUIRES_OK(context, context->input_list("values", &values_list_in));
    OpInputList shapes_list_in;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes_list_in));
    OpInputList dense_list_in;
    OP_REQUIRES_OK(context,
                   context->input_list("dense_inputs", &dense_list_in));

    DataType internal_type = internal_type_;
    OP_REQUIRES_OK(
        context, ValidateInput(indices_list_in, values_list_in, shapes_list_in,
                               dense_list_in, internal_type));

    std::vector<std::unique_ptr<ColumnInterface<InternalType>>> columns =
        GenerateColumnsFromInput<InternalType>(indices_list_in, values_list_in,
                                               shapes_list_in, dense_list_in);

    const tstring k_feature_separator = "_X_";
    typename CrossTraits<HASHED_OUTPUT, InternalType>::Crosser crosser(
        columns, num_buckets_, hash_key_, k_feature_separator);
    Tensor* indices_out;
    Tensor* values_out;
    Tensor* shape_out;
    const int64_t batch_size =
        CalculateBatchSize(shapes_list_in, dense_list_in);
    std::vector<int64_t> output_start_indices(batch_size);
    OP_REQUIRES_OK(
        context,
        CreateOutputTensors(columns, batch_size, context, &indices_out,
                            &values_out, &shape_out, &output_start_indices));

    typename CrossTraits<HASHED_OUTPUT, InternalType>::Updater updater(
        output_start_indices, indices_out, values_out);
    auto do_work = [&columns, crosser, updater](int64_t begin, int64_t end) {
      for (int b = begin; b < end; b++) {
        ProductIterator<InternalType> product_iterator(columns, b);
        int64_t cross_count = 0;
        while (product_iterator.HasNext()) {
          const auto permutation = product_iterator.Next();
          updater.Update(b, cross_count,
                         crosser.Generate(b, permutation, false));
          cross_count++;
        }
      }
    };

    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    // TODO(zakaria): optimize kCostPerUnit
    const int kCostPerUnit = 5000 * indices_list_in.size();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
          kCostPerUnit, do_work);
  }

 private:
  int64_t num_buckets_;
  uint64 hash_key_;
  DataType internal_type_;
};

class SparseCrossV2Op : public OpKernel {
 public:
  explicit SparseCrossV2Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OpInputList indices_list_in;
    OP_REQUIRES_OK(context, context->input_list("indices", &indices_list_in));
    OpInputList values_list_in;
    OP_REQUIRES_OK(context, context->input_list("values", &values_list_in));
    OpInputList shapes_list_in;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes_list_in));
    OpInputList dense_list_in;
    OP_REQUIRES_OK(context,
                   context->input_list("dense_inputs", &dense_list_in));

    // Set internal_type to invalid_type so that the check will be ignored.
    DataType internal_type = DT_INVALID;
    OP_REQUIRES_OK(
        context, ValidateInput(indices_list_in, values_list_in, shapes_list_in,
                               dense_list_in, internal_type));

    const Tensor* sep_t;
    OP_REQUIRES_OK(context, context->input("sep", &sep_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sep_t->shape()),
                errors::InvalidArgument("Input separator should be a scalar. "
                                        "Received: ",
                                        sep_t->DebugString()));
    const tstring separator = sep_t->scalar<tstring>()();

    std::vector<std::unique_ptr<ColumnInterface<tstring>>> columns =
        GenerateColumnsFromInput<tstring>(indices_list_in, values_list_in,
                                          shapes_list_in, dense_list_in);
    Tensor* indices_out;
    Tensor* values_out;
    Tensor* shape_out;
    const int64_t batch_size =
        CalculateBatchSize(shapes_list_in, dense_list_in);
    std::vector<int64_t> output_start_indices(batch_size);
    OP_REQUIRES_OK(
        context,
        CreateOutputTensors(columns, batch_size, context, &indices_out,
                            &values_out, &shape_out, &output_start_indices));
    StringCrosser<tstring> crosser(columns, 0, 0, separator);
    OutputUpdater<tstring> updater(output_start_indices, indices_out,
                                   values_out);
    auto do_work = [&columns, crosser, updater](int64_t begin, int64_t end) {
      for (int b = begin; b < end; b++) {
        ProductIterator<tstring> product_iterator(columns, b);
        int64_t cross_count = 0;
        while (product_iterator.HasNext()) {
          const auto permutation = product_iterator.Next();
          updater.Update(b, cross_count,
                         crosser.Generate(b, permutation, false));
          cross_count++;
        }
      }
    };

    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    // TODO(zakaria): optimize kCostPerUnit
    const int kCostPerUnit = 5000 * indices_list_in.size();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
          kCostPerUnit, do_work);
  }
};

class SparseCrossHashedOp : public OpKernel {
 public:
  explicit SparseCrossHashedOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OpInputList indices_list_in;
    OP_REQUIRES_OK(context, context->input_list("indices", &indices_list_in));
    OpInputList values_list_in;
    OP_REQUIRES_OK(context, context->input_list("values", &values_list_in));
    OpInputList shapes_list_in;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes_list_in));
    OpInputList dense_list_in;
    OP_REQUIRES_OK(context,
                   context->input_list("dense_inputs", &dense_list_in));

    // Set internal_type to invalid_type so that the check will be ignored.
    DataType internal_type = DT_INVALID;
    OP_REQUIRES_OK(
        context, ValidateInput(indices_list_in, values_list_in, shapes_list_in,
                               dense_list_in, internal_type));
    const Tensor* num_buckets_t;
    OP_REQUIRES_OK(context, context->input("num_buckets", &num_buckets_t));
    const int64_t num_buckets = num_buckets_t->scalar<int64_t>()();

    const Tensor* strong_hash_t;
    OP_REQUIRES_OK(context, context->input("strong_hash", &strong_hash_t));
    const bool strong_hash = strong_hash_t->scalar<bool>()();

    const Tensor* salt_t;
    OP_REQUIRES_OK(context, context->input("salt", &salt_t));
    const auto salt = salt_t->flat<int64_t>();
    OP_REQUIRES_OK(
        context, salt.size() == 2
                     ? Status()
                     : errors::InvalidArgument(
                           "Input \"salt\" must have length 2 but has length ",
                           salt.size()));
    std::vector<int64_t> key_{salt(0), salt(1)};

    std::vector<std::unique_ptr<ColumnInterface<int64_t>>> columns =
        GenerateKeyedColumnsFromInput<int64_t>(indices_list_in, values_list_in,
                                               shapes_list_in, dense_list_in,
                                               key_);
    Tensor* indices_out;
    Tensor* values_out;
    Tensor* shape_out;
    const int64_t batch_size =
        CalculateBatchSize(shapes_list_in, dense_list_in);
    std::vector<int64_t> output_start_indices(batch_size);
    OP_REQUIRES_OK(
        context,
        CreateOutputTensors(columns, batch_size, context, &indices_out,
                            &values_out, &shape_out, &output_start_indices));
    const tstring unused_sep;
    HashCrosserV2 crosser(columns, num_buckets, 0, unused_sep);
    OutputUpdater<int64_t> updater(output_start_indices, indices_out,
                                   values_out);
    auto do_work = [&columns, crosser, updater, strong_hash](int64_t begin,
                                                             int64_t end) {
      for (int b = begin; b < end; b++) {
        ProductIterator<int64_t> product_iterator(columns, b);
        int64_t cross_count = 0;
        while (product_iterator.HasNext()) {
          const auto permutation = product_iterator.Next();
          updater.Update(b, cross_count,
                         crosser.Generate(b, permutation, strong_hash));
          cross_count++;
        }
      }
    };

    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    // TODO(zakaria): optimize kCostPerUnit
    const int kCostPerUnit = 5000 * indices_list_in.size();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
          kCostPerUnit, do_work);
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tstring>("out_type")
                            .TypeConstraint<tstring>("internal_type"),
                        SparseCrossOp<false, StringPiece>);

REGISTER_KERNEL_BUILDER(Name("SparseCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tstring>("out_type")
                            .TypeConstraint<int64_t>("internal_type"),
                        SparseCrossOp<false, tstring>);

REGISTER_KERNEL_BUILDER(Name("SparseCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64_t>("out_type")
                            .TypeConstraint<tstring>("internal_type"),
                        SparseCrossOp<true, int64>);

REGISTER_KERNEL_BUILDER(Name("SparseCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64_t>("out_type")
                            .TypeConstraint<int64_t>("internal_type"),
                        SparseCrossOp<true, int64>);

REGISTER_KERNEL_BUILDER(Name("SparseCrossV2").Device(DEVICE_CPU),
                        SparseCrossV2Op);

REGISTER_KERNEL_BUILDER(Name("SparseCrossHashed").Device(DEVICE_CPU),
                        SparseCrossHashedOp);

}  // namespace tensorflow
