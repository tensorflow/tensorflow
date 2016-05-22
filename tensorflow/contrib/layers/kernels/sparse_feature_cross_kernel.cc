/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {
// Seed is chosen based on third_party/tensorflow/core/lib/hash/hash.h
const int64 kInitialHashSeed = 0xDECAFCAFFE;

// Following functions are a copy of Hash64. It will be replaced by a
// fingerprint function.
// Original code: third_party/tensorflow/core/lib/hash/hash.h
static inline uint64 ByteAs64(char c) { return static_cast<uint64>(c) & 0xff; }

inline uint32 DecodeFixed32(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint32 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint32>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[1])) << 8) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[2])) << 16) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}

inline uint64 DecodeFixed64(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint64 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    uint64 lo = DecodeFixed32(ptr);
    uint64 hi = DecodeFixed32(ptr + 4);
    return (hi << 32) | lo;
  }
}

uint64 LegacyHashFunction(const char* data, size_t n, uint64 seed) {
  const uint64 m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64 h = seed ^ (n * m);

  while (n >= 8) {
    uint64 k = DecodeFixed64(data);
    data += 8;
    n -= 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
      TF_FALLTHROUGH_INTENDED;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
      TF_FALLTHROUGH_INTENDED;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
      TF_FALLTHROUGH_INTENDED;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
      TF_FALLTHROUGH_INTENDED;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
      TF_FALLTHROUGH_INTENDED;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
      TF_FALLTHROUGH_INTENDED;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

// An interface that represents a column with batches.
template <typename StringType>
class ColumnInterface {
 public:
  // Returns the number of features in the specified batch.
  virtual int64 FeatureCount(int64 batch) const = 0;

  // Returns the nth feature from the specified batch.
  virtual StringType Feature(int64 batch, int64 n) const = 0;

  virtual ~ColumnInterface() {}
};

// A column that is backed by a sparse tensor.
template <typename StringType>
class SparseTensorColumn : public ColumnInterface<StringType> {
 public:
  SparseTensorColumn(const Tensor& values, std::vector<int64> feature_counts,
                     std::vector<int64> feature_start_indices)
      : values_(values),
        feature_counts_(std::move(feature_counts)),
        feature_start_indices_(std::move(feature_start_indices)) {
    CHECK_EQ(feature_counts_.size(), feature_start_indices_.size());
  }

  int64 FeatureCount(int64 batch) const override {
    return feature_counts_[batch];
  }

  StringType Feature(int64 batch, int64 n) const override {
    const int64 start = feature_start_indices_[batch];
    if (DT_STRING == values_.dtype())
      return values_.vec<string>().data()[start + n];
    return StringType(std::to_string(values_.vec<int64>().data()[start + n]));
  }

  ~SparseTensorColumn() override {}

 private:
  const Tensor& values_;
  std::vector<int64> feature_counts_;
  std::vector<int64> feature_start_indices_;
};

// A column that is backed by a dense tensor.
template <typename StringType>
class DenseTensorColumn : public ColumnInterface<StringType> {
 public:
  explicit DenseTensorColumn(const Tensor& tensor) : tensor_(tensor) {}

  int64 FeatureCount(int64 batch) const override { return tensor_.dim_size(1); }

  StringType Feature(int64 batch, int64 n) const override {
    if (DT_STRING == tensor_.dtype()) return tensor_.matrix<string>()(batch, n);
    return StringType(std::to_string(tensor_.matrix<int64>()(batch, n)));
  }

  ~DenseTensorColumn() override {}

 private:
  const Tensor& tensor_;
};

// Updates Output tensors with sparse crosses.
template <typename OutType>
class OutputUpdater {
 public:
  OutputUpdater(const std::vector<int64>& output_start_indices,
                Tensor* indices_out, Tensor* values_out)
      : output_start_indices_(output_start_indices),
        indices_out_(indices_out),
        values_out_(values_out) {}

  void Update(const int64 batch_index, const int64 cross_count,
              const OutType& cross) const {
    const int64 output_index = output_start_indices_[batch_index] + cross_count;

    auto indices_matrix = indices_out_->matrix<int64>();
    indices_matrix(output_index, 0) = batch_index;
    indices_matrix(output_index, 1) = cross_count;

    auto value_vec = values_out_->vec<OutType>();
    value_vec(output_index) = cross;
  }

 private:
  const std::vector<int64>& output_start_indices_;
  Tensor* indices_out_;
  Tensor* values_out_;
};

// Generates the sparse crosses as concatenation of strings.
template <typename StringType>
class StringCrosser {
 public:
  StringCrosser(
      const std::vector<std::unique_ptr<ColumnInterface<StringType>>>& columns,
      const int64 not_used)
      : columns_(columns) {}

  string Generate(const int64 batch_index,
                  const std::vector<int>& permutation) const {
    static const auto k_feature_separator = "_X_";

    gtl::InlinedVector<StringType, 6> cross_vec(columns_.size());
    for (int i = 0; i < permutation.size(); i++) {
      cross_vec[i] = columns_[i]->Feature(batch_index, permutation[i]);
    }
    // TODO(zakaria): this will copy the string twice, might effect
    // performance.
    return str_util::Join(cross_vec, k_feature_separator);
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<StringType>>>& columns_;
};

// Generates the sparse crosses as nested hash to avoid string manipulations.
template <typename StringType>
class HashCrosser {
 public:
  HashCrosser(
      const std::vector<std::unique_ptr<ColumnInterface<StringType>>>& columns,
      const int64 num_buckets)
      : columns_(columns), num_buckets_(num_buckets) {}

  int64 Generate(const int64 batch_index,
                 const std::vector<int>& permutation) const {
    uint64 hashed_output = kInitialHashSeed;
    for (int i = 0; i < permutation.size(); i++) {
      StringType str = columns_[i]->Feature(batch_index, permutation[i]);
      hashed_output = LegacyHashFunction(str.data(), str.size(), hashed_output);
    }
    if (num_buckets_ > 0) {
      return hashed_output % num_buckets_;
    } else {
      // To perevent negative output we take module to max int64.
      return hashed_output % std::numeric_limits<int64>::max();
    }
  }

 private:
  const std::vector<std::unique_ptr<ColumnInterface<StringType>>>& columns_;
  const int64 num_buckets_;
};

// ProductIterator generates cartesian products based on indices.
template <typename StringType>
class ProductIterator {
 public:
  explicit ProductIterator(
      const std::vector<std::unique_ptr<ColumnInterface<StringType>>>& columns,
      int64 batch_index)
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
  const std::vector<std::unique_ptr<ColumnInterface<StringType>>>& columns_;
  const int64 batch_index_;
  std::vector<int> next_permutation_;
};

template <bool HASHED_OUTPUT, typename StringType>
struct CrossTraits;

template <typename StringType>
struct CrossTraits<true, StringType> {
  typedef StringCrosser<StringType> Crosser;
  typedef OutputUpdater<string> Updater;
};

template <typename StringType>
struct CrossTraits<false, StringType> {
  typedef HashCrosser<StringType> Crosser;
  typedef OutputUpdater<int64> Updater;
};
}  // namespace

template <bool HASHED_OUTPUT, typename StringType>
class SparseFeatureCrossOp : public OpKernel {
 public:
  explicit SparseFeatureCrossOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList indices_list_in;
    OP_REQUIRES_OK(context, context->input_list("indices", &indices_list_in));
    OpInputList values_list_in;
    OP_REQUIRES_OK(context, context->input_list("values", &values_list_in));
    OpInputList shapes_list_in;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes_list_in));
    OpInputList dense_list_in;
    OP_REQUIRES_OK(context, context->input_list("dense", &dense_list_in));

    ValidateInput(context, indices_list_in, values_list_in, shapes_list_in,
                  dense_list_in);

    std::vector<std::unique_ptr<ColumnInterface<StringType>>> columns =
        GenerateColumnsFromInput(indices_list_in, values_list_in,
                                 shapes_list_in, dense_list_in);

    typename CrossTraits<HASHED_OUTPUT, StringType>::Crosser crosser(
        columns, num_buckets_);
    Tensor* indices_out;
    Tensor* values_out;
    Tensor* shape_out;
    const int64 batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
    std::vector<int64> output_start_indices(batch_size);
    CreateOutputTensors(columns, batch_size, context, &indices_out, &values_out,
                        &shape_out, &output_start_indices);

    typename CrossTraits<HASHED_OUTPUT, StringType>::Updater updater(
        output_start_indices, indices_out, values_out);
    auto do_work = [this, &columns, crosser, updater](int64 begin, int64 end) {
      for (int b = begin; b < end; b++) {
        ProductIterator<StringType> product_iterator(columns, b);
        int64 cross_count = 0;
        while (product_iterator.HasNext()) {
          const auto permutation = product_iterator.Next();
          updater.Update(b, cross_count, crosser.Generate(b, permutation));
          cross_count++;
        }
      }
    };

    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    // TODO(zakaria): optimize kCostPerUnit cross on column id should be
    // treated cheaper.
    const int kCostPerUnit = 50000 * indices_list_in.size();
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
          kCostPerUnit, do_work);
  }

 private:
  // Validates input tensors.
  void ValidateInput(OpKernelContext* context,
                     const OpInputList& indices_list_in,
                     const OpInputList& values_list_in,
                     const OpInputList& shapes_list_in,
                     const OpInputList& dense_list_in) {
    const auto size = indices_list_in.size();
    // Validates indices_list_in OpInputList.
    for (int i = 0; i < size; i++) {
      OP_REQUIRES(
          context, TensorShapeUtils::IsMatrix(indices_list_in[i].shape()),
          errors::InvalidArgument(
              "Input indices should be a matrix but received shape ",
              indices_list_in[i].shape().DebugString(), " at position ", i));
      OP_REQUIRES(
          context, indices_list_in[i].shape().dim_size(1) == 2,
          errors::InvalidArgument("Expected D2 of index to be 2 got",
                                  indices_list_in[i].shape().dim_size(1),
                                  " at position ", i));
    }

    // Validates values_list_in OpInputList.
    OP_REQUIRES(
        context, values_list_in.size() == size,
        errors::InvalidArgument("Expected ", size, " input values, got ",
                                values_list_in.size()));
    for (int i = 0; i < size; i++) {
      OP_REQUIRES(
          context, TensorShapeUtils::IsVector(values_list_in[i].shape()),
          errors::InvalidArgument(
              "Input values should be a std::vector but received shape ",
              values_list_in[i].shape().DebugString(), " at position ", i));
      OP_REQUIRES(
          context, indices_list_in[i].shape().dim_size(0) ==
                       values_list_in[i].shape().dim_size(0),
          errors::InvalidArgument(
              "Expected size of values to be ",
              indices_list_in[i].shape().dim_size(0), " got ",
              values_list_in[i].shape().dim_size(0), " at position ", i));
    }

    // Validates shapes_list_in OpInputList
    OP_REQUIRES(
        context, shapes_list_in.size() == size,
        errors::InvalidArgument("Expected ", size, " input shapes, got ",
                                shapes_list_in.size()));
    const auto batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
    for (int i = 0; i < size; i++) {
      OP_REQUIRES(
          context, TensorShapeUtils::IsVector(shapes_list_in[i].shape()),
          errors::InvalidArgument(
              "Input shapes should be a std::vector but received shape ",
              shapes_list_in[i].shape().DebugString(), " at position ", i));

      OP_REQUIRES(
          context, shapes_list_in[i].vec<int64>().size() == 2,
          errors::InvalidArgument("shape should imply a 2D tensor, but got ",
                                  shapes_list_in[i].shape().DebugString(),
                                  " at position ", i));
      OP_REQUIRES(context, shapes_list_in[i].vec<int64>()(0) == batch_size,
                  errors::InvalidArgument(
                      "Expected batch size ", batch_size, " got ",
                      shapes_list_in[i].vec<int64>()(0), " at position ", i));
    }

    // Validates dense_list_in OpInputList
    for (int i = 0; i < dense_list_in.size(); ++i) {
      OP_REQUIRES(
          context, TensorShapeUtils::IsMatrix(dense_list_in[i].shape()),
          errors::InvalidArgument(
              "Dense inputs should be a matrix but received shape ",
              indices_list_in[i].shape().DebugString(), " at position ", i));
      OP_REQUIRES(context, dense_list_in[i].dim_size(0) == batch_size,
                  errors::InvalidArgument("Expected batch size ", batch_size,
                                          " got ", dense_list_in[i].dim_size(0),
                                          " at dense tensor ", i));
    }
  }

  // Calculate the batch size from either the shapes input or the dense input.
  int64 CalculateBatchSize(const OpInputList& shapes_list_in,
                           const OpInputList& dense_list_in) {
    if (shapes_list_in.size() > 0) {
      return shapes_list_in[0].vec<int64>()(0);
    }

    if (dense_list_in.size() > 0) {
      return dense_list_in[0].dim_size(0);
    }

    return 0;
  }

  // Generate the columns given the sparse and dense inputs.
  std::vector<std::unique_ptr<ColumnInterface<StringType>>>
  GenerateColumnsFromInput(const OpInputList& indices_list_in,
                           const OpInputList& values_list_in,
                           const OpInputList& shapes_list_in,
                           const OpInputList& dense_list_in) {
    std::vector<std::unique_ptr<ColumnInterface<StringType>>> columns;
    const int64 batch_size = CalculateBatchSize(shapes_list_in, dense_list_in);
    const int64 number_of_columns = shapes_list_in.size();

    std::vector<std::vector<int64>> feature_counts(number_of_columns,
                                                   std::vector<int64>());
    std::vector<std::vector<int64>> feature_start_indices(number_of_columns,
                                                          std::vector<int64>());

    ExtractFeatureData(indices_list_in, batch_size, &feature_counts,
                       &feature_start_indices);

    for (int i = 0; i < values_list_in.size(); ++i) {
      columns.emplace_back(new SparseTensorColumn<StringType>(
          values_list_in[i], std::move(feature_counts[i]),
          std::move(feature_start_indices[i])));
    }
    for (int i = 0; i < dense_list_in.size(); ++i) {
      columns.emplace_back(new DenseTensorColumn<StringType>(dense_list_in[i]));
    }

    return columns;
  }

  // Extrats data about the features and populates feature data.
  void ExtractFeatureData(
      const OpInputList& indices_list_in, int64 batch_size,
      std::vector<std::vector<int64>>* feature_counts,
      std::vector<std::vector<int64>>* feature_start_indices) {
    gtl::InlinedVector<int64, 8> current_row(indices_list_in.size(), 0);
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < indices_list_in.size(); i++) {
        const auto indices = indices_list_in[i].matrix<int64>();
        int64 feature_count = 0;
        int64 start_index = current_row[i];
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

  // Allocates output tensors with proper size and sets the shape tensor of
  // the output SparseTensor.
  // It also output_start_indices which contains the start indices for each
  // input in the output SparseTensor.
  void CreateOutputTensors(
      const std::vector<std::unique_ptr<ColumnInterface<StringType>>>& columns,
      int64 batch_size, OpKernelContext* context, Tensor** indices_out,
      Tensor** values_out, Tensor** shape_out,
      std::vector<int64>* output_start_indices) {
    // Calculates dimensions for output tensors.
    int64 cross_count_total = 0;
    int64 max_cross_count = 0;
    for (int64 b = 0; b < batch_size; b++) {
      // For each input, sets starting indices in output SparseTensor
      (*output_start_indices)[b] = cross_count_total;
      const auto cross_count = CrossCountByBatchIndex(columns, b);
      max_cross_count = std::max(max_cross_count, cross_count);
      cross_count_total += cross_count;
    }

    // Allocates tensors.
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({cross_count_total, 2}), indices_out));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({cross_count_total}),
                                            values_out));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({2}), shape_out));

    // Sets shape.
    auto shape_vec = (*shape_out)->vec<int64>();
    shape_vec(0) = batch_size;
    shape_vec(1) = max_cross_count;
  }

  // Returns number of crosses for a given batch_index
  int64 CrossCountByBatchIndex(
      const std::vector<std::unique_ptr<ColumnInterface<StringType>>>& columns,
      int batch_index) {
    int64 cross_count = 1;
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
  int64 num_buckets_;
};

REGISTER_KERNEL_BUILDER(Name("SparseFeatureCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<string>("out_type")
                            .TypeConstraint<string>("internal_type"),
                        SparseFeatureCrossOp<true, StringPiece>);

REGISTER_KERNEL_BUILDER(Name("SparseFeatureCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<string>("out_type")
                            .TypeConstraint<int64>("internal_type"),
                        SparseFeatureCrossOp<true, string>);

REGISTER_KERNEL_BUILDER(Name("SparseFeatureCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("out_type")
                            .TypeConstraint<string>("internal_type"),
                        SparseFeatureCrossOp<false, StringPiece>);

REGISTER_KERNEL_BUILDER(Name("SparseFeatureCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("out_type")
                            .TypeConstraint<int64>("internal_type"),
                        SparseFeatureCrossOp<false, string>);

}  // namespace tensorflow
