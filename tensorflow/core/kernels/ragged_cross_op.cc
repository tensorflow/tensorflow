/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ragged_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

namespace {

//==============================================================================
// Feature Readers
//==============================================================================

// A `FeatureReader` is used to read the feature values from a single input
// tensor.  Subclasses are used for reading different tensor types:
//   * RaggedFeatureReader<value_type, splits_type>
//   * SparseFeatureReader<value_type>
//   * DenseFeatureReader<value_type>
//
// Where value_type is one of: {tstring, int64}; and SplitsType is one of:
// {int32, int64}.
class FeatureReader {
 public:
  // Returns the number of feature values in the specified batch.
  virtual int64_t FeatureCount(int64_t batch) const = 0;

  // Copies the value for the specified feature to `out`.
  virtual void ReadValue(int64_t batch, int64_t n, uint64* out) const = 0;
  virtual void ReadValue(int64_t batch, int64_t n, tstring* out) const = 0;

  virtual ~FeatureReader() {}
};

using FeatureReaders = std::vector<std::unique_ptr<FeatureReader>>;

// Copies a feature value `src` to a tstring `dst`, using a view if appropriate.
void CopyToString(const tstring& src, tstring* dst) {
  if (src.type() == tstring::SMALL) {
    *dst = src;  // string buffer fits in the tstring object (under ~24 bytes)
  } else {
    dst->assign_as_view(src);
  }
}
void CopyToString(int64_t src, tstring* dst) { *dst = std::to_string(src); }

// Copies a feature value `src` to an int64 fingerprint `dst`.
void CopyToFingerprint(const tstring& feature, uint64* dst) {
  *dst = Fingerprint64(feature);
}
void CopyToFingerprint(int64_t feature, uint64* dst) { *dst = feature; }

// A FeatureReader that is backed by a ragged tensor.
template <typename ValuesType, typename SplitsType>
class RaggedFeatureReader : public FeatureReader {
 public:
  RaggedFeatureReader(const Tensor& values, const Tensor& row_splits)
      : values_(values.flat<ValuesType>()),
        row_splits_(row_splits.flat<SplitsType>()) {}

  int64_t FeatureCount(int64_t batch) const override {
    return row_splits_(batch + 1) - row_splits_(batch);
  }

  void ReadValue(int64_t batch, int64_t n, uint64* out) const override {
    CopyToFingerprint(values_(row_splits_(batch) + n), out);
  }

  void ReadValue(int64_t batch, int64_t n, tstring* out) const override {
    CopyToString(values_(row_splits_(batch) + n), out);
  }

 private:
  const typename TTypes<ValuesType>::ConstFlat values_;
  const typename TTypes<SplitsType>::ConstFlat row_splits_;
};

// A FeatureReader that is backed by a dense tensor.
template <typename ValuesType>
class DenseFeatureReader : public FeatureReader {
 public:
  explicit DenseFeatureReader(const Tensor& tensor)
      : values_(tensor.matrix<ValuesType>()),
        feature_count_(tensor.dim_size(1)) {}

  int64_t FeatureCount(int64_t batch) const override { return feature_count_; }

  void ReadValue(int64_t batch, int64_t n, uint64* out) const override {
    CopyToFingerprint(values_(batch, n), out);
  }

  void ReadValue(int64_t batch, int64_t n, tstring* out) const override {
    CopyToString(values_(batch, n), out);
  }

 private:
  const typename TTypes<ValuesType>::ConstMatrix values_;
  const int64_t feature_count_;
};

// A FeatureReader that is backed by a sparse tensor.
template <typename ValuesType>
class SparseFeatureReader : public FeatureReader {
 public:
  SparseFeatureReader(const Tensor& indices_t, const Tensor& values_t,
                      int64_t batch_size)
      : values_(values_t.flat<ValuesType>()) {
    row_splits_.reserve(batch_size + 1);
    row_splits_.push_back(0);
    auto indices = indices_t.matrix<int64_t>();
    int64_t num_values = values_.size();
    int64_t i = 0;  // value index
    for (int row = 0; row < batch_size; row++) {
      while (i < num_values && indices(i, 0) <= row) ++i;
      row_splits_.push_back(i);
    }
  }

  int64_t FeatureCount(int64_t batch) const override {
    return row_splits_[batch + 1] - row_splits_[batch];
  }

  void ReadValue(int64_t batch, int64_t n, uint64* out) const override {
    CopyToFingerprint(values_(row_splits_[batch] + n), out);
  }

  void ReadValue(int64_t batch, int64_t n, tstring* out) const override {
    CopyToString(values_(row_splits_[batch] + n), out);
  }

 private:
  const typename TTypes<ValuesType>::ConstFlat values_;
  std::vector<int64_t> row_splits_;
};

//==============================================================================
// Output Writers
//==============================================================================

// An `OutputWriter` is used to write the feature crosses to the output values
// tensor.  Different subclasses are used for writing different output dtypes:
//   * OutputWriterImpl<tstring, SplitsType> (for tf.ragged.cross)
//   * OutputWriterImpl<int64, SplitsType> (for tf.ragged.cross_hashed)
class OutputWriter {
 public:
  virtual void WriteOutputSlice(int64_t begin, int64_t end) = 0;
  virtual ~OutputWriter() {}
};

template <typename ValuesType, typename SplitsType>
class OutputWriterImpl : public OutputWriter {
 public:
  using FlatValues = typename TTypes<ValuesType>::Flat;
  using FlatSplits = typename TTypes<SplitsType>::ConstFlat;

  OutputWriterImpl(const FeatureReaders& features, int64_t num_buckets,
                   uint64 hash_key, const Tensor* splits_out,
                   Tensor* values_out)
      : features_(features),
        num_buckets_(num_buckets),
        hash_key_(hash_key),
        splits_out_(splits_out->flat<SplitsType>()),
        values_out_(values_out->flat<ValuesType>()) {}

  // Reads features from the specified slice of batch indices, computes
  // feature crosses for each one, and writes them to values_out_.
  void WriteOutputSlice(int64_t begin, int64_t end) override {
    std::vector<int> combination(features_.size(), 0);
    for (int64_t b = begin; b < end; ++b) {
      auto row_start = splits_out_(b);
      auto row_limit = splits_out_(b + 1);
      for (auto i = row_start; i < row_limit; ++i) {
        WriteCombination(b, combination, &values_out_(i));
        NextCombination(b, &combination);
      }
      combination.assign(features_.size(), 0);  // reset for next batch.
    }
  }

 private:
  // Joins the specified combination of input features into a single string,
  // and writes it to *out.
  void WriteCombination(int64_t batch_index,
                        const std::vector<int>& combination, tstring* out) {
    static const auto k_feature_separator = "_X_";
    absl::InlinedVector<tstring, 6> cross_vec(features_.size());
    for (int i = 0; i < combination.size(); ++i) {
      features_[i]->ReadValue(batch_index, combination[i], &cross_vec[i]);
    }
    *out = absl::StrJoin(cross_vec, k_feature_separator);
  }

  // Joins the specified combination of input features into a single
  // fingerprint, and writes it to *out.
  void WriteCombination(int64_t batch_index,
                        const std::vector<int>& combination, int64_t* out) {
    // Do the fingerprint concatenation on uint64.
    uint64 hashed_output = hash_key_;
    for (size_t i = 0; i < combination.size(); ++i) {
      uint64 hash_i;
      features_[i]->ReadValue(batch_index, combination[i], &hash_i);
      hashed_output = FingerprintCat64(hashed_output, hash_i);
    }
    // The return value is int64 based on the number of buckets.
    if (num_buckets_ > 0) {
      *out = hashed_output % num_buckets_;
    } else {
      // To prevent negative output we take modulo to max int64.
      *out = hashed_output % std::numeric_limits<int64_t>::max();
    }
  }

  // Updates `combination` to the next combination of input features.
  void NextCombination(int64_t batch_index,
                       std::vector<int>* combination) const {
    bool carry = true;
    for (int i = combination->size() - 1; i >= 0; i--) {
      if (carry) {
        (*combination)[i] = (*combination)[i] + 1;
      }
      if ((*combination)[i] == features_[i]->FeatureCount(batch_index)) {
        (*combination)[i] = 0;
      } else {
        carry = false;
        break;
      }
    }
  }

  const FeatureReaders& features_;
  const int64_t num_buckets_;
  const uint64 hash_key_;
  FlatSplits splits_out_;
  FlatValues values_out_;
};

// Returns an appropriate OutputWriter, based on the dtypes of the
// given tensors.
std::unique_ptr<OutputWriter> MakeOutputWriter(const FeatureReaders& features,
                                               int64_t num_buckets,
                                               uint64 hash_key,
                                               const Tensor* splits_out,
                                               Tensor* values_out) {
  if (values_out->dtype() == DT_INT64) {
    if (splits_out->dtype() == DT_INT64) {
      return std::make_unique<OutputWriterImpl<int64_t, int64_t>>(
          features, num_buckets, hash_key, splits_out, values_out);
    } else {
      return std::make_unique<OutputWriterImpl<int64_t, int32>>(
          features, num_buckets, hash_key, splits_out, values_out);
    }
  } else {
    if (splits_out->dtype() == DT_INT64) {
      return std::make_unique<OutputWriterImpl<tstring, int64_t>>(
          features, num_buckets, hash_key, splits_out, values_out);
    } else {
      return std::make_unique<OutputWriterImpl<tstring, int32>>(
          features, num_buckets, hash_key, splits_out, values_out);
    }
  }
}

//==============================================================================
// RaggedCross Kernel
//==============================================================================

template <typename SplitsType>
class RaggedCrossOp : public OpKernel {
 public:
  explicit RaggedCrossOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
    // Read signed_hash_key_ as int64 since uint64 attributes are not
    // supported by REGISTER_OP.
    int64_t signed_hash_key_;
    OP_REQUIRES_OK(context, context->GetAttr("hash_key", &signed_hash_key_));
    hash_key_ = static_cast<uint64>(signed_hash_key_);

    int num_sparse;
    OP_REQUIRES_OK(context, context->GetAttr("Nsparse", &num_sparse));

    OP_REQUIRES_OK(context, context->GetAttr("ragged_values_types",
                                             &ragged_values_types_));
    OP_REQUIRES_OK(context, context->GetAttr("ragged_splits_types",
                                             &ragged_splits_types_));
    OP_REQUIRES_OK(context, context->GetAttr("sparse_values_types",
                                             &sparse_values_types_));
    OP_REQUIRES_OK(context, context->GetAttr("dense_types", &dense_types_));
    OP_REQUIRES_OK(context, context->GetAttr("input_order", &input_order_));
    OP_REQUIRES(context,
                ragged_values_types_.size() == ragged_splits_types_.size(),
                errors::InvalidArgument(
                    "ragged values and splits must have the same length"));
    OP_REQUIRES(context, num_sparse == sparse_values_types_.size(),
                errors::InvalidArgument(
                    "sparse indices and values must have the same length"));
    OP_REQUIRES(context,
                ragged_values_types_.size() + sparse_values_types_.size() +
                        dense_types_.size() ==
                    input_order_.size(),
                errors::InvalidArgument("Invalid length for input_order"));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList ragged_values_list;
    OpInputList ragged_splits_list;
    OpInputList sparse_indices_list;
    OpInputList sparse_values_list;
    OpInputList sparse_shape_list;
    OpInputList dense_list;
    OP_REQUIRES_OK(context,
                   context->input_list("ragged_values", &ragged_values_list));
    OP_REQUIRES_OK(
        context, context->input_list("ragged_row_splits", &ragged_splits_list));
    OP_REQUIRES_OK(context,
                   context->input_list("sparse_indices", &sparse_indices_list));
    OP_REQUIRES_OK(context,
                   context->input_list("sparse_values", &sparse_values_list));
    OP_REQUIRES_OK(context,
                   context->input_list("sparse_shape", &sparse_shape_list));
    OP_REQUIRES_OK(context, context->input_list("dense_inputs", &dense_list));
    OP_REQUIRES_OK(context,
                   ValidateInput(ragged_values_list, ragged_splits_list,
                                 sparse_indices_list, sparse_values_list,
                                 sparse_shape_list, dense_list));

    int64_t batch_size =
        CalculateBatchSize(ragged_splits_list, sparse_shape_list, dense_list);

    FeatureReaders features;
    OP_REQUIRES_OK(context,
                   BuildFeatureReaders(ragged_values_list, ragged_splits_list,
                                       sparse_indices_list, sparse_values_list,
                                       dense_list, batch_size, &features));

    Tensor* values_out;
    Tensor* row_splits_out;
    OP_REQUIRES_OK(context, BuildOutputTensors(features, batch_size, context,
                                               &values_out, &row_splits_out));

    std::unique_ptr<OutputWriter> output_writer = MakeOutputWriter(
        features, num_buckets_, hash_key_, row_splits_out, values_out);

    auto do_work = [&output_writer](int64_t begin, int64_t end) {
      output_writer->WriteOutputSlice(begin, end);
    };

    // TODO(edloper): optimize cost_per_batch
    const int cost_per_batch = 5000 * ragged_values_list.size();
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(batch_size, cost_per_batch, do_work);
  }

 private:
  // Validates input tensors.
  Status ValidateInput(const OpInputList& ragged_values_list,
                       const OpInputList& ragged_splits_list,
                       const OpInputList& sparse_indices_list,
                       const OpInputList& sparse_values_list,
                       const OpInputList& sparse_shape_list,
                       const OpInputList& dense_list) {
    const auto num_ragged = ragged_values_list.size();
    const auto num_sparse = sparse_indices_list.size();

    // Validate tensor shapes.
    for (int i = 0; i < num_ragged; ++i) {
      if (!TensorShapeUtils::IsVector(ragged_values_list[i].shape()) ||
          !TensorShapeUtils::IsVector(ragged_splits_list[i].shape())) {
        return absl::InvalidArgumentError(
            "tf.ragged.cross only supports inputs with rank=2.");
      }

      int64_t num_values = ragged_values_list[i].NumElements();
      TF_RETURN_IF_ERROR(RaggedTensorVerifySplits<SplitsType>(
          ragged_splits_list[i], true, num_values));
    }
    for (int i = 0; i < num_sparse; ++i) {
      if (!TensorShapeUtils::IsMatrix(sparse_indices_list[i].shape()) ||
          !TensorShapeUtils::IsVector(sparse_values_list[i].shape()) ||
          !TensorShapeUtils::IsVector(sparse_shape_list[i].shape())) {
        return errors::InvalidArgument("Invalid SparseTensor ", i);
      }
      if (sparse_shape_list[i].NumElements() != 2) {
        return errors::InvalidArgument(
            "tf.ragged.cross only supports inputs with rank=2.");
      }
    }
    for (int i = 0; i < dense_list.size(); ++i) {
      if (!TensorShapeUtils::IsMatrix(dense_list[i].shape())) {
        return errors::InvalidArgument(
            "tf.ragged.cross only supports inputs with rank=2.");
      }
    }

    // Check that batch sizes are consistent.
    int64_t batch_size =
        CalculateBatchSize(ragged_splits_list, sparse_shape_list, dense_list);
    for (int i = 0; i < num_ragged; ++i) {
      if (ragged_splits_list[i].NumElements() - 1 != batch_size) {
        return errors::InvalidArgument(
            "inputs must all have the same batch dimension size.");
      }
    }
    for (int i = 0; i < num_sparse; ++i) {
      if (sparse_shape_list[i].flat<int64_t>()(0) != batch_size) {
        return errors::InvalidArgument(
            "inputs must all have the same batch dimension size.");
      }
    }
    for (int i = 0; i < dense_list.size(); ++i) {
      if (dense_list[i].dim_size(0) != batch_size) {
        return errors::InvalidArgument(
            "inputs must all have the same batch dimension size.");
      }
    }

    return absl::OkStatus();
  }

  // Calculate the batch size from any input tensor.  (We check that all input
  // tensors have the same batch size in `ValidateInput`).
  int64_t CalculateBatchSize(const OpInputList& ragged_splits_list,
                             const OpInputList& sparse_shape_list,
                             const OpInputList& dense_list) {
    if (ragged_splits_list.size() > 0) {
      return ragged_splits_list[0].NumElements() - 1;
    } else if (dense_list.size() > 0) {
      return dense_list[0].dim_size(0);
    } else if (sparse_shape_list.size() > 0) {
      return sparse_shape_list[0].flat<int64_t>()(0);
    } else {
      return 0;
    }
  }

  // Build a feature reader for each input tensor, and store them in `features`.
  Status BuildFeatureReaders(const OpInputList& ragged_values_list,
                             const OpInputList& ragged_splits_list,
                             const OpInputList& sparse_indices_list,
                             const OpInputList& sparse_values_list,
                             const OpInputList& dense_list, int64_t batch_size,
                             FeatureReaders* features) {
    features->reserve(input_order_.size());

    int next_ragged = 0;
    int next_sparse = 0;
    int next_dense = 0;
    for (char c : input_order_) {
      if (c == 'R') {
        if (next_ragged >= ragged_values_list.size())
          return errors::InvalidArgument(
              "input_order \"", input_order_,
              "\" specifies reading a ragged tensor value at index ",
              next_ragged, " from a list of ", ragged_values_list.size(),
              " values.");
        if (next_ragged >= ragged_splits_list.size())
          return errors::InvalidArgument(
              "input_order \"", input_order_,
              "\" specifies reading a ragged tensor split at index ",
              next_ragged, " from a list of ", ragged_splits_list.size(),
              " splits.");
        TF_RETURN_IF_ERROR(BuildRaggedFeatureReader(
            ragged_values_list[next_ragged], ragged_splits_list[next_ragged],
            features));
        next_ragged++;
      } else if (c == 'S') {
        if (next_sparse >= sparse_values_list.size())
          return errors::InvalidArgument(
              "input_order \"", input_order_,
              "\" specifies reading a sparse tensor value at index ",
              next_sparse, " from a list of ", sparse_values_list.size(),
              " values.");
        if (next_sparse >= sparse_indices_list.size())
          return errors::InvalidArgument(
              "input_order \"", input_order_,
              "\" specifies reading a sparse tensor index at index ",
              next_sparse, " from a list of ", sparse_indices_list.size(),
              " indices.");
        TF_RETURN_IF_ERROR(BuildSparseFeatureReader(
            sparse_indices_list[next_sparse], sparse_values_list[next_sparse],
            batch_size, features));
        next_sparse++;
      } else if (c == 'D') {
        if (next_dense >= dense_list.size())
          return errors::InvalidArgument(
              "input_order \"", input_order_,
              "\" specifies reading a dense tensor at index ", next_dense,
              " from a list of ", dense_list.size(), " tensors.");
        TF_RETURN_IF_ERROR(
            BuildDenseFeatureReader(dense_list[next_dense++], features));
      } else {
        return errors::InvalidArgument("Unexpected input_order value.");
      }
    }

    return absl::OkStatus();
  }

  // Builds a RaggedReatureReader
  static Status BuildRaggedFeatureReader(const Tensor& values,
                                         const Tensor& splits,
                                         FeatureReaders* features) {
    if (values.dtype() != DT_INT64 && values.dtype() != DT_STRING) {
      return errors::InvalidArgument("Unexpected dtype for input ",
                                     (features->size() + 1), ": ",
                                     values.dtype());
    }
    if (splits.dtype() != DT_INT64 && splits.dtype() != DT_INT32) {
      return errors::InvalidArgument("Unexpected row_splits.dtype for input ",
                                     (features->size() + 1), ": ",
                                     values.dtype());
    }
    if (values.dtype() == DT_INT64) {
      if (splits.dtype() == DT_INT64) {
        features->emplace_back(
            new RaggedFeatureReader<int64_t, int64_t>(values, splits));
      } else {
        features->emplace_back(
            new RaggedFeatureReader<int64_t, int32>(values, splits));
      }
    } else {
      if (splits.dtype() == DT_INT64) {
        features->emplace_back(
            new RaggedFeatureReader<tstring, int64_t>(values, splits));
      } else {
        features->emplace_back(
            new RaggedFeatureReader<tstring, int32>(values, splits));
      }
    }
    return absl::OkStatus();
  }

  // Builds a DenseFaggedReatureReader.
  static Status BuildDenseFeatureReader(const Tensor& values,
                                        FeatureReaders* features) {
    if (values.dtype() == DT_INT64) {
      features->emplace_back(new DenseFeatureReader<int64_t>(values));
    } else if (values.dtype() == DT_STRING) {
      features->emplace_back(new DenseFeatureReader<tstring>(values));
    } else {
      return errors::InvalidArgument("Unexpected dtype for input ",
                                     (features->size() + 1), ": ",
                                     values.dtype());
    }
    return absl::OkStatus();
  }

  // Builds a SparseFaggedReatureReader.
  static Status BuildSparseFeatureReader(const Tensor& indices,
                                         const Tensor& values,
                                         int64_t batch_size,
                                         FeatureReaders* features) {
    if (values.dtype() == DT_INT64) {
      features->emplace_back(
          new SparseFeatureReader<int64_t>(indices, values, batch_size));
    } else if (values.dtype() == DT_STRING) {
      features->emplace_back(
          new SparseFeatureReader<tstring>(indices, values, batch_size));
    } else {
      return errors::InvalidArgument("Unexpected dtype for input ",
                                     (features->size() + 1), ": ",
                                     values.dtype());
    }
    return absl::OkStatus();
  }

  // Allocates output tensors with proper size, and populates row_splits_out.
  Status BuildOutputTensors(const FeatureReaders& features, int64_t batch_size,
                            OpKernelContext* context, Tensor** values_out,
                            Tensor** row_splits_out) {
    // Allocate and populate the row_splits output tensor.
    TF_RETURN_IF_ERROR(context->allocate_output(
        1, TensorShape({batch_size + 1}), row_splits_out));
    auto flat_row_splits = (*row_splits_out)->flat<SplitsType>();
    int64_t cross_count_total = 0;
    flat_row_splits(0) = 0;
    for (int64_t b = 0; b < batch_size; b++) {
      int64_t cross_count_by_batch_index = CrossCountByBatchIndex(features, b);
      if (cross_count_by_batch_index < 0) {
        return errors::InvalidArgument("Invalid RaggedTensor");
      }
      cross_count_total += cross_count_by_batch_index;
      flat_row_splits(b + 1) = cross_count_total;
    }

    // Allocate the values output tensor.
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({cross_count_total}), values_out));

    return absl::OkStatus();
  }

  // Returns number of crosses for a given batch_index
  int64_t CrossCountByBatchIndex(const FeatureReaders& features,
                                 int batch_index) {
    int64_t cross_count = 1;
    for (int i = 0; i < features.size(); ++i) {
      const auto feature_count = features[i]->FeatureCount(batch_index);
      // If feature_count is invalid, return -1 to let caller know.
      if (feature_count < 0) return -1;
      if (feature_count == 0) return 0;
      cross_count *= feature_count;
    }
    return cross_count;
  }

  int64_t num_buckets_;
  uint64 hash_key_;
  std::vector<DataType> ragged_values_types_;
  std::vector<DataType> ragged_splits_types_;
  std::vector<DataType> sparse_values_types_;
  std::vector<DataType> dense_types_;
  tstring input_order_;
};

REGISTER_KERNEL_BUILDER(Name("RaggedCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("out_row_splits_type"),
                        RaggedCrossOp<int32>);
REGISTER_KERNEL_BUILDER(Name("RaggedCross")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64_t>("out_row_splits_type"),
                        RaggedCrossOp<int64_t>);

}  // namespace
}  // namespace tensorflow
