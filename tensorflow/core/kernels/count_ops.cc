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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using BatchedIntMap = std::vector<absl::flat_hash_map<int64, int64>>;

namespace {
// TODO(momernick): Extend this function to work with outputs of rank > 2.
Status OutputSparse(const BatchedIntMap& per_batch_counts, int num_values,
                    bool is_1d, OpKernelContext* context) {
  int total_values = 0;
  int num_batches = per_batch_counts.size();
  for (const auto& per_batch_count : per_batch_counts) {
    total_values += per_batch_count.size();
  }

  Tensor* indices;
  int inner_dim = is_1d ? 1 : 2;
  TF_RETURN_IF_ERROR(context->allocate_output(
      0, TensorShape({total_values, inner_dim}), &indices));

  Tensor* values;
  TF_RETURN_IF_ERROR(
      context->allocate_output(1, TensorShape({total_values}), &values));

  auto output_indices = indices->matrix<int64>();
  auto output_values = values->flat<int64>();
  int64 value_loc = 0;
  for (int b = 0; b < num_batches; ++b) {
    const auto& per_batch_count = per_batch_counts[b];
    std::vector<std::pair<int, int>> pairs(per_batch_count.begin(),
                                           per_batch_count.end());
    std::sort(pairs.begin(), pairs.end());
    for (const auto& x : pairs) {
      if (is_1d) {
        output_indices(value_loc, 0) = x.first;
      } else {
        output_indices(value_loc, 0) = b;
        output_indices(value_loc, 1) = x.first;
      }
      output_values(value_loc) = x.second;
      ++value_loc;
    }
  }
  Tensor* dense_shape;
  if (is_1d) {
    TF_RETURN_IF_ERROR(
        context->allocate_output(2, TensorShape({1}), &dense_shape));
    dense_shape->flat<int64>().data()[0] = num_values;
  } else {
    TF_RETURN_IF_ERROR(
        context->allocate_output(2, TensorShape({2}), &dense_shape));
    dense_shape->flat<int64>().data()[0] = num_batches;
    dense_shape->flat<int64>().data()[1] = num_values;
  }

  return Status::OK();
}

Status OutputWeightedSparse(const BatchedIntMap& per_batch_counts,
                            int num_values, const Tensor& weights, bool is_1d,
                            OpKernelContext* context) {
  if (!TensorShapeUtils::IsVector(weights.shape())) {
    return errors::InvalidArgument(
        "Weights must be a 1-dimensional tensor. Got: ",
        weights.shape().DebugString());
  }

  if (num_values > weights.dim_size(0)) {
    return errors::InvalidArgument("The maximum array value was ", num_values,
                                   ", but the weight array has size ",
                                   weights.shape().DebugString());
  }
  auto weight_values = weights.flat<float>();

  int total_values = 0;
  int num_batches = per_batch_counts.size();
  for (const auto& per_batch_count : per_batch_counts) {
    total_values += per_batch_count.size();
  }

  Tensor* indices;
  int inner_dim = is_1d ? 1 : 2;
  TF_RETURN_IF_ERROR(context->allocate_output(
      0, TensorShape({total_values, inner_dim}), &indices));

  Tensor* values;
  TF_RETURN_IF_ERROR(
      context->allocate_output(1, TensorShape({total_values}), &values));

  auto output_indices = indices->matrix<int64>();
  auto output_values = values->flat<float>();
  int64 value_loc = 0;
  for (int b = 0; b < num_batches; ++b) {
    const auto& per_batch_count = per_batch_counts[b];
    std::vector<std::pair<int, int>> pairs(per_batch_count.begin(),
                                           per_batch_count.end());
    std::sort(pairs.begin(), pairs.end());
    for (const auto& x : pairs) {
      if (is_1d) {
        output_indices(value_loc, 0) = x.first;
      } else {
        output_indices(value_loc, 0) = b;
        output_indices(value_loc, 1) = x.first;
      }
      output_values(value_loc) = x.second * weight_values(x.first);
      ++value_loc;
    }
  }

  Tensor* dense_shape;
  if (is_1d) {
    TF_RETURN_IF_ERROR(
        context->allocate_output(2, TensorShape({1}), &dense_shape));
    dense_shape->flat<int64>().data()[0] = num_values;
  } else {
    TF_RETURN_IF_ERROR(
        context->allocate_output(2, TensorShape({2}), &dense_shape));
    dense_shape->flat<int64>().data()[0] = num_batches;
    dense_shape->flat<int64>().data()[1] = num_values;
  }
  return Status::OK();
}

template <class T>
T GetOutputSize(T max_seen, T max_length, T min_length) {
  return max_length > 0 ? max_length : std::max((max_seen + 1), min_length);
}

}  // namespace

template <class T>
class DenseCount : public OpKernel {
 public:
  explicit DenseCount(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("minlength", &minlength_));
    OP_REQUIRES_OK(context, context->GetAttr("maxlength", &maxlength_));
    OP_REQUIRES_OK(context, context->GetAttr("binary_count", &binary_count_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& weights = context->input(1);
    bool use_weights = weights.NumElements() > 0;

    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(data.shape()) ||
                    TensorShapeUtils::IsMatrix(data.shape()),
                errors::InvalidArgument(
                    "Input must be a 1 or 2-dimensional tensor. Got: ",
                    data.shape().DebugString()));

    bool is_1d = TensorShapeUtils::IsVector(data.shape());
    int negative_valued_axis = -1;
    int num_batch_dimensions = (data.shape().dims() + negative_valued_axis);

    int num_batch_elements = 1;
    for (int i = 0; i < num_batch_dimensions; ++i) {
      num_batch_elements *= data.shape().dim_size(i);
    }
    int num_value_elements = data.shape().num_elements() / num_batch_elements;
    auto per_batch_counts = BatchedIntMap(num_batch_elements);
    T max_value = 0;

    const auto data_values = data.flat<T>();
    int i = 0;
    for (int b = 0; b < num_batch_elements; ++b) {
      for (int v = 0; v < num_value_elements; ++v) {
        const auto& value = data_values(i);
        if (value >= 0 && (maxlength_ <= 0 || value < maxlength_)) {
          if (binary_count_) {
            (per_batch_counts[b])[value] = 1;
          } else {
            (per_batch_counts[b])[value]++;
          }
          if (value > max_value) {
            max_value = value;
          }
        }
        ++i;
      }
    }

    T num_output_values = GetOutputSize<T>(max_value, maxlength_, minlength_);
    if (use_weights) {
      OP_REQUIRES_OK(context,
                     OutputWeightedSparse(per_batch_counts, num_output_values,
                                          weights, is_1d, context));
    } else {
      OP_REQUIRES_OK(context, OutputSparse(per_batch_counts, num_output_values,
                                           is_1d, context));
    }
  }

 private:
  T minlength_;
  T maxlength_;
  bool binary_count_;
};

template <class T>
class SparseCount : public OpKernel {
 public:
  explicit SparseCount(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("minlength", &minlength_));
    OP_REQUIRES_OK(context, context->GetAttr("maxlength", &maxlength_));
    OP_REQUIRES_OK(context, context->GetAttr("binary_count", &binary_count_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& shape = context->input(2);
    const Tensor& weights = context->input(3);
    bool use_weights = weights.NumElements() > 0;

    bool is_1d = shape.NumElements() == 1;
    const auto indices_values = indices.matrix<int64>();
    const auto values_values = values.flat<T>();

    int num_batches = is_1d ? 1 : shape.flat<int64>()(0);
    int num_values = values.NumElements();

    auto per_batch_counts = BatchedIntMap(num_batches);
    T max_value = 0;

    for (int idx = 0; idx < num_values; ++idx) {
      int batch = is_1d ? 0 : indices_values(idx, 0);
      const auto& value = values_values(idx);
      if (value >= 0 && (maxlength_ <= 0 || value < maxlength_)) {
        if (binary_count_) {
          (per_batch_counts[batch])[value] = 1;
        } else {
          (per_batch_counts[batch])[value]++;
        }
        if (value > max_value) {
          max_value = value;
        }
      }
    }

    T num_output_values = GetOutputSize<T>(max_value, maxlength_, minlength_);
    if (use_weights) {
      OP_REQUIRES_OK(context,
                     OutputWeightedSparse(per_batch_counts, num_output_values,
                                          weights, is_1d, context));
    } else {
      OP_REQUIRES_OK(context, OutputSparse(per_batch_counts, num_output_values,
                                           is_1d, context));
    }
  }

 private:
  T minlength_;
  T maxlength_;
  bool binary_count_;
};

template <class T>
class RaggedCount : public OpKernel {
 public:
  explicit RaggedCount(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("minlength", &minlength_));
    OP_REQUIRES_OK(context, context->GetAttr("maxlength", &maxlength_));
    OP_REQUIRES_OK(context, context->GetAttr("binary_count", &binary_count_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& splits = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& weights = context->input(2);
    bool use_weights = weights.NumElements() > 0;

    const auto splits_values = splits.flat<int64>();
    const auto values_values = values.flat<T>();
    int num_batches = splits.NumElements() - 1;
    int num_values = values.NumElements();

    auto per_batch_counts = BatchedIntMap(num_batches);
    T max_value = 0;
    int batch_idx = 0;

    for (int idx = 0; idx < num_values; ++idx) {
      while (idx >= splits_values(batch_idx)) {
        batch_idx++;
      }
      const auto& value = values_values(idx);
      if (value >= 0 && (maxlength_ <= 0 || value < maxlength_)) {
        if (binary_count_) {
          (per_batch_counts[batch_idx - 1])[value] = 1;
        } else {
          (per_batch_counts[batch_idx - 1])[value]++;
        }
        if (value > max_value) {
          max_value = value;
        }
      }
    }

    T num_output_values = GetOutputSize<T>(max_value, maxlength_, minlength_);
    if (use_weights) {
      OP_REQUIRES_OK(context,
                     OutputWeightedSparse(per_batch_counts, num_output_values,
                                          weights, false, context));
    } else {
      OP_REQUIRES_OK(context, OutputSparse(per_batch_counts, num_output_values,
                                           false, context));
    }
  }

 private:
  T minlength_;
  T maxlength_;
  bool binary_count_;
};

#define REGISTER(TYPE)                                    \
                                                          \
  REGISTER_KERNEL_BUILDER(Name("DenseCountSparseOutput")  \
                              .TypeConstraint<TYPE>("T")  \
                              .Device(DEVICE_CPU),        \
                          DenseCount<TYPE>)               \
                                                          \
  REGISTER_KERNEL_BUILDER(Name("SparseCountSparseOutput") \
                              .TypeConstraint<TYPE>("T")  \
                              .Device(DEVICE_CPU),        \
                          SparseCount<TYPE>)              \
                                                          \
  REGISTER_KERNEL_BUILDER(Name("RaggedCountSparseOutput") \
                              .TypeConstraint<TYPE>("T")  \
                              .Device(DEVICE_CPU),        \
                          RaggedCount<TYPE>)

REGISTER(int32);
REGISTER(int64);
#undef REGISTER

}  // namespace tensorflow
