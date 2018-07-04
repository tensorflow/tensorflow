/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/reshape_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

namespace {

using sparse::SparseTensor;

class DeserializeSparseOp : public OpKernel {
 public:
  explicit DeserializeSparseOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& serialized_sparse = context->input(0);
    const int ndims = serialized_sparse.shape().dims();

    OP_REQUIRES(
        context, ndims > 0,
        errors::InvalidArgument("Serialized sparse should have non-zero rank ",
                                serialized_sparse.shape().DebugString()));

    OP_REQUIRES(context, serialized_sparse.shape().dim_size(ndims - 1) == 3,
                errors::InvalidArgument(
                    "Serialized sparse should have 3 as the last dimension ",
                    serialized_sparse.shape().DebugString()));

    int num_sparse_tensors = 1;
    for (int i = 0; i < ndims - 1; ++i) {
      num_sparse_tensors *= serialized_sparse.shape().dim_size(i);
    }

    OP_REQUIRES(
        context, num_sparse_tensors > 0,
        errors::InvalidArgument(
            "Serialized sparse should have at least 1 serialized tensor, "
            "but has a zero dimension ",
            serialized_sparse.shape().DebugString()));

    if (num_sparse_tensors == 1 && ndims == 1) {
      // Special case with a single sparse tensor. We can avoid data
      // motion in the Concat and Reshape.
      const auto& serialized_sparse_t = serialized_sparse.vec<string>();

      Tensor output_indices;
      Tensor output_values;
      Tensor output_shape;
      OP_REQUIRES_OK(context,
                     this->GetAndValidateSparseTensor(
                         serialized_sparse_t(0), serialized_sparse_t(1),
                         serialized_sparse_t(2), dtype_, 0 /* index */,
                         &output_indices, &output_values, &output_shape));
      context->set_output(0, output_indices);
      context->set_output(1, output_values);
      context->set_output(2, output_shape);
      return;
    }

    std::vector<Tensor> indices;
    std::vector<Tensor> values;
    TensorShape shape;
    indices.reserve(num_sparse_tensors);
    values.reserve(num_sparse_tensors);

    const auto& serialized_sparse_t =
        serialized_sparse.flat_inner_dims<string, 2>();
    for (int i = 0; i < num_sparse_tensors; ++i) {
      Tensor output_indices;
      Tensor output_values;
      Tensor output_shape;
      OP_REQUIRES_OK(context,
                     this->GetAndValidateSparseTensor(
                         serialized_sparse_t(i, 0), serialized_sparse_t(i, 1),
                         serialized_sparse_t(i, 2), dtype_, i, &output_indices,
                         &output_values, &output_shape));
      int64 num_entries = output_indices.dim_size(0);
      int rank = output_indices.dim_size(1);

      // Now we expand each SparseTensors' indices and shape by
      // prefixing a dimension
      Tensor expanded_indices(DT_INT64, TensorShape({num_entries, 1 + rank}));
      const auto& output_indices_t = output_indices.matrix<int64>();
      auto expanded_indices_t = expanded_indices.matrix<int64>();
      expanded_indices_t.chip<1>(0).setZero();
      if (rank > 0) {
        Eigen::DSizes<Eigen::DenseIndex, 2> indices_start(0, 1);
        Eigen::DSizes<Eigen::DenseIndex, 2> indices_sizes(num_entries, rank);
        expanded_indices_t.slice(indices_start, indices_sizes) =
            output_indices_t;
      }
      Tensor expanded_shape(DT_INT64, TensorShape({1 + rank}));
      const auto& output_shape_t = output_shape.vec<int64>();
      auto expanded_shape_t = expanded_shape.vec<int64>();
      expanded_shape_t(0) = 1;
      std::copy_n(&output_shape_t(0), rank, &expanded_shape_t(1));

      TensorShape expanded_tensor_shape(expanded_shape.vec<int64>());

      indices.push_back(expanded_indices);
      values.push_back(output_values);
      if (i == 0) {
        shape = expanded_tensor_shape;
      } else {
        OP_REQUIRES(
            context, shape.dims() == expanded_tensor_shape.dims(),
            errors::InvalidArgument(
                "Inconsistent shape across SparseTensors: rank prior to "
                "SparseTensor[",
                i, "] was: ", shape.dims() - 1, " but rank of SparseTensor[", i,
                "] is: ", expanded_tensor_shape.dims() - 1));
        for (int j = 1; j < shape.dims(); ++j) {
          // NOTE(mrry): For compatibility with the implementations of
          // DeserializeManySparse, and many ops that generate
          // SparseTensors to batch that do not have a fixed
          // dense_shape (e.g. `tf.parse_single_example()`), we
          // compute the maximum in each dimension to find the
          // smallest dense_shape that bounds all of the input
          // SparseTensors.
          shape.set_dim(j, std::max(shape.dim_size(j),
                                    expanded_tensor_shape.dim_size(j)));
        }
      }
    }

    // Dimension 0 is the primary dimension.
    int rank = shape.dims();
    gtl::InlinedVector<int64, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);

    std::vector<SparseTensor> tensors;
    tensors.reserve(num_sparse_tensors);
    for (int i = 0; i < num_sparse_tensors; ++i) {
      tensors.emplace_back(indices[i], values[i], shape, std_order);
    }

    gtl::optional<SparseTensor> maybe_output;
#define HANDLE_TYPE(T)                               \
  case DataTypeToEnum<T>::value: {                   \
    maybe_output = SparseTensor::Concat<T>(tensors); \
    break;                                           \
  }

    switch (dtype_) {
      TF_CALL_ALL_TYPES(HANDLE_TYPE);
      TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
      default:
        OP_REQUIRES(context, false,
                    errors::Unimplemented(
                        "DeserializeSparse Unhandled data type: ", dtype_));
    }
    DCHECK(maybe_output);
    SparseTensor& output = maybe_output.value();

    // Compute the input shape for the reshape operation.
    Tensor input_shape(DT_INT64, TensorShape({output.dims()}));
    std::copy_n(output.shape().data(), output.dims(),
                input_shape.vec<int64>().data());

    // Compute the target shape for the reshape operation.
    Tensor target_shape(DT_INT64, TensorShape({ndims + output.dims() - 2}));
    for (int i = 0; i < ndims - 1; ++i) {
      target_shape.vec<int64>()(i) = serialized_sparse.shape().dim_size(i);
    }
    for (int i = 0; i < output.dims() - 1; ++i) {
      target_shape.vec<int64>()(i + ndims - 1) = output.shape().data()[i + 1];
    }

    Tensor output_indices;
    Tensor output_shape;
    Reshape(context, output.indices(), input_shape, target_shape,
            0 /* output indices index */, 2 /* output shape index */);
    context->set_output(1, output.values());
  }

 private:
  Status Deserialize(const string& serialized, Tensor* result) {
    TensorProto proto;
    if (!ParseProtoUnlimited(&proto, serialized)) {
      return errors::InvalidArgument("Could not parse serialized proto");
    }
    Tensor tensor;
    if (!tensor.FromProto(proto)) {
      return errors::InvalidArgument("Could not construct tensor from proto");
    }
    *result = tensor;
    return Status::OK();
  }

  Status GetAndValidateSparseTensor(
      const string& serialized_indices, const string& serialized_values,
      const string& serialized_shape, DataType values_dtype, int index,
      Tensor* output_indices, Tensor* output_values, Tensor* output_shape) {
    // Deserialize and validate the indices.
    TF_RETURN_IF_ERROR(this->Deserialize(serialized_indices, output_indices));
    if (!TensorShapeUtils::IsMatrix(output_indices->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 0] to represent an index matrix but received shape ",
          output_indices->shape().DebugString());
    }
    int64 num_entries = output_indices->dim_size(0);
    int rank = output_indices->dim_size(1);

    // Deserialize and validate the values.
    TF_RETURN_IF_ERROR(this->Deserialize(serialized_values, output_values));
    if (!TensorShapeUtils::IsVector(output_values->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 1] to represent a values vector but received shape ",
          output_values->shape().DebugString());
    }
    if (values_dtype != output_values->dtype()) {
      return errors::InvalidArgument(
          "Requested SparseTensor of type ", DataTypeString(values_dtype),
          " but SparseTensor[", index,
          "].values.dtype() == ", DataTypeString(output_values->dtype()));
    }
    if (num_entries != output_values->dim_size(0)) {
      return errors::InvalidArgument(
          "Expected row counts of SparseTensor[", index,
          "].indices and SparseTensor[", index,
          "].values to match but they do not: ", num_entries, " vs. ",
          output_values->dim_size(0));
    }

    // Deserialize and validate the shape.
    TF_RETURN_IF_ERROR(this->Deserialize(serialized_shape, output_shape));
    if (!TensorShapeUtils::IsVector(output_shape->shape())) {
      return errors::InvalidArgument(
          "Expected serialized_sparse[", index,
          ", 1] to be a shape vector but its shape is ",
          output_shape->shape().DebugString());
    }
    if (rank != output_shape->dim_size(0)) {
      return errors::InvalidArgument("Expected column counts of SparseTensor[",
                                     index,
                                     "].indices to match size of SparseTensor[",
                                     index, "].shape but they do not: ", rank,
                                     " vs. ", output_shape->dim_size(0));
    }
    return Status::OK();
  }

  DataType dtype_;
};

REGISTER_KERNEL_BUILDER(Name("DeserializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<string>("Tserialized"),
                        DeserializeSparseOp)

REGISTER_KERNEL_BUILDER(Name("DeserializeManySparse").Device(DEVICE_CPU),
                        DeserializeSparseOp)

}  // namespace

}  // namespace tensorflow
