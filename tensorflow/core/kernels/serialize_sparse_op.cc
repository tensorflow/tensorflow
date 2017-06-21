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
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using sparse::SparseTensor;

class SerializeSparseOp : public OpKernel {
 public:
  explicit SerializeSparseOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;
    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));

    TensorProto proto_indices;
    TensorProto proto_values;
    TensorProto proto_shape;

    input_indices->AsProtoTensorContent(&proto_indices);
    input_values->AsProtoTensorContent(&proto_values);
    input_shape->AsProtoTensorContent(&proto_shape);

    Tensor serialized_sparse(DT_STRING, TensorShape({3}));
    auto serialized_sparse_t = serialized_sparse.vec<string>();

    serialized_sparse_t(0) = proto_indices.SerializeAsString();
    serialized_sparse_t(1) = proto_values.SerializeAsString();
    serialized_sparse_t(2) = proto_shape.SerializeAsString();

    context->set_output(0, serialized_sparse);
  }
};

REGISTER_KERNEL_BUILDER(Name("SerializeSparse").Device(DEVICE_CPU),
                        SerializeSparseOp);

template <typename T>
class SerializeManySparseOp : public OpKernel {
 public:
  explicit SerializeManySparseOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;
    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));

    int rank = input_shape->NumElements();

    OP_REQUIRES(
        context, rank > 1,
        errors::InvalidArgument(
            "Rank of input SparseTensor should be > 1, but saw rank: ", rank));

    TensorShape tensor_input_shape(input_shape->vec<int64>());
    gtl::InlinedVector<int64, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);
    SparseTensor input_st(*input_indices, *input_values, tensor_input_shape,
                          std_order);

    auto input_shape_t = input_shape->vec<int64>();
    const int64 N = input_shape_t(0);

    Tensor serialized_sparse(DT_STRING, TensorShape({N, 3}));
    auto serialized_sparse_t = serialized_sparse.matrix<string>();

    OP_REQUIRES_OK(context, input_st.IndicesValid());

    // We can generate the output shape proto string now, for all
    // minibatch entries.
    Tensor output_shape(DT_INT64, {rank - 1});
    auto output_shape_t = output_shape.vec<int64>();
    for (int d = 1; d < rank; d++) output_shape_t(d - 1) = input_shape_t(d);
    TensorProto proto_shape;
    output_shape.AsProtoTensorContent(&proto_shape);
    const string proto_shape_string = proto_shape.SerializeAsString();

    Tensor output_blank_indices(DT_INT64, {0, rank - 1});
    Tensor output_blank_values(DataTypeToEnum<T>::value, {0});
    TensorProto proto_blank_indices;
    TensorProto proto_blank_values;
    output_blank_indices.AsProtoTensorContent(&proto_blank_indices);
    output_blank_values.AsProtoTensorContent(&proto_blank_values);

    const string proto_blank_indices_string =
        proto_blank_indices.SerializeAsString();
    const string proto_blank_values_string =
        proto_blank_values.SerializeAsString();

    // Initialize output with empty values and the proper shapes.
    serialized_sparse_t.chip<1>(0).setConstant(proto_blank_indices_string);
    serialized_sparse_t.chip<1>(1).setConstant(proto_blank_values_string);
    serialized_sparse_t.chip<1>(2).setConstant(proto_shape_string);

    // Get groups by minibatch dimension
    sparse::GroupIterable minibatch = input_st.group({0});
    for (const auto& subset : minibatch) {
      const int64 b = subset.group()[0];
      OP_REQUIRES(
          context, b > -1 && b < N,
          errors::InvalidArgument(
              "Received unexpected column 0 value in input SparseTensor: ", b,
              " < 0 or >= N (= ", N, ")"));

      const auto indices = subset.indices();
      const auto values = subset.values<T>();
      const int64 num_entries = values.size();

      Tensor output_indices = Tensor(DT_INT64, {num_entries, rank - 1});
      Tensor output_values = Tensor(DataTypeToEnum<T>::value, {num_entries});

      auto output_indices_t = output_indices.matrix<int64>();
      auto output_values_t = output_values.vec<T>();

      for (int i = 0; i < num_entries; ++i) {
        for (int d = 1; d < rank; ++d) {
          output_indices_t(i, d - 1) = indices(i, d);
        }
        output_values_t(i) = values(i);
      }

      TensorProto proto_indices;
      TensorProto proto_values;
      output_indices.AsProtoTensorContent(&proto_indices);
      output_values.AsProtoTensorContent(&proto_values);

      serialized_sparse_t(b, 0) = proto_indices.SerializeAsString();
      serialized_sparse_t(b, 1) = proto_values.SerializeAsString();
    }

    context->set_output(0, serialized_sparse);
  }
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SerializeManySparse")     \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          SerializeManySparseOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename T>
class DeserializeManySparseOp : public OpKernel {
 public:
  explicit DeserializeManySparseOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& serialized_sparse = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(serialized_sparse.shape()),
                errors::InvalidArgument(
                    "Serialized sparse should be a matrix but received shape ",
                    serialized_sparse.shape().DebugString()));
    OP_REQUIRES(
        context, serialized_sparse.shape().dim_size(1) == 3,
        errors::InvalidArgument(
            "Serialize sparse should have 3 columns but received shape ",
            serialized_sparse.shape().DebugString()));

    int num_sparse_tensors = serialized_sparse.shape().dim_size(0);

    OP_REQUIRES(
        context, num_sparse_tensors > 0,
        errors::InvalidArgument("Must have at least 1 serialized SparseTensor, "
                                "but input matrix has 0 rows"));

    std::vector<Tensor> indices_to_concat;
    std::vector<Tensor> values_to_concat;
    std::vector<TensorShape> shapes_to_concat;

    const auto& serialized_sparse_t = serialized_sparse.matrix<string>();

    for (int i = 0; i < num_sparse_tensors; ++i) {
      Tensor output_indices(DT_INT64);
      Tensor output_values(DataTypeToEnum<T>::value);
      Tensor output_shape(DT_INT64);
      TensorProto proto_indices;
      TensorProto proto_values;
      TensorProto proto_shape;

      OP_REQUIRES(context, ParseProtoUnlimited(&proto_indices,
                                               serialized_sparse_t(i, 0)),
                  errors::InvalidArgument("Could not parse serialized_sparse[",
                                          i, ", 0]"));
      OP_REQUIRES(context,
                  ParseProtoUnlimited(&proto_values, serialized_sparse_t(i, 1)),
                  errors::InvalidArgument("Could not parse serialized_sparse[",
                                          i, ", 1]"));
      OP_REQUIRES(context,
                  ParseProtoUnlimited(&proto_shape, serialized_sparse_t(i, 2)),
                  errors::InvalidArgument("Could not parse serialized_sparse[",
                                          i, ", 2]"));

      OP_REQUIRES(context, output_indices.FromProto(proto_indices),
                  errors::InvalidArgument(
                      "Could not construct Tensor serialized_sparse[", i,
                      ", 0] (indices)"));
      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(output_indices.shape()),
                  errors::InvalidArgument(
                      "Expected serialized_sparse[", i,
                      ", 1] to represent an index matrix but received shape ",
                      output_indices.shape().DebugString()));
      OP_REQUIRES(context, output_values.FromProto(proto_values),
                  errors::InvalidArgument(
                      "Could not construct Tensor serialized_sparse[", i,
                      ", 1] (values)"));
      OP_REQUIRES(context, TensorShapeUtils::IsVector(output_values.shape()),
                  errors::InvalidArgument(
                      "Expected serialized_sparse[", i,
                      ", 1] to represent a values vector but received shape ",
                      output_values.shape().DebugString()));
      OP_REQUIRES(context, output_shape.FromProto(proto_shape),
                  errors::InvalidArgument(
                      "Could not construct Tensor serialized_sparse[", i,
                      ", 2] (shape)"));
      OP_REQUIRES(
          context, TensorShapeUtils::IsVector(output_shape.shape()),
          errors::InvalidArgument("Expected serialized_sparse[", i,
                                  ", 1] to be a shape vector but its shape is ",
                                  output_shape.shape().DebugString()));

      OP_REQUIRES(
          context, DataTypeToEnum<T>::value == output_values.dtype(),
          errors::InvalidArgument(
              "Requested SparseTensor of type ",
              DataTypeString(DataTypeToEnum<T>::value), " but SparseTensor[", i,
              "].values.dtype() == ", DataTypeString(output_values.dtype())));

      int64 num_entries = output_indices.dim_size(0);
      OP_REQUIRES(context, num_entries == output_values.dim_size(0),
                  errors::InvalidArgument(
                      "Expected row counts of SparseTensor[", i,
                      "].indices and SparseTensor[", i,
                      "].values to match but they do not: ", num_entries,
                      " vs. ", output_values.dim_size(0)));
      int rank = output_indices.dim_size(1);
      OP_REQUIRES(
          context, rank == output_shape.dim_size(0),
          errors::InvalidArgument("Expected column counts of SparseTensor[", i,
                                  "].indices to match size of SparseTensor[", i,
                                  "].shape "
                                  "but they do not: ",
                                  rank, " vs. ", output_shape.dim_size(0)));

      // Now we expand each SparseTensors' indices and shape by
      // prefixing a dimension
      Tensor expanded_indices(
          DT_INT64, TensorShape({num_entries, 1 + output_indices.dim_size(1)}));
      Tensor expanded_shape(DT_INT64,
                            TensorShape({1 + output_shape.dim_size(0)}));
      const auto& output_indices_t = output_indices.matrix<int64>();
      const auto& output_shape_t = output_shape.vec<int64>();
      auto expanded_indices_t = expanded_indices.matrix<int64>();
      auto expanded_shape_t = expanded_shape.vec<int64>();
      expanded_indices_t.chip<1>(0).setZero();
      Eigen::DSizes<Eigen::DenseIndex, 2> indices_start(0, 1);
      Eigen::DSizes<Eigen::DenseIndex, 2> indices_sizes(num_entries, rank);
      expanded_indices_t.slice(indices_start, indices_sizes) = output_indices_t;
      expanded_shape_t(0) = 1;
      std::copy_n(&output_shape_t(0), rank, &expanded_shape_t(1));

      TensorShape expanded_tensor_shape(expanded_shape.vec<int64>());

      indices_to_concat.push_back(expanded_indices);
      values_to_concat.push_back(output_values);
      shapes_to_concat.push_back(expanded_tensor_shape);
    }

    int rank = -1;
    for (int i = 0; i < num_sparse_tensors; ++i) {
      if (rank < 0) rank = shapes_to_concat[i].dims();
      OP_REQUIRES(context, rank == shapes_to_concat[i].dims(),
                  errors::InvalidArgument(
                      "Inconsistent rank across SparseTensors: rank prior to "
                      "SparseTensor[",
                      i, "] was: ", rank, " but rank of SparseTensor[", i,
                      "] is: ", shapes_to_concat[i].dims()));
    }

    // SparseTensor::Concat requires consistent shape for all but the
    // primary order dimension (dimension 0 in this case).  So we get
    // the maximum value across all the input SparseTensors for each
    // dimension and use that.
    TensorShape preconcat_shape(shapes_to_concat[0]);
    for (int i = 0; i < num_sparse_tensors; ++i) {
      for (int d = 0; d < rank; ++d) {
        preconcat_shape.set_dim(d, std::max(preconcat_shape.dim_size(d),
                                            shapes_to_concat[i].dim_size(d)));
      }
    }

    // Dimension 0 is the primary dimension.
    gtl::InlinedVector<int64, 8> std_order(rank);
    std::iota(std_order.begin(), std_order.end(), 0);

    std::vector<SparseTensor> tensors_to_concat;
    tensors_to_concat.reserve(num_sparse_tensors);
    for (int i = 0; i < num_sparse_tensors; ++i) {
      tensors_to_concat.emplace_back(indices_to_concat[i], values_to_concat[i],
                                     preconcat_shape, std_order);
    }

    SparseTensor output = SparseTensor::Concat<T>(tensors_to_concat);

    Tensor final_output_shape(DT_INT64, TensorShape({output.dims()}));

    std::copy_n(output.shape().data(), output.dims(),
                final_output_shape.vec<int64>().data());

    context->set_output(0, output.indices());
    context->set_output(1, output.values());
    context->set_output(2, final_output_shape);
  }
};

#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("DeserializeManySparse")       \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          DeserializeManySparseOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
