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

using sparse::SparseTensor;

template <typename T>
class SerializeSparseOp : public OpKernel {
 public:
  explicit SerializeSparseOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  bool IsExpensive() override;

  Status Initialize(Tensor* result);
  Status Serialize(const Tensor& input, T* result);

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

    Tensor serialized_sparse;
    OP_REQUIRES_OK(context, Initialize(&serialized_sparse));

    auto serialized_sparse_t = serialized_sparse.vec<T>();
    OP_REQUIRES_OK(context, Serialize(*input_indices, &serialized_sparse_t(0)));
    OP_REQUIRES_OK(context, Serialize(*input_values, &serialized_sparse_t(1)));
    OP_REQUIRES_OK(context, Serialize(*input_shape, &serialized_sparse_t(2)));

    context->set_output(0, serialized_sparse);
  }
};

// NOTE(mrry): We specialize the IsExpensive() method differently for
// the string and variant cases, because (i) the string version
// actually performs memory copies as part of its serialization (and
// is hence potentially expensive), and (ii) the variant version
// performs O(1) shallow copies (and hence is much cheaper than
// dispatching to another thread would be).
template <>
bool SerializeSparseOp<string>::IsExpensive() {
  return true;
}
template <>
bool SerializeSparseOp<Variant>::IsExpensive() {
  return false;
}

template <>
Status SerializeSparseOp<string>::Initialize(Tensor* result) {
  *result = Tensor(DT_STRING, TensorShape({3}));
  return Status::OK();
}

template <>
Status SerializeSparseOp<string>::Serialize(const Tensor& input,
                                            string* result) {
  TensorProto proto;
  input.AsProtoTensorContent(&proto);
  *result = proto.SerializeAsString();
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("SerializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<string>("out_type"),
                        SerializeSparseOp<string>);

template <>
Status SerializeSparseOp<Variant>::Initialize(Tensor* result) {
  *result = Tensor(DT_VARIANT, TensorShape({3}));
  return Status::OK();
}

template <>
Status SerializeSparseOp<Variant>::Serialize(const Tensor& input,
                                             Variant* result) {
  *result = input;
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("SerializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Variant>("out_type"),
                        SerializeSparseOp<Variant>);

template <typename T>
class SerializeManySparseOpBase : public OpKernel {
 public:
  explicit SerializeManySparseOpBase(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {}

 protected:
  Status Initialize(const int64 n, Tensor* result);
  Status Serialize(const Tensor& input, T* result);
};

template <typename T, typename U>
class SerializeManySparseOp : public SerializeManySparseOpBase<U> {
 public:
  explicit SerializeManySparseOp(OpKernelConstruction* context)
      : SerializeManySparseOpBase<U>(context) {}

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
    Tensor serialized_sparse;
    OP_REQUIRES_OK(context, this->Initialize(N, &serialized_sparse));
    auto serialized_sparse_t = serialized_sparse.matrix<U>();

    OP_REQUIRES_OK(context, input_st.IndicesValid());

    // Initialize output with empty values and the proper shapes.
    Tensor output_blank_indices(DT_INT64, {0, rank - 1});
    U serialized_indices;
    OP_REQUIRES_OK(context,
                   this->Serialize(output_blank_indices, &serialized_indices));
    serialized_sparse_t.template chip<1>(0).setConstant(serialized_indices);

    Tensor output_blank_values(DataTypeToEnum<T>::value, {0});
    U serialized_values;
    OP_REQUIRES_OK(context,
                   this->Serialize(output_blank_values, &serialized_values));
    serialized_sparse_t.template chip<1>(1).setConstant(serialized_values);

    Tensor output_shape(DT_INT64, {rank - 1});
    auto output_shape_t = output_shape.vec<int64>();
    for (int d = 1; d < rank; d++) output_shape_t(d - 1) = input_shape_t(d);
    U serialized_shape;
    OP_REQUIRES_OK(context, this->Serialize(output_shape, &serialized_shape));
    serialized_sparse_t.template chip<1>(2).setConstant(serialized_shape);

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

      OP_REQUIRES_OK(
          context, this->Serialize(output_indices, &serialized_sparse_t(b, 0)));
      OP_REQUIRES_OK(
          context, this->Serialize(output_values, &serialized_sparse_t(b, 1)));
    }

    context->set_output(0, serialized_sparse);
  }
};

template <>
Status SerializeManySparseOpBase<string>::Initialize(const int64 n,
                                                     Tensor* result) {
  *result = Tensor(DT_STRING, TensorShape({n, 3}));
  return Status::OK();
}

template <>
Status SerializeManySparseOpBase<string>::Serialize(const Tensor& input,
                                                    string* result) {
  TensorProto proto;
  input.AsProtoTensorContent(&proto);
  *result = proto.SerializeAsString();
  return Status::OK();
}

#define REGISTER_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("SerializeManySparse")              \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<string>("out_type"), \
                          SerializeManySparseOp<type, string>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <>
Status SerializeManySparseOpBase<Variant>::Initialize(const int64 n,
                                                      Tensor* result) {
  *result = Tensor(DT_VARIANT, TensorShape({n, 3}));
  return Status::OK();
}

template <>
Status SerializeManySparseOpBase<Variant>::Serialize(const Tensor& input,
                                                     Variant* result) {
  *result = input;
  return Status::OK();
}

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("SerializeManySparse")               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<Variant>("out_type"), \
                          SerializeManySparseOp<type, Variant>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename T>
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
      const auto& serialized_sparse_t = serialized_sparse.vec<T>();

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

    const auto& serialized_sparse_t = serialized_sparse.flat_inner_dims<T, 2>();
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

 protected:
  Status Deserialize(const T& serialized, Tensor* result);

  Status GetAndValidateSparseTensor(
      const T& serialized_indices, const T& serialized_values,
      const T& serialized_shape, DataType values_dtype, int index,
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

template <>
Status DeserializeSparseOp<string>::Deserialize(const string& serialized,
                                                Tensor* result) {
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

REGISTER_KERNEL_BUILDER(Name("DeserializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<string>("Tserialized"),
                        DeserializeSparseOp<string>)

REGISTER_KERNEL_BUILDER(Name("DeserializeManySparse").Device(DEVICE_CPU),
                        DeserializeSparseOp<string>)

template <>
Status DeserializeSparseOp<Variant>::Deserialize(const Variant& serialized,
                                                 Tensor* result) {
  *result = *serialized.get<Tensor>();
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("DeserializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Variant>("Tserialized"),
                        DeserializeSparseOp<Variant>)

}  // namespace tensorflow
