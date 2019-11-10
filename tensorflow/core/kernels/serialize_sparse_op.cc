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

namespace {

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
bool SerializeSparseOp<tstring>::IsExpensive() {
  return true;
}
template <>
bool SerializeSparseOp<Variant>::IsExpensive() {
  return false;
}

template <>
Status SerializeSparseOp<tstring>::Initialize(Tensor* result) {
  *result = Tensor(DT_STRING, TensorShape({3}));
  return Status::OK();
}

template <>
Status SerializeSparseOp<tstring>::Serialize(const Tensor& input,
                                             tstring* result) {
  TensorProto proto;
  input.AsProtoTensorContent(&proto);
  *result = proto.SerializeAsString();
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("SerializeSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tstring>("out_type"),
                        SerializeSparseOp<tstring>);

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
    SparseTensor input_st;
    OP_REQUIRES_OK(context, SparseTensor::Create(*input_indices, *input_values,
                                                 tensor_input_shape, std_order,
                                                 &input_st));

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
Status SerializeManySparseOpBase<tstring>::Initialize(const int64 n,
                                                      Tensor* result) {
  *result = Tensor(DT_STRING, TensorShape({n, 3}));
  return Status::OK();
}

template <>
Status SerializeManySparseOpBase<tstring>::Serialize(const Tensor& input,
                                                     tstring* result) {
  TensorProto proto;
  input.AsProtoTensorContent(&proto);
  *result = proto.SerializeAsString();
  return Status::OK();
}

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("SerializeManySparse")               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<tstring>("out_type"), \
                          SerializeManySparseOp<type, tstring>)

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

}  // namespace

}  // namespace tensorflow
