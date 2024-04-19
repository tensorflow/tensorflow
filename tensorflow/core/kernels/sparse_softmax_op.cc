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

// See docs in ../ops/sparse_ops.cc.

#define EIGEN_USE_THREADS

#include <numeric>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

using tensorflow::gtl::ArraySlice;
using tensorflow::sparse::SparseTensor;

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T>
class SparseSoftmaxOp : public OpKernel {
 public:
  explicit SparseSoftmaxOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor *indices_t, *values_t, *shape_t;
    OP_REQUIRES_OK(context, context->input("sp_indices", &indices_t));
    OP_REQUIRES_OK(context, context->input("sp_values", &values_t));
    OP_REQUIRES_OK(context, context->input("sp_shape", &shape_t));

    // Validations.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_t->shape()),
                errors::InvalidArgument(
                    "Input sp_indices should be a matrix but received shape: ",
                    indices_t->shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(values_t->shape()) &&
                    TensorShapeUtils::IsVector(shape_t->shape()),
                errors::InvalidArgument(
                    "Inputs sp_values and sp_shape should be vectors "
                    "but received shapes: ",
                    values_t->shape().DebugString(), " and ",
                    shape_t->shape().DebugString()));
    OP_REQUIRES(context, shape_t->NumElements() >= 2,
                errors::InvalidArgument(
                    "Input should have rank >= 2, but received shape: ",
                    shape_t->SummarizeValue(3)));
    TensorShape shape;
    OP_REQUIRES_OK(context, TensorShape::BuildTensorShape(
                                shape_t->flat<int64_t>(), &shape));

    const int64_t nnz = indices_t->dim_size(0);
    const int rank = static_cast<int>(indices_t->dim_size(1));
    SparseTensor st;
    OP_REQUIRES_OK(
        context, SparseTensor::Create(tensor::DeepCopy(*indices_t),
                                      tensor::DeepCopy(*values_t), shape, &st));

    Tensor *output_values = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({nnz}),
                                                     &output_values));
    typename TTypes<T>::Flat output_flat = output_values->flat<T>();

    Tensor tmp_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   TensorShape({}), &tmp_t));
    typename TTypes<T>::Scalar tmp_scalar = tmp_t.scalar<T>();

    gtl::InlinedVector<int64_t, 4> dims(rank);
    std::iota(dims.begin(), dims.end(), 0);
    // { 0, ..., rank-1 }.
    const absl::Span<const int64_t> kReorderDims(dims);
    // All but the last dim -- the class dimension to be max-reduced along.
    const absl::Span<const int64_t> kGroupByDims =
        kReorderDims.subspan(0, rank - 1);
    st.Reorder<T>(kReorderDims);
    int count = 0;

    // The SparseTensor has logical shape [..., b, c], where the
    // innermost size-"c" dimension is the class dimension to be max-reduced.
    // Therefore we group by the first (rank - 1) dimensions.
    const Device &device = context->eigen_device<Device>();
    for (const auto &g : st.group(kGroupByDims)) {
      const auto group_vals = g.values<T>();
      const int group_size = group_vals.size();

      // Shifts by max, exponentiates, then renormalizes.
      tmp_scalar.device(context->eigen_device<Device>()) = group_vals.maximum();
      const T group_max = tmp_scalar();

      Eigen::Tensor<T, 1, Eigen::RowMajor> tmp(group_size);
      tmp.device(device) = (group_vals - tmp.constant(group_max)).exp();

      tmp_scalar.device(device) = tmp.sum().inverse();
      tmp.device(device) = tmp * tmp.constant(tmp_scalar());

      // Assigns back to output[count, count + group_size).
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> output_part(
          output_flat.data() + count, group_size);
      output_part.device(device) = tmp;

      count += group_size;
    }
  }
};

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("SparseSoftmax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseSoftmaxOp<CPUDevice, T>)

REGISTER_KERNEL(Eigen::half);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

}  // namespace tensorflow
