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

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

template <typename T>
class SparseSliceOp : public OpKernel {
 public:
  explicit SparseSliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_indices = context->input(0);
    const Tensor& input_values = context->input(1);
    const Tensor& input_shape = context->input(2);
    const Tensor& input_start = context->input(3);
    const Tensor& input_size = context->input(4);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values.shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_indices.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape.shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_start.shape()),
                errors::InvalidArgument(
                    "Input start should be a vector but received shape ",
                    input_start.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_size.shape()),
                errors::InvalidArgument(
                    "Input size should be a vector but received shape ",
                    input_size.shape().DebugString()));

    const int input_dims = input_shape.NumElements();
    OP_REQUIRES(context, input_dims == input_start.NumElements(),
                errors::InvalidArgument(
                    "Expected start to be a vector of length ", input_dims,
                    " but got length ", input_start.NumElements()));

    OP_REQUIRES(context, input_dims == input_size.NumElements(),
                errors::InvalidArgument(
                    "Expected size to be a vector of length ", input_dims,
                    " but got length ", input_size.NumElements()));

    sparse::SparseTensor sparse_tensor(input_indices, input_values,
                                       TensorShape(input_shape.vec<int64>()));

    const gtl::ArraySlice<int64> start(input_start.flat<int64>().data(),
                                       input_dims);
    const gtl::ArraySlice<int64> size(input_size.flat<int64>().data(),
                                      input_dims);

    const sparse::SparseTensor output =
        sparse::SparseTensor::Slice<T>(sparse_tensor, start, size);

    context->set_output(0, output.indices());
    context->set_output(1, output.values());

    const TensorShape output_shape(output.shape());

    Tensor* shape = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, {output_shape.dims()}, &shape));
    for (int dim = 0; dim < output_shape.dims(); ++dim) {
      shape->vec<int64>()(dim) = output_shape.dim_size(dim);
    }
  }

 private:
};

#define REGISTER_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSlice").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSliceOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
