/* Copyright 2015 Google Inc. All Rights Reserved.

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
class SparseSplitOp : public OpKernel {
 public:
  explicit SparseSplitOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_split", &num_split_));
  }

  void Compute(OpKernelContext* context) override {
    const int32 split_dim = context->input(0).scalar<int>()();
    const Tensor& input_indices = context->input(1);
    const Tensor& input_values = context->input(2);
    const Tensor& input_shape = context->input(3);

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

    OP_REQUIRES(context, input_shape.dim_size(0) &&
                             split_dim < input_shape.vec<int64>().size(),
                errors::InvalidArgument(
                    "Input split_dim should be between 0 and rank (",
                    input_shape.vec<int64>().size(), "), got ", split_dim));

    OP_REQUIRES(context, num_split_ >= 1 &&
                             num_split_ <= input_shape.vec<int64>()(split_dim),
                errors::InvalidArgument("Input num_split should be between 1 "
                                        "and the splitting dimension size (",
                                        input_shape.vec<int64>()(split_dim),
                                        "), got ", num_split_));

    sparse::SparseTensor sparse_tensor(input_indices, input_values,
                                       TensorShape(input_shape.vec<int64>()));
    const std::vector<sparse::SparseTensor> outputs =
        sparse::SparseTensor::Split<T>(sparse_tensor, split_dim, num_split_);

    for (int slice_index = 0; slice_index < num_split_; ++slice_index) {
      context->set_output(slice_index, outputs[slice_index].indices());
      context->set_output(slice_index + num_split_,
                          outputs[slice_index].values());
      Tensor* shape = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(
                         slice_index + 2 * num_split_,
                         {outputs[slice_index].shape().dims()}, &shape));
      for (int dim = 0; dim < outputs[slice_index].shape().dims(); ++dim) {
        shape->vec<int64>()(dim) = outputs[slice_index].shape().dim_size(dim);
      }
    }
  }

 private:
  int num_split_;
};

#define REGISTER_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSplit").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSplitOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
