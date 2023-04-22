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
class SparseSplitOp : public OpKernel {
 public:
  explicit SparseSplitOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_split", &num_split_));
  }

  void Compute(OpKernelContext* context) override {
    const int64 axis_input = context->input(0).scalar<int64>()();
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

    const int64 input_rank = input_shape.vec<int64>().size();
    const int64 axis = (axis_input < 0) ? input_rank + axis_input : axis_input;

    OP_REQUIRES(
        context, axis >= 0 && axis < input_rank,
        errors::InvalidArgument("Input axis should be in range [", -input_rank,
                                ", ", input_rank, "), got ", axis_input));

    OP_REQUIRES(context,
                num_split_ >= 1 && num_split_ <= input_shape.vec<int64>()(axis),
                errors::InvalidArgument("Input num_split should be between 1 "
                                        "and the splitting dimension size (",
                                        input_shape.vec<int64>()(axis),
                                        "), got ", num_split_));

    // Prevent overflow by constructing the dense shape separately
    TensorShape dense_shape;
    const auto input_shape_flat = input_shape.flat<int64>();
    for (int i = 0; i < input_shape.NumElements(); i++) {
      OP_REQUIRES_OK(context,
                     dense_shape.AddDimWithStatus(input_shape_flat(i)));
    }

    sparse::SparseTensor sparse_tensor;
    OP_REQUIRES_OK(context,
                   sparse::SparseTensor::Create(input_indices, input_values,
                                                dense_shape, &sparse_tensor));

    std::vector<sparse::SparseTensor> outputs;
    OP_REQUIRES_OK(context, sparse::SparseTensor::Split<T>(
                                sparse_tensor, axis, num_split_, &outputs));

    for (int slice_index = 0; slice_index < num_split_; ++slice_index) {
      context->set_output(slice_index, outputs[slice_index].indices());
      context->set_output(slice_index + num_split_,
                          outputs[slice_index].values());
      Tensor* shape = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  slice_index + 2 * num_split_,
                                  {outputs[slice_index].dims()}, &shape));
      auto output_shape = outputs[slice_index].shape();
      for (int dim = 0; dim < outputs[slice_index].dims(); ++dim) {
        shape->vec<int64>()(dim) = output_shape[dim];
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
