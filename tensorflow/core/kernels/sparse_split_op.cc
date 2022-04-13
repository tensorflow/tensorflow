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

#include "tensorflow/core/kernels/sparse_split_op.h"

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename T>
struct SparseSplitFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const TensorShape& dense_shape,
                  const int64_t axis, const int num_split,
                  typename AsyncOpKernel::DoneCallback done) {
    (void)done;  // Unused (only used in GPU implementation)
    sparse::SparseTensor sparse_tensor;
    OP_REQUIRES_OK(context,
                   sparse::SparseTensor::Create(input_indices, input_values,
                                                dense_shape, &sparse_tensor));

    std::vector<sparse::SparseTensor> outputs;
    OP_REQUIRES_OK(context, sparse::SparseTensor::Split<T>(
                                sparse_tensor, axis, num_split, &outputs));

    for (int slice_index = 0; slice_index < num_split; ++slice_index) {
      context->set_output(slice_index, outputs[slice_index].indices());
      context->set_output(slice_index + num_split,
                          outputs[slice_index].values());
      Tensor* shape = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  slice_index + 2 * num_split,
                                  {outputs[slice_index].dims()}, &shape));
      auto output_shape = outputs[slice_index].shape();
      for (int dim = 0; dim < outputs[slice_index].dims(); ++dim) {
        shape->vec<int64_t>()(dim) = output_shape[dim];
      }
    }
  }
};

}  // namespace functor

namespace {

template <typename Device, typename T>
void SparseSplitOpImpl(OpKernelContext* context, int num_split,
                       AsyncOpKernel::DoneCallback done = nullptr) {
  // Note that setting this empty lambda as the default parameter value directly
  // can cause strange compiler/linker errors, so we do it like this instead.
  if (!done) {
    done = [] {};
  }

  const Tensor& input_axis = context->input(0);
  const Tensor& input_indices = context->input(1);
  const Tensor& input_values = context->input(2);
  const Tensor& input_shape = context->input(3);

  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsScalar(input_axis.shape()),
                    errors::InvalidArgument(
                        "Input axis should be a scalar but received shape ",
                        input_axis.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
                    errors::InvalidArgument(
                        "Input indices should be a matrix but received shape ",
                        input_indices.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_values.shape()),
                    errors::InvalidArgument(
                        "Input values should be a vector but received shape ",
                        input_indices.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_shape.shape()),
                    errors::InvalidArgument(
                        "Input shape should be a vector but received shape ",
                        input_shape.shape().DebugString()),
                    done);

  const int64_t axis_input = input_axis.scalar<int64_t>()();
  const int64_t input_rank = input_shape.vec<int64_t>().size();
  const int64_t axis = (axis_input < 0) ? input_rank + axis_input : axis_input;

  OP_REQUIRES_ASYNC(
      context, axis >= 0 && axis < input_rank,
      errors::InvalidArgument("Input axis should be in range [", -input_rank,
                              ", ", input_rank, "), got ", axis_input),
      done);

  OP_REQUIRES_ASYNC(
      context, num_split >= 1 && num_split <= input_shape.vec<int64_t>()(axis),
      errors::InvalidArgument("Input num_split should be between 1 "
                              "and the splitting dimension size (",
                              input_shape.vec<int64_t>()(axis), "), got ",
                              num_split),
      done);

  // Prevent overflow by constructing the dense shape separately
  TensorShape dense_shape;
  const auto input_shape_flat = input_shape.flat<int64_t>();
  for (int i = 0; i < input_shape.NumElements(); i++) {
    OP_REQUIRES_OK_ASYNC(
        context, dense_shape.AddDimWithStatus(input_shape_flat(i)), done);
  }

  functor::SparseSplitFunctor<Device, T>()(context, input_indices, input_values,
                                           dense_shape, axis, num_split, done);
}

}  // namespace

template <typename T>
class SparseSplitOp : public OpKernel {
 public:
  explicit SparseSplitOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_split", &num_split_));
  }

  void Compute(OpKernelContext* context) override {
    SparseSplitOpImpl<CPUDevice, T>(context, num_split_);
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

// The GPU implementation is async because it requires waiting for a
// host->device memcpy before the output is allocated (similar to
// SegmentSumGPUOp).
template <typename T>
class SparseSplitGPUOp : public AsyncOpKernel {
 public:
  explicit SparseSplitGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_split", &num_split_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    SparseSplitOpImpl<GPUDevice, T>(context, num_split_, done);
  }

 private:
  int num_split_;
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseSplit")             \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("split_dim")    \
                              .HostMemory("shape")        \
                              .HostMemory("output_shape") \
                              .TypeConstraint<type>("T"), \
                          SparseSplitGPUOp<type>)
TF_CALL_POD_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
