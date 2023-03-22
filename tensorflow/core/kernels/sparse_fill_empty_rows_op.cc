/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_empty_rows_functor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename Device, typename T, typename Tindex>
void SparseFillEmptyRowsOpImpl(OpKernelContext* context,
                               AsyncOpKernel::DoneCallback done = nullptr) {
  // Note that setting this empty lambda as the default parameter value directly
  // can cause strange compiler/linker errors, so we do it like this instead.
  if (!done) {
    done = [] {};
  }

  const int kIndicesInput = 0;
  const int kValuesInput = 1;
  const int kDenseShapeInput = 2;
  const int kDefaultValueInput = 3;

  const Tensor& indices_t = context->input(kIndicesInput);
  const Tensor& values_t = context->input(kValuesInput);
  const Tensor& dense_shape_t = context->input(kDenseShapeInput);
  const Tensor& default_value_t = context->input(kDefaultValueInput);

  OP_REQUIRES_ASYNC(
      context, TensorShapeUtils::IsVector(dense_shape_t.shape()),
      errors::InvalidArgument("dense_shape must be a vector, saw: ",
                              dense_shape_t.shape().DebugString()),
      done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsMatrix(indices_t.shape()),
                    errors::InvalidArgument("indices must be a matrix, saw: ",
                                            indices_t.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(values_t.shape()),
                    errors::InvalidArgument("values must be a vector, saw: ",
                                            values_t.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(
      context, indices_t.dim_size(0) == values_t.dim_size(0),
      errors::InvalidArgument("The length of `values` (", values_t.dim_size(0),
                              ") must match the first dimension of `indices` (",
                              indices_t.dim_size(0), ")."),
      done);
  OP_REQUIRES_ASYNC(
      context, indices_t.dim_size(1) == dense_shape_t.dim_size(0),
      errors::InvalidArgument("The length of `dense_shape` (",
                              dense_shape_t.dim_size(0),
                              ") must match the second dimension of `indices` ",
                              "(", indices_t.dim_size(1), ")."),
      done);
  OP_REQUIRES_ASYNC(
      context, TensorShapeUtils::IsScalar(default_value_t.shape()),
      errors::InvalidArgument("default_value must be a scalar, saw: ",
                              default_value_t.shape().DebugString()),
      done);
  // TODO(ebrevdo): add shape checks between values, indices,
  // Also add check that dense rank > 0.
  OP_REQUIRES_ASYNC(context, dense_shape_t.NumElements() != 0,
                    errors::InvalidArgument("Dense shape cannot be empty."),
                    done);

  using FunctorType =
      functor::FillEmptyRows<Device, T, Tindex, /*RaggedOperands=*/false>;
  OP_REQUIRES_OK_ASYNC(context,
                       FunctorType()(context, default_value_t, indices_t,
                                     values_t, dense_shape_t, done),
                       done);
}

}  // namespace

template <typename Device, typename T, typename Tindex>
class SparseFillEmptyRowsOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    SparseFillEmptyRowsOpImpl<Device, T, Tindex>(context);
  }
};

#define REGISTER_KERNELS(D, T, Tindex)                   \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRows")    \
                              .Device(DEVICE_##D)        \
                              .HostMemory("dense_shape") \
                              .TypeConstraint<T>("T"),   \
                          SparseFillEmptyRowsOp<D##Device, T, Tindex>)

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T, int64)
TF_CALL_ALL_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// The GPU implementation is async because it requires waiting for a
// host->device memcpy before the output is allocated (similar to
// SegmentSumGPUOp).
template <typename T, typename Tindex>
class SparseFillEmptyRowsGPUOp : public AsyncOpKernel {
 public:
  explicit SparseFillEmptyRowsGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    SparseFillEmptyRowsOpImpl<GPUDevice, T, Tindex>(context, done);
  }
};

#define REGISTER_KERNELS(T, Tindex)                      \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRows")    \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("dense_shape") \
                              .TypeConstraint<T>("T"),   \
                          SparseFillEmptyRowsGPUOp<T, Tindex>)


#define REGISTER_KERNELS_TINDEX(T) REGISTER_KERNELS(T, int64)
TF_CALL_POD_TYPES(REGISTER_KERNELS_TINDEX)
#undef REGISTER_KERNELS_TINDEX

#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


template <typename Device, typename T, typename Tindex>
class SparseFillEmptyRowsGradOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* reverse_index_map_t;
    const Tensor* grad_values_t;
    OP_REQUIRES_OK(context,
                   context->input("reverse_index_map", &reverse_index_map_t));
    OP_REQUIRES_OK(context, context->input("grad_values", &grad_values_t));

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(reverse_index_map_t->shape()),
        errors::InvalidArgument("reverse_index_map must be a vector, saw: ",
                                reverse_index_map_t->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(grad_values_t->shape()),
                errors::InvalidArgument("grad_values must be a vector, saw: ",
                                        grad_values_t->shape().DebugString()));

    const auto reverse_index_map = reverse_index_map_t->vec<Tindex>();
    const auto grad_values = grad_values_t->vec<T>();

    const Tindex N = reverse_index_map_t->shape().dim_size(0);

    Tensor* d_values_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "d_values", TensorShape({N}), &d_values_t));
    auto d_values = d_values_t->vec<T>();
    Tensor* d_default_value_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output("d_default_value", TensorShape({}),
                                            &d_default_value_t));
    auto d_default_value = d_default_value_t->scalar<T>();

    OP_REQUIRES_OK(context, functor::FillEmptyRowsGrad<Device, T, Tindex>()(
                                context, reverse_index_map, grad_values,
                                d_values, d_default_value));
  }
};

#define REGISTER_KERNELS(D, T, Tindex)                    \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRowsGrad") \
                              .Device(DEVICE_##D)         \
                              .TypeConstraint<T>("T"),    \
                          SparseFillEmptyRowsGradOp<D##Device, T, Tindex>)

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T, int64)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T, int64)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_KERNELS
}  // namespace tensorflow
