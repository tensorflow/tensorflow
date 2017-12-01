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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using thread::ThreadPool;

template <typename T>
class BincountOp : public OpKernel {
 public:
  explicit BincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& arr_t = ctx->input(0);
    const Tensor& size_tensor = ctx->input(1);
    const Tensor& weights_t = ctx->input(2);
    int32 size = size_tensor.scalar<int32>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));
    const bool has_weights = weights_t.NumElements() > 0;
    OP_REQUIRES(ctx, !(has_weights && arr_t.shape() != weights_t.shape()),
                errors::InvalidArgument(
                    "If weights are passed, they must have the same shape (" +
                    weights_t.shape().DebugString() + ") as arr (" +
                    arr_t.shape().DebugString() + ")"));
    const auto arr = arr_t.flat<int32>();
    const auto weights = weights_t.flat<T>();

    Tensor all_nonneg_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_BOOL, TensorShape({}), &all_nonneg_t,
                                      AllocatorAttributes()));
    all_nonneg_t.scalar<bool>().device(ctx->eigen_cpu_device()) =
        (arr >= 0).all();
    OP_REQUIRES(ctx, all_nonneg_t.scalar<bool>()(),
                errors::InvalidArgument("Input arr must be non-negative!"));

    // Allocate partial output bin sums for each worker thread. Worker ids in
    // ParallelForWithWorkerId range from 0 to NumThreads() inclusive.
    ThreadPool* thread_pool =
        ctx->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 num_threads = thread_pool->NumThreads() + 1;
    Tensor partial_bins_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(weights_t.dtype(),
                                           TensorShape({num_threads, size}),
                                           &partial_bins_t));
    auto partial_bins = partial_bins_t.matrix<T>();
    partial_bins.setZero();
    thread_pool->ParallelForWithWorkerId(
        arr.size(), 8 /* cost */,
        [&](int64 start_ind, int64 limit_ind, int64 worker_id) {
          for (int64 i = start_ind; i < limit_ind; i++) {
            int32 value = arr(i);
            if (value < size) {
              if (has_weights) {
                partial_bins(worker_id, value) += weights(i);
              } else {
                // Complex numbers don't support "++".
                partial_bins(worker_id, value) += T(1);
              }
            }
          }
        });
    TensorShape output_shape({size});
    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_t));
    // Sum the partial bins along the 0th axis.
    Eigen::array<int, 1> reduce_dims({0});
    output_t->flat<T>().device(ctx->eigen_cpu_device()) =
        partial_bins.sum(reduce_dims);
  }
};

#define REGISTER(TYPE)                                               \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Bincount").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      BincountOp<TYPE>)

TF_CALL_NUMBER_TYPES(REGISTER);

// TODO(ringwalt): Add a GPU implementation. We probably want to take a
// different approach, e.g. threads in a warp each taking a pass over the same
// data, and each thread summing a single bin.

}  // end namespace tensorflow
