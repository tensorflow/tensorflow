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

// See docs in ../ops/nn_ops.cc.
#include "tensorflow/core/kernels/nth_element_op.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class NthElementOp : public OpKernel {
 public:
  explicit NthElementOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("reverse", &reverse_));
  }

  void Compute(OpKernelContext* context) override {
    // The second args is N, which must be a positive scalar.
    const auto& n_in = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(n_in.shape()),
        errors::InvalidArgument("N must be scalar but has rank ", n_in.dims()));
    int n = n_in.scalar<int32>()();
    OP_REQUIRES(context, n >= 0,
                errors::InvalidArgument("n must be non-negative but is ", n));

    // The first args is input tensor, which must have 1 dimension at least.
    const Tensor& input_in = context->input(0);
    const int num_dims = input_in.dims();
    OP_REQUIRES(context, num_dims >= 1,
                errors::InvalidArgument(
                    "Input must be at least rank 1 but is rank ", num_dims));
    // The last dimension of input tensor must be greater than N.
    OP_REQUIRES(
        context, input_in.dim_size(num_dims - 1) > n,
        errors::InvalidArgument("Input must have last dimension > n = ", n));

    // std::nth_element only support the nth-smallest selection.
    if (reverse_) {
      n = input_in.dim_size(num_dims - 1) - n - 1;
    }

    // Assume input_shape is [d1,d2,...dk], and output_shape is [d1,d2...dk-1].
    TensorShape out_shape;
    for (int i = 0; i < num_dims - 1; ++i) {
      OP_REQUIRES_OK(context, out_shape.AddDimWithStatus(input_in.dim_size(i)));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));

    functor::NthElementFunctor<Device, T> nthElementFunc;
    nthElementFunc(context, input_in, *output_tensor, n, reverse_);
  }

 private:
  bool reverse_;
};

namespace functor {

template <typename T>
struct NthElementFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_tensor,
                  Tensor& output_tensor, int n, bool reverse) {
    const T* input = input_tensor.flat<T>().data();
    T* output = output_tensor.flat<T>().data();

    // Assume input_shape is [d1,d2,...dk], and output_shape is [d1,d2...dk-1],
    // then num_rows = d1*d2...dk-1, last_dim = dk.
    const int num_rows = output_tensor.NumElements();
    const int last_dim = input_tensor.dim_size(input_tensor.dims() - 1);

    // Allocate each row to different shard.
    auto SubNthElement = [&, input, output, last_dim, n](int64_t start,
                                                         int64_t limit) {
      // std::nth_element would rearrange the array, so we need a new buffer.
      std::vector<T> buf(last_dim);

      for (int b = start; b < limit; ++b) {
        // Copy from one row of elements to buffer
        const T* input_start = input + b * last_dim;
        const T* input_end = input + (b + 1) * last_dim;
        std::copy(input_start, input_end, buf.begin());

        std::nth_element(buf.begin(), buf.begin() + n, buf.end());
        // The element placed in the nth position is exactly the element that
        // would occur in this position if the range was fully sorted.
        output[b] = buf[n];
      }
    };

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    // The average time complexity of partition-based nth_element (BFPRT) is
    // O(n), although the worst time complexity could be O(n^2). Here, 20 is a
    // empirical factor of cost_per_unit.
    Shard(worker_threads.num_threads, worker_threads.workers, num_rows,
          20 * last_dim, SubNthElement);
  }
};

}  // namespace functor

#define REGISTER_NTHOP(T)                                           \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("NthElement").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      NthElementOp<CPUDevice, T>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_NTHOP);
#undef REGISTER_NTHOP

}  // end namespace tensorflow
