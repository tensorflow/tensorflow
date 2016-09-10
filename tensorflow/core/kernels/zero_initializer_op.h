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

#ifndef TENSORFLOW_KERNELS_ZERO_INITIALIZER_OP_H_
#define TENSORFLOW_KERNELS_ZERO_INITIALIZER_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

template <typename T>
class ZeroInitializerOp : public OpKernel {
  public:
    explicit ZeroInitializerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
      OP_REQUIRES(ctx, IsRefType(ctx->input_type(0)),
          errors::InvalidArgument("input need to be a ref type"));
    }
    void Compute(OpKernelContext* ctx) override {
      if (use_exclusive_lock_) {
        mutex_lock l(*ctx->input_ref_mutex(0));
        DoCompute(ctx);
      } else {
        DoCompute(ctx);
      }
    }
  private:
    bool use_exclusive_lock_;

    void DoCompute(OpKernelContext* ctx) {
      Tensor input = ctx->mutable_input(0, use_exclusive_lock_);
      if (!input.IsInitialized()) {
        AllocatorAttributes attr;
        attr.set_gpu_compatible(true);
        attr.set_nic_compatible(true);
        PersistentTensor out_persistent;
        Tensor* out_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_persistent(
              input.dtype(), input.shape(), &out_persistent,
              &out_tensor, attr));
        auto out_tensor_flat = out_tensor->flat<T>();
        int total_size = static_cast<int>(1);
        for (int d = static_cast<int>(0); d < out_tensor->dims(); d++) {
          total_size *= out_tensor->dim_size(d);
        }
        for (int idx = static_cast<int>(0); idx < total_size; idx++) {
          out_tensor_flat(idx) = static_cast<T>(0);
        }
        ctx->replace_ref_input(0, *out_tensor, use_exclusive_lock_);
      }
      // we always return the input ref.
      ctx->forward_ref_input_to_ref_output(0, 0);
    }
};

} // end namespace tensorflow
#endif // TENSORFLOW_KERNELS_ZERO_INITIALIZER_OP_H_
