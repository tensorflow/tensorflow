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

#ifndef TENSORFLOW_KERNELS_ASSIGN_OP_H_
#define TENSORFLOW_KERNELS_ASSIGN_OP_H_

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

// TODO(jeff): Get rid of use_exclusive_lock_ option

// Computes *input[0] = input[1]
class AssignOp : public OpKernel {
 public:
  explicit AssignOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_shape", &validate_shape_));
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& rhs = context->input(1);

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    // We can't always know how this value will be used downstream,
    // so make conservative assumptions in specifying constraints on
    // the memory allocation attributes.
    // TODO(rmlarsen): These conservative constraints make buffer
    // forwarding unlikely to happen very often. Try to use graph analysis
    // (possibly the InferAllocAttr pass in the executer) to improve the
    // situation.
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);

    {
      mutex_lock l(*context->input_ref_mutex(0));
      const Tensor& old_lhs = context->mutable_input(0, /* lock_held */ true);
      const bool same_shape = old_lhs.shape().IsSameSize(rhs.shape());
      if (validate_shape_) {
        OP_REQUIRES(
            context, same_shape,
            errors::InvalidArgument(
                "Assign requires shapes of both tensors to match. lhs shape= ",
                old_lhs.shape().DebugString(),
                " rhs shape= ", rhs.shape().DebugString()));
      }

      // In the code below we try to minimize the amount of memory allocation
      // and copying by trying the following two shortcuts:
      // 1. If we can reuse the rhs buffer we avoid both a memory allocation
      //   and copying.
      // 2. If the lhs is initialized and has the same number of elements as the
      //    rhs we can avoid a memory allocation.

      // 1. Try to reuse the rhs.
      std::unique_ptr<Tensor> input_alias = context->forward_input(
          1, OpKernelContext::Params::kNoReservation /*output_index*/,
          old_lhs.dtype(), old_lhs.shape(), DEVICE_MEMORY, attr);
      if (input_alias != nullptr) {
        // Transfer ownership to the ref.
        context->replace_ref_input(0, *input_alias.release(),
                                   /* lock_held */ true);
        return;
      }

      // 2. Try to copy into an existing buffer.
      if (old_lhs.IsInitialized() &&
          old_lhs.shape().num_elements() == rhs.shape().num_elements()) {
        // The existing lhs tensor has already been initialized and the right
        // hand side can fit in the underlying buffer.
        Tensor reshaped_old_lhs;
        if (same_shape) {
          reshaped_old_lhs = old_lhs;
        } else {
          CHECK(reshaped_old_lhs.CopyFrom(old_lhs, rhs.shape()));
          context->replace_ref_input(0, reshaped_old_lhs, /* lock_held */ true);
        }
        if (use_exclusive_lock_) {
          Copy(context, &reshaped_old_lhs, rhs);
          return;
        }
      } else {
        // Create a new persistent tensor whose shape matches the right hand
        // side, hand off to lhs and copy the rhs into it.
        PersistentTensor copy;
        Tensor* copyTensor = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_persistent(old_lhs.dtype(), rhs.shape(),
                                                  &copy, &copyTensor, attr));
        // We track memory of variables in variable ops instead of in this
        // assign op.
        context->clear_recorded_memory();
        context->replace_ref_input(0, *copyTensor, /* lock_held */ true);
        if (use_exclusive_lock_) {
          Copy(context, copyTensor, rhs);
          return;
        }
      }
    }

    // The tensor has already been initialized and the right hand side
    // matches the left hand side's shape. We have been told to do the
    // copy outside the lock.
    Tensor old_unlocked_lhs = context->mutable_input(0, /* lock_held */ false);
    Copy(context, &old_unlocked_lhs, rhs);
  }

  virtual void Copy(OpKernelContext* context, Tensor* lhs,
                    const Tensor& rhs) = 0;

  bool use_exclusive_lock_;
  bool validate_shape_;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ASSIGN_OP_H_
