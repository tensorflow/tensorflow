/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
void AssignRefVariable(
    OpKernelContext* context, int input_ref_index, int output_ref_index,
    int value_index, bool use_locking, bool validate_shape,
    bool relax_constraints,
    std::function<void(OpKernelContext*, Tensor*, const Tensor&)> copy) {
  const Tensor& rhs = context->input(value_index);

  // We always return the input ref.
  context->forward_ref_input_to_ref_output(input_ref_index, output_ref_index);

  // Prevent copying uninitialized data, to solve harder to debug undefined
  // behaviors that cannot be traced back to the original tensor.
  OP_REQUIRES(
      context, rhs.IsInitialized(),
      errors::Internal("Right hand side of AssignOp is not initialized"));

  // We can't always know how this value will be used downstream, so make
  // conservative assumptions in specifying constraints on the memory
  // allocation attributes, unless the Grappler graph analysis determined that
  // it was safe not to.
  AllocatorAttributes attr;
  if (!relax_constraints) {
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
  }

  {
    mutex_lock l(*context->input_ref_mutex(input_ref_index));
    const Tensor& old_lhs =
        context->mutable_input(input_ref_index, /* lock_held */ true);
    const bool same_shape = old_lhs.shape().IsSameSize(rhs.shape());
    if (validate_shape) {
      OP_REQUIRES(context, same_shape,
                  errors::InvalidArgument(
                      "Assign requires shapes of both tensors to match. "
                      "lhs shape= ",
                      old_lhs.shape().DebugString(),
                      " rhs shape= ", rhs.shape().DebugString()));
    }

    // In the code below we try to minimize the amount of memory allocation
    // and copying by trying the following two shortcuts:
    // 1. If the lhs is initialized and has the same number of elements as
    //    the rhs we can avoid a memory allocation.
    // 2. If we can reuse the rhs buffer we avoid both a memory allocation
    //    and copying.

    // 1. Try to copy into an existing buffer.
    if (old_lhs.IsInitialized() &&
        old_lhs.shape().num_elements() == rhs.shape().num_elements()) {
      // The existing lhs tensor has already been initialized and the right
      // hand side can fit in the underlying buffer.
      Tensor reshaped_old_lhs;
      if (same_shape) {
        reshaped_old_lhs = old_lhs;
      } else {
        OP_REQUIRES(context, reshaped_old_lhs.CopyFrom(old_lhs, rhs.shape()),
                    errors::Internal(
                        "Unable to copy the value tensor to the ref input"));
        context->replace_ref_input(input_ref_index, reshaped_old_lhs,
                                   /* lock_held */ true);
      }
      if (use_locking) {
        copy(context, &reshaped_old_lhs, rhs);
        return;
      }
    } else {
      // 2. Try to reuse the rhs.
      std::unique_ptr<Tensor> input_alias = context->forward_input(
          value_index, OpKernelContext::Params::kNoReservation /*output_index*/,
          rhs.dtype(), rhs.shape(), DEVICE_MEMORY, attr);
      if (input_alias != nullptr) {
        // Update the ref to point to the new buffer.
        context->replace_ref_input(input_ref_index, *input_alias,
                                   /* lock_held */ true);
        return;
      }

      // Otherwise, create a new tensor whose shape matches the
      // right hand side, hand off to lhs and copy the rhs into it.
      Tensor copy_tensor;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(old_lhs.dtype(), rhs.shape(),
                                            &copy_tensor, attr));
      // We track memory of variables in variable ops instead of in this
      // assign op.
      context->clear_recorded_memory();
      context->replace_ref_input(input_ref_index, copy_tensor,
                                 /* lock_held */ true);
      if (use_locking) {
        copy(context, &copy_tensor, rhs);
        return;
      }
    }
  }

  // The tensor has already been initialized and the right hand side
  // matches the left hand side's shape. We have been told to do the
  // copy outside the lock.
  Tensor old_unlocked_lhs =
      context->mutable_input(input_ref_index, /* lock_held */ false);
  copy(context, &old_unlocked_lhs, rhs);
}
}  //  end namespace tensorflow
