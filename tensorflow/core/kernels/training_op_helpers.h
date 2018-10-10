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

#ifndef TENSORFLOW_CORE_KERNELS_TRAINING_OP_HELPERS_H_
#define TENSORFLOW_CORE_KERNELS_TRAINING_OP_HELPERS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/variable_ops.h"

namespace tensorflow {

// Returns a borrowed pointer to the mutex for the variable `input` in `ctx`.
//
// If `input` corresponds to a `DT_RESOURCE`-type variable input,
// `*maybe_resource` will be updated to contain the underlying resource, and the
// caller will be responsible for calling `Unref()` on that resource.
mutex* GetTrainingVariableMutex(OpKernelContext* ctx, int input,
                                Var** maybe_resource);

// Utility structure that releases a sequence of borrowed mutexes when it is
// deleted.
struct VariableInputLockHolder {
 public:
  VariableInputLockHolder(std::vector<Var*> vars,
                          std::unique_ptr<std::vector<mutex_lock>> locks)
      : vars_(std::move(vars)), locks_(std::move(locks)) {}

  VariableInputLockHolder(VariableInputLockHolder&& other)
      : vars_(std::move(other.vars_)), locks_(std::move(other.locks_)) {}

  ~VariableInputLockHolder() {
    // Release the locks before unreffing the Vars, because each lock
    // is potentially borrowed from a Var in vars_.
    locks_.reset();
    for (Var* var : vars_) {
      var->Unref();
    }
  }

 private:
  std::vector<Var*> vars_;
  // NOTE: Use a `std::unique_ptr` instead of moving in a vector directly,
  // because a `std::vector<mutex_lock>` is not movable on all platforms.
  std::unique_ptr<std::vector<mutex_lock>> locks_;
};

VariableInputLockHolder MaybeLockVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, const std::vector<int>& input_ids);

void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input,
                                     int output);

// This is for use with ResourceVariables to ensure *tensor has a
// reference count of 1 before you update it.
// REQUIRES: If you pass in variable->tensor(), *variable->mu() must be held.
template <typename Device, typename T>
Status PrepareToUpdateVariable(OpKernelContext* ctx, Tensor* tensor) {
  if (!tensor->RefCountIsOne()) {
    // Tensor's buffer is in use by some read, so we need to copy before
    // updating.
    PersistentTensor unused;
    Tensor* tmp;
    if (std::is_same<T, Variant>::value) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      TF_RETURN_IF_ERROR(ctx->allocate_persistent(
          tensor->dtype(), tensor->shape(), &unused, &tmp, attr));

      const auto elements_in = tensor->flat<Variant>();
      auto elements_out = tmp->flat<Variant>();
      for (int64 i = 0; i < elements_in.size(); ++i) {
        elements_out(i) = elements_in(i);
      }
    } else {
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_nic_compatible(true);
      TF_RETURN_IF_ERROR(ctx->allocate_persistent(
          tensor->dtype(), tensor->shape(), &unused, &tmp, attr));
      functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
      copy_functor(ctx->eigen_device<Device>(), tmp->flat<T>(),
                   const_cast<const Tensor*>(tensor)->flat<T>());
    }
    *tensor = *tmp;
  }
  return Status::OK();
}

// This gives you `*out`, a tensor you can update, corresponding to a
// variable passed as input index `input`.  This handles the
// differences between reference and resource variables.  For resource
// variables, we ensure `*out` has a reference count of 1 (using
// PrepareToUpdateVariable() to copy if necessary) unless
// sparse && !lock_held, in which case it never copies.
template <typename Device, typename T>
Status GetInputTensorFromVariable(OpKernelContext* ctx, int input,
                                  bool lock_held, bool sparse, Tensor* out) {
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    Var* var;
    TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, input), &var));
    core::ScopedUnref unref_var(var);
    TF_RETURN_IF_ERROR(PrepareToUpdateVariable<Device, T>(ctx, var->tensor()));
    *out = *var->tensor();
    return Status::OK();
  }
  *out = ctx->mutable_input(input, lock_held);
  return Status::OK();
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_OP_HELPERS_H_
