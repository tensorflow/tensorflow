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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

// Must be called before performing a sparse operation on a variable. Ensures
// that no concurrent dense operations can happen while holding the variable's
// lock.
template <typename Device, typename T>
Status EnsureSparseVariableAccess(OpKernelContext* ctx, Var* var) {
  if (var->copy_on_read_mode.load()) {
    return Status::OK();
  }
  mutex_lock ml(*var->mu());
  // Once copy-on-read mode is True the refcount is guaranteed to be 1. This can
  // also happen if there are no concurrent reads of the variable and
  // copy-on-read mode is false.
  if (var->tensor()->RefCountIsOne()) {
    var->copy_on_read_mode.store(true);
    return Status::OK();
  }
  Tensor tmp;
  if (std::is_same<T, Variant>::value) {
    AllocatorAttributes attr;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(ctx->allocate_temp(var->tensor()->dtype(),
                                          var->tensor()->shape(), &tmp, attr));

    const auto elements_in = var->tensor()->flat<Variant>();
    auto elements_out = tmp.flat<Variant>();
    for (int64_t i = 0; i < elements_in.size(); ++i) {
      elements_out(i) = elements_in(i);
    }
  } else {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    TF_RETURN_IF_ERROR(ctx->allocate_temp(var->tensor()->dtype(),
                                          var->tensor()->shape(), &tmp, attr));
    functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
    copy_functor(ctx->eigen_device<Device>(), tmp.flat<T>(),
                 const_cast<const Tensor*>(var->tensor())->flat<T>());
  }
  *var->tensor() = tmp;
  var->copy_on_read_mode.store(true);
  return Status::OK();
}

// Utility structure that releases a sequence of borrowed mutexes when it is
// deleted.
struct VariableInputLockHolder {
 public:
  VariableInputLockHolder(
      std::vector<Var*> vars, std::unique_ptr<std::vector<mutex_lock>> locks,
      std::unique_ptr<std::vector<tf_shared_lock>> shared_locks)
      : vars_(std::move(vars)),
        locks_(std::move(locks)),
        shared_locks_(std::move(shared_locks)) {}

  VariableInputLockHolder(VariableInputLockHolder&& other)
      : vars_(std::move(other.vars_)),
        locks_(std::move(other.locks_)),
        shared_locks_(std::move(other.shared_locks_)) {}

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
  std::unique_ptr<std::vector<tf_shared_lock>> shared_locks_;
};

// Returns a borrowed pointer to the mutex for the variable `input` in `ctx`.
//
// If `input` corresponds to a `DT_RESOURCE`-type variable input,
// `*maybe_resource` will be updated to contain the underlying resource, and the
// caller will be responsible for calling `Unref()` on that resource.
template <typename Device, typename T>
mutex* GetTrainingVariableMutex(OpKernelContext* ctx, int input, bool sparse,
                                Var** maybe_resource) {
  *maybe_resource = nullptr;
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    if (LookupResource(ctx, HandleFromInput(ctx, input), maybe_resource).ok()) {
      if (sparse) {
        EnsureSparseVariableAccess<Device, T>(ctx, *maybe_resource)
            .IgnoreError();
      }
      return (*maybe_resource)->mu();
    } else {
      ctx->CtxFailureWithWarning(
          errors::Internal("Invalid variable reference."));
      return nullptr;
    }
  }
  return ctx->input_ref_mutex(input);
}

// MaybeLockVariableInputMutexesInOrder is a helper function to acquire mutexes
// in address order to mitigate deadlock.  Returns a structure that, when
// deleted, will release the acquired mutexes. Safe to pass duplicates - will
// only lock each distinct mutex once. If sparse is true will ensure the
// variable gets switched to copy-on-read mode before trying to acquire the
// locks. If do_lock is false, returns immediately for reference variables. For
// resource variables in copy-on-read-mode it will grab a shared lock if do_lock
// is false, exclusive lock otherwise.  Note that this silently doesn't lock
// mutexes for invalid variable references; in all usages this is followed by
// GetInputTensor which will signal a failure.
template <typename Device, typename T>
VariableInputLockHolder MaybeLockVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, bool sparse,
    const std::vector<int>& input_ids) {
  bool any_resource = false;
  for (auto i : input_ids) {
    if (ctx->input_dtype(i) == DT_RESOURCE) {
      any_resource = true;
      break;
    }
  }
  if (!do_lock && !any_resource) {
    return VariableInputLockHolder({}, {}, {});
  }
  std::vector<Var*> vars;
  std::vector<mutex*> mutexes;
  std::vector<int> acquire_order;
  for (auto input : input_ids) {
    Var* var;
    mutex* mutex =
        GetTrainingVariableMutex<Device, T>(ctx, input, sparse, &var);
    if (var) vars.push_back(var);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  auto locks = absl::make_unique<std::vector<mutex_lock>>();
  auto shared_locks = absl::make_unique<std::vector<tf_shared_lock>>();
  locks->reserve(acquire_order.size());

  for (auto input : acquire_order) {
    Var* var;
    mutex* mu = GetTrainingVariableMutex<Device, T>(ctx, input, sparse, &var);
    core::ScopedUnref scoped_unref(var);
    if (mu != nullptr) {
      if (!sparse || do_lock) {
        locks->emplace_back(*mu);
      } else {
        shared_locks->emplace_back(*mu);
      }
    }
  }
  return VariableInputLockHolder(std::move(vars), std::move(locks),
                                 std::move(shared_locks));
}

void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input,
                                     int output);

// This is for use with ResourceVariables to ensure *tensor has a
// reference count of 1 before you update it.
// REQUIRES: If you pass in variable->tensor(), *variable->mu() must be held.
template <typename Device, typename T>
Status PrepareToUpdateVariable(OpKernelContext* ctx, Tensor* tensor,
                               bool copy_on_read_mode) {
  if (copy_on_read_mode || !tensor->RefCountIsOne()) {
    // Tensor's buffer is in use by some read, so we need to copy before
    // updating.
    Tensor tmp;
    if (std::is_same<T, Variant>::value) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(tensor->dtype(), tensor->shape(), &tmp, attr));

      const auto elements_in = tensor->flat<Variant>();
      auto elements_out = tmp.flat<Variant>();
      for (int64_t i = 0; i < elements_in.size(); ++i) {
        elements_out(i) = elements_in(i);
      }
    } else {
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_nic_compatible(true);
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(tensor->dtype(), tensor->shape(), &tmp, attr));
      functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
      copy_functor(ctx->eigen_device<Device>(), tmp.flat<T>(),
                   const_cast<const Tensor*>(tensor)->flat<T>());
    }
    *tensor = tmp;
  }
  return Status::OK();
}

// This gives you `*out`, a tensor you can update, corresponding to a variable
// passed as input index `input`.  This handles the differences between
// reference and resource variables. For reference variables we can just grab
// the tensor, grabbing the lock if lock_held is False.
//
// For resource variables we, if sparse is true, ensure it's in copy-on-read
// mode, and then, regardless of the value of sparse, ensure its refcount is 1
// (by potentially copying its contents). In this case lock_held is ignored.
template <typename Device, typename T>
Status GetInputTensorFromVariable(OpKernelContext* ctx, int input,
                                  bool lock_held, bool sparse, Tensor* out) {
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    core::RefCountPtr<Var> var;
    TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, input), &var));
    if (sparse) {
      TF_RETURN_IF_ERROR(EnsureSparseVariableAccess<Device, T>(ctx, var.get()));
      *out = *var->tensor();
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(PrepareToUpdateVariable<Device, T>(
        ctx, var->tensor(), var->copy_on_read_mode.load()));
    *out = *var->tensor();
    return Status::OK();
  }
  *out = ctx->mutable_input(input, lock_held);
  return Status::OK();
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_OP_HELPERS_H_
