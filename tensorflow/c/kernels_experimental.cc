/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/kernels_experimental.h"

#include <algorithm>
#include <utility>

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"

using tensorflow::AllocatorAttributes;
using tensorflow::mutex_lock;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TF_TensorFromTensor;
using tensorflow::Var;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

struct TF_VariableInputLockHolder {
  TF_VariableInputLockHolder(
      std::vector<tensorflow::Var*> vars,
      std::unique_ptr<std::vector<tensorflow::mutex_lock>> locks,
      std::unique_ptr<std::vector<tensorflow::tf_shared_lock>> shared_locks)
      : vars(std::move(vars)),
        locks(std::move(locks)),
        shared_locks(std::move(shared_locks)) {}

  std::vector<tensorflow::Var*> vars;
  std::unique_ptr<std::vector<tensorflow::mutex_lock>> locks;
  std::unique_ptr<std::vector<tensorflow::tf_shared_lock>> shared_locks;
};

tensorflow::Status EnsureSparseVariableAccess(
    TF_OpKernelContext* ctx, bool variantType,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    tensorflow::Var* var) {
  auto* context = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
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
  if (variantType) {
    AllocatorAttributes attr;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(context->allocate_temp(
        var->tensor()->dtype(), var->tensor()->shape(), &tmp, attr));

    const auto elements_in = var->tensor()->flat<Variant>();
    auto elements_out = tmp.flat<Variant>();
    for (int64_t i = 0; i < elements_in.size(); ++i) {
      elements_out(i) = elements_in(i);
    }
  } else {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    TF_RETURN_IF_ERROR(context->allocate_temp(
        var->tensor()->dtype(), var->tensor()->shape(), &tmp, attr));
    tensorflow::Status s;
    TF_Tensor* tf_tmp = TF_TensorFromTensor(tmp, &s);
    TF_Tensor* tf_tensor = TF_TensorFromTensor(*var->tensor(), &s);
    copyFunc(ctx, tf_tensor, tf_tmp);
  }
  *var->tensor() = tmp;
  var->copy_on_read_mode.store(true);
  return Status::OK();
}

tensorflow::Status PrepareToUpdateVariable(
    TF_OpKernelContext* ctx, tensorflow::Tensor* tensor, bool copy_on_read_mode,
    bool variantType,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest)) {
  auto* context = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (copy_on_read_mode || !tensor->RefCountIsOne()) {
    // Tensor's buffer is in use by some read, so we need to copy before
    // updating.
    Tensor tmp;
    if (variantType) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      TF_RETURN_IF_ERROR(
          context->allocate_temp(tensor->dtype(), tensor->shape(), &tmp, attr));

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
          context->allocate_temp(tensor->dtype(), tensor->shape(), &tmp, attr));
      tensorflow::Status s;
      TF_Tensor* tf_tmp = TF_TensorFromTensor(tmp, &s);
      TF_Tensor* tf_tensor = TF_TensorFromTensor(*tensor, &s);
      copyFunc(ctx, tf_tensor, tf_tmp);
    }
    *tensor = tmp;
  }
  return Status::OK();
}

tensorflow::mutex* GetTrainingVariableMutex(
    TF_OpKernelContext* ctx, int32_t input, bool sparse,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    tensorflow::Var** maybe_resource) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  *maybe_resource = nullptr;
  if (cc_ctx->input_dtype(input) == tensorflow::DT_RESOURCE) {
    if (LookupResource(cc_ctx, HandleFromInput(cc_ctx, input), maybe_resource)
            .ok()) {
      if (sparse) {
        TF_CHECK_OK(
            EnsureSparseVariableAccess(ctx, false, copyFunc, *maybe_resource));
      }
      return (*maybe_resource)->mu();
    } else {
      cc_ctx->CtxFailureWithWarning(
          tensorflow::errors::Internal("Invalid variable reference."));
      return nullptr;
    }
  }
  return cc_ctx->input_ref_mutex(input);
}

void TF_AssignVariable(TF_OpKernelContext* ctx, int input_index,
                       int value_index,
                       void (*copyFunc)(TF_OpKernelContext* ctx,
                                        TF_Tensor* source, TF_Tensor* dest),
                       TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  tensorflow::core::RefCountPtr<tensorflow::Var> variable;
  const tensorflow::Tensor& value = cc_ctx->input(value_index);
  OP_REQUIRES_OK(cc_ctx, tensorflow::LookupOrCreateResource<tensorflow::Var>(
                             cc_ctx, HandleFromInput(cc_ctx, input_index),
                             &variable, [&value](tensorflow::Var** ptr) {
                               *ptr = new tensorflow::Var(value.dtype());
                               *(*ptr)->tensor() = value;
                               (*ptr)->is_initialized = true;
                               return tensorflow::Status::OK();
                             }));
  tensorflow::mutex_lock ml(*variable->mu());

  if (variable->copy_on_read_mode.load()) {
    tensorflow::Tensor tmp;
    tensorflow::AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    OP_REQUIRES_OK(cc_ctx, cc_ctx->allocate_temp(value.dtype(), value.shape(),
                                                 &tmp, attr));
    tensorflow::Status s;
    TF_Tensor* tf_tmp = TF_TensorFromTensor(tmp, &s);
    TF_Tensor* tf_value = TF_TensorFromTensor(value, &s);
    copyFunc(ctx, tf_value, tf_tmp);
    *variable->tensor() = tmp;
  } else {
    *variable->tensor() = value;
  }
  variable->is_initialized = true;
  TF_SetStatus(status, TF_OK, "");
}

void TF_AssignUpdateVariable(TF_OpKernelContext* ctx, int input_index,
                             int value_index, int Op, int isVariantType,
                             void (*copyFunc)(TF_OpKernelContext* ctx,
                                              TF_Tensor* source,
                                              TF_Tensor* dest),
                             void (*updateFunc)(TF_OpKernelContext* ctx,
                                                TF_Tensor* tensor,
                                                TF_Tensor* value, int Op),
                             TF_Status* tf_status) {
  auto* context = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  tensorflow::core::RefCountPtr<Var> variable;
  Status status =
      LookupResource(context, HandleFromInput(context, input_index), &variable);
  if (!status.ok()) {
    printf("Failed with error: %s\n", status.error_message().c_str());
    abort();
  }
  const Tensor& value = context->input(value_index);
  mutex_lock ml(*variable->mu());
  Tensor* var_tensor = variable->tensor();
  OP_REQUIRES(
      context, var_tensor->shape().IsSameSize(value.shape()),
      InvalidArgument("Cannot update variable with shape ",
                      var_tensor->shape().DebugString(),
                      " using a Tensor with shape ",
                      value.shape().DebugString(), ", shapes must be equal."));
  OP_REQUIRES_OK(context,
                 PrepareToUpdateVariable(ctx, var_tensor,
                                         variable->copy_on_read_mode.load(),
                                         isVariantType, copyFunc));
  tensorflow::Status s;
  TF_Tensor* tf_var_tensor = TF_TensorFromTensor(*var_tensor, &s);
  TF_Tensor* tf_value = TF_TensorFromTensor(value, &s);
  updateFunc(ctx, tf_var_tensor, tf_value, Op);
  TF_SetStatus(tf_status, TF_OK, "");
}

void TF_MaybeLockVariableInputMutexesInOrder(
    TF_OpKernelContext* ctx, bool do_lock, bool sparse, const int* const inputs,
    size_t len,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    TF_VariableInputLockHolder** lockHolder, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  bool any_resource = false;
  std::vector<int> input_ids(inputs, inputs + len);
  for (auto i : input_ids) {
    if (cc_ctx->input_dtype(i) == tensorflow::DT_RESOURCE) {
      any_resource = true;
      break;
    }
  }
  if (!do_lock && !any_resource) {
    *lockHolder = new TF_VariableInputLockHolder({}, {}, {});
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  std::vector<tensorflow::Var*> vars;
  std::vector<tensorflow::mutex*> mutexes;
  std::vector<int32_t> acquire_order;
  for (auto input : input_ids) {
    tensorflow::Var* var;
    tensorflow::mutex* mutex =
        GetTrainingVariableMutex(ctx, input, sparse, copyFunc, &var);
    if (var) vars.push_back(var);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  auto locks = absl::make_unique<std::vector<tensorflow::mutex_lock>>();
  auto shared_locks =
      absl::make_unique<std::vector<tensorflow::tf_shared_lock>>();
  locks->reserve(acquire_order.size());

  for (auto input : acquire_order) {
    tensorflow::Var* var;
    tensorflow::mutex* mu =
        GetTrainingVariableMutex(ctx, input, sparse, copyFunc, &var);
    tensorflow::core::ScopedUnref scoped_unref(var);
    if (mu != nullptr) {
      if (do_lock) {
        locks->emplace_back(*mu);
      } else {
        shared_locks->emplace_back(*mu);
      }
    }
  }
  *lockHolder = new TF_VariableInputLockHolder(
      std::move(vars), std::move(locks), std::move(shared_locks));
  TF_SetStatus(status, TF_OK, "");
}

void TF_GetInputTensorFromVariable(TF_OpKernelContext* ctx, int input,
                                   bool lock_held, bool isVariantType,
                                   bool sparse,
                                   void (*copyFunc)(TF_OpKernelContext* ctx,
                                                    TF_Tensor* source,
                                                    TF_Tensor* dest),
                                   TF_Tensor** out, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  tensorflow::Status s;
  if (cc_ctx->input_dtype(input) == tensorflow::DT_RESOURCE) {
    tensorflow::core::RefCountPtr<tensorflow::Var> var;
    OP_REQUIRES_OK(
        cc_ctx, LookupResource(cc_ctx, HandleFromInput(cc_ctx, input), &var));
    if (sparse) {
      OP_REQUIRES_OK(cc_ctx, EnsureSparseVariableAccess(ctx, isVariantType,
                                                        copyFunc, var.get()));
      *out = ::tensorflow::TF_TensorFromTensor(*var->tensor(), &s);
      TF_SetStatus(status, TF_OK, "");
      return;
    }
    OP_REQUIRES_OK(cc_ctx, PrepareToUpdateVariable(
                               ctx, var->tensor(),
                               var->copy_on_read_mode.load(), false, copyFunc));
    *out = ::tensorflow::TF_TensorFromTensor(*var->tensor(), &s);
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  *out = ::tensorflow::TF_TensorFromTensor(
      cc_ctx->mutable_input(input, lock_held), &s);
  TF_SetStatus(status, TF_OK, "");
}

void TF_OpKernelContext_ForwardRefInputToRefOutput(TF_OpKernelContext* ctx,
                                                   int32_t input_index,
                                                   int32_t output_index) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (cc_ctx->input_dtype(input_index) != tensorflow::DT_RESOURCE) {
    cc_ctx->forward_ref_input_to_ref_output(input_index, output_index);
  }
}

void TF_ReleaseVariableInputLockHolder(TF_VariableInputLockHolder* lockHolder) {
  if (lockHolder != nullptr) {
    lockHolder->locks.reset();
    for (tensorflow::Var* var : lockHolder->vars) {
      var->Unref();
    }
    delete lockHolder;
  }
}

void TF_GetInputByName(TF_OpKernelContext* ctx, const char* inputName,
                       TF_Tensor** tensor, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  const ::tensorflow::Tensor* cc_tensor = nullptr;
  tensorflow::Status s = cc_ctx->input(inputName, &cc_tensor);

  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return;
  }
  TF_Tensor* result =
      ::tensorflow::TF_TensorFromTensor(*cc_tensor, &status->status);
  if (TF_GetCode(status) == TF_OK) {
    *tensor = result;
  }
}

void TF_OpKernelConstruction_GetAttrTensorShape(TF_OpKernelConstruction* ctx,
                                                const char* attr_name,
                                                int64_t* dims, size_t num_dims,
                                                TF_Status* status) {
  ::tensorflow::TensorShape shape;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &shape);
  ::tensorflow::Set_TF_Status_from_Status(status, s);
  size_t rank = static_cast<size_t>(shape.dims());

  if (!status->status.ok()) return;

  if (num_dims != rank) {
    status->status = InvalidArgument("Expected rank is ", num_dims,
                                     " but actual rank is ", rank);
    return;
  }

  for (int i = 0; i < rank; ++i) {
    dims[i] = static_cast<int64_t>(shape.dim_size(i));
  }
}
