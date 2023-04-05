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
#include <optional>
#include <string>
#include <utility>

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/ref_var.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

#ifndef IS_MOBILE_PLATFORM
#include "tensorflow/core/kernels/data/optional_ops_util.h"
#include "tensorflow/core/kernels/tensor_list.h"
#include "tensorflow/core/kernels/tensor_list_util.h"
#include "tensorflow/core/kernels/variant_ops_util.h"
#include "tensorflow/core/platform/abi.h"
#endif  // IS_MOBILE_PLATFORM

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
    tensorflow::Var* var, bool lock_held = false) {
  auto* context = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (var->copy_on_read_mode.load()) {
    return ::tensorflow::OkStatus();
  }

  std::optional<mutex_lock> ml;
  if (!lock_held) {
    ml.emplace(*var->mu());
  }

  // Once copy-on-read mode is True the refcount is guaranteed to be 1. This can
  // also happen if there are no concurrent reads of the variable and
  // copy-on-read mode is false.
  if (var->tensor()->RefCountIsOne()) {
    var->copy_on_read_mode.store(true);
    return ::tensorflow::OkStatus();
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
  return ::tensorflow::OkStatus();
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
  return ::tensorflow::OkStatus();
}

tensorflow::mutex* GetTrainingVariableMutex(TF_OpKernelContext* ctx,
                                            int32_t input,
                                            tensorflow::Var** maybe_resource) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  *maybe_resource = nullptr;
  if (cc_ctx->input_dtype(input) == tensorflow::DT_RESOURCE) {
    if (LookupResource(cc_ctx, HandleFromInput(cc_ctx, input), maybe_resource)
            .ok()) {
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
                       int value_index, bool validate_shape,
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
                               return ::tensorflow::OkStatus();
                             }));
  tensorflow::mutex_lock ml(*variable->mu());

  if (validate_shape) {
    OP_REQUIRES(cc_ctx,
                (!variable->is_initialized ||
                 variable->tensor()->shape().IsSameSize(value.shape())),
                InvalidArgument(
                    "Trying to assign to variable with tensor with wrong shape."
                    " Expected ",
                    variable->tensor()->shape().DebugString(), " got ",
                    value.shape().DebugString()));
  }

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

void TF_AssignRefVariable(TF_OpKernelContext* ctx, int input_ref_index,
                          int output_ref_index, int value_index,
                          bool use_locking, bool validate_shape,
                          void (*copyFunc)(TF_OpKernelContext* ctx,
                                           TF_Tensor* source, TF_Tensor* dest),
                          TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);

  auto copy = [copyFunc, ctx](::tensorflow::OpKernelContext* cc_ctx,
                              ::tensorflow::Tensor* lhs,
                              const ::tensorflow::Tensor& rhs) {
    ::tensorflow::Status s;
    TF_Tensor* tf_lhs = TF_TensorFromTensor(*lhs, &s);
    OP_REQUIRES_OK(cc_ctx, s);

    TF_Tensor* tf_rhs = TF_TensorFromTensor(rhs, &s);

    if (!s.ok()) {
      TF_DeleteTensor(tf_lhs);
      OP_REQUIRES_OK(cc_ctx, s);
    }

    copyFunc(ctx, tf_rhs, tf_lhs);
  };

  ::tensorflow::AssignRefVariable(cc_ctx, input_ref_index, output_ref_index,
                                  value_index, use_locking, validate_shape,
                                  false, copy);
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
    tensorflow::mutex* mutex = GetTrainingVariableMutex(ctx, input, &var);
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

  for (auto acquire : acquire_order) {
    tensorflow::mutex* mu = mutexes[acquire];
    if (mu != nullptr) {
      if (do_lock) {
        locks->emplace_back(*mu);
      } else {
        shared_locks->emplace_back(*mu);
      }
    }
  }
  *lockHolder = new TF_VariableInputLockHolder(vars, std::move(locks),
                                               std::move(shared_locks));
  if (sparse) {
    // Enable sparse variables' access.
    // NOTE: This can not be done before the variable input locks are held,
    // because a race condition can happen between this and another thread that
    // turns off some variable's `copy_on_read_mode` after this thread enables
    // sparse access; when a later function sees `copy_on_read_mode` is off, it
    // will try to lock the variable again for updating `copy_on_read_mode` and
    // cause the deadlock, since the variable mutex is non-re-entrant.
    for (auto* var : vars) {
      TF_CHECK_OK(EnsureSparseVariableAccess(
          ctx, /*variantType=*/false, copyFunc, var, /*lock_held=*/true));
    }
  }
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

  auto status_setter = ::tensorflow::gtl::MakeCleanup([cc_ctx, status]() {
    ::tensorflow::Set_TF_Status_from_Status(status, cc_ctx->status());
  });

  tensorflow::Status s;
  if (cc_ctx->input_dtype(input) == tensorflow::DT_RESOURCE) {
    tensorflow::core::RefCountPtr<tensorflow::Var> var;
    OP_REQUIRES_OK(
        cc_ctx, LookupResource(cc_ctx, HandleFromInput(cc_ctx, input), &var));
    if (sparse) {
      OP_REQUIRES_OK(cc_ctx, EnsureSparseVariableAccess(ctx, isVariantType,
                                                        copyFunc, var.get()));
      *out = ::tensorflow::TF_TensorFromTensor(*var->tensor(), &s);
      OP_REQUIRES_OK(cc_ctx, s);
      return;
    }
    OP_REQUIRES_OK(cc_ctx, PrepareToUpdateVariable(
                               ctx, var->tensor(),
                               var->copy_on_read_mode.load(), false, copyFunc));
    *out = ::tensorflow::TF_TensorFromTensor(*var->tensor(), &s);
    OP_REQUIRES_OK(cc_ctx, s);
    return;
  }
  *out = ::tensorflow::TF_TensorFromTensor(
      cc_ctx->mutable_input(input, lock_held), &s);
  OP_REQUIRES_OK(cc_ctx, s);
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

bool TF_IsRefInput(TF_OpKernelContext* ctx, int i, TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return false;
  }
  TF_SetStatus(status, TF_OK, "");
  return cc_ctx->input_is_ref(i);
}

#ifndef IS_MOBILE_PLATFORM
template <typename T>
static Status ValidateVariantType(const Variant& variant) {
  if (variant.get<T>() == nullptr) {
    const std::string type_index_name =
        ::tensorflow::port::MaybeAbiDemangle(variant.TypeId().name());

    return ::tensorflow::errors::Internal(
        "VariantBinaryOpFn: Could not access object 'a', type_index: ",
        type_index_name);
  }

  return ::tensorflow::OkStatus();
}

void TF_AddNVariant(TF_OpKernelContext* ctx,
                    void (*binary_add_func)(TF_OpKernelContext* ctx,
                                            TF_Tensor* a, TF_Tensor* b,
                                            TF_Tensor* out),
                    TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);

  auto cc_binary_add_func = [binary_add_func](
                                ::tensorflow::OpKernelContext* cc_ctx,
                                const Tensor& cc_a, const Tensor& cc_b,
                                Tensor* cc_out) {
    if (cc_a.dtype() == ::tensorflow::DT_INVALID) {
      *cc_out = cc_b;
      return ::tensorflow::OkStatus();
    }
    if (cc_b.dtype() == ::tensorflow::DT_INVALID) {
      *cc_out = cc_a;
      return ::tensorflow::OkStatus();
    }

    Status status;
    TF_Tensor* a = TF_TensorFromTensor(cc_a, &status);
    TF_RETURN_IF_ERROR(status);

    TF_Tensor* b = TF_TensorFromTensor(cc_b, &status);
    if (!status.ok()) {
      TF_DeleteTensor(a);
      return status;
    }

    ::tensorflow::AllocatorAttributes attr;
    if (cc_a.dtype() == ::tensorflow::DT_VARIANT) {
      attr.set_on_host(true);
    }

    status = cc_ctx->allocate_temp(cc_a.dtype(), cc_a.shape(), cc_out, attr);
    if (!status.ok()) {
      TF_DeleteTensor(a);
      TF_DeleteTensor(b);
      return status;
    }

    TF_Tensor* out = TF_TensorFromTensor(*cc_out, &status);
    if (!status.ok()) {
      TF_DeleteTensor(a);
      TF_DeleteTensor(b);
      return status;
    }

    auto* ctx = reinterpret_cast<TF_OpKernelContext*>(cc_ctx);
    binary_add_func(ctx, a, b, out);
    return cc_ctx->status();
  };

  auto binary_add_variant = [cc_binary_add_func](
                                ::tensorflow::OpKernelContext* cc_ctx,
                                const Variant& a, const Variant& b,
                                Variant* out) {
    if (out == nullptr) {
      return ::tensorflow::errors::Internal(
          "The output variant hasn't been initialized");
    }

    if (a.TypeId() != b.TypeId()) {
      return ::tensorflow::errors::Internal(
          "BinaryOpVariants: Variants a and b have different "
          "type ids.  Type names: '",
          a.TypeName(), "' vs. '", b.TypeName(), "'");
    }

    if (a.TypeId() == tensorflow::TypeIndex::Make<::tensorflow::TensorList>()) {
      TF_RETURN_IF_ERROR(ValidateVariantType<::tensorflow::TensorList>(a));
      *out = ::tensorflow::TensorList();

      return ::tensorflow::TensorListBinaryAdd(
          cc_ctx, *a.get<::tensorflow::TensorList>(),
          *b.get<::tensorflow::TensorList>(),
          out->get<::tensorflow::TensorList>(), cc_binary_add_func);
    } else if (a.TypeId() == tensorflow::TypeIndex::Make<
                                 ::tensorflow::data::OptionalVariant>()) {
      TF_RETURN_IF_ERROR(
          ValidateVariantType<::tensorflow::data::OptionalVariant>(a));
      *out = ::tensorflow::data::OptionalVariant();

      return ::tensorflow::data::OptionalBinaryAdd(
          cc_ctx, *a.get<::tensorflow::data::OptionalVariant>(),
          *b.get<::tensorflow::data::OptionalVariant>(),
          out->get<::tensorflow::data::OptionalVariant>(), cc_binary_add_func);
    }

    const std::string type_index_name =
        ::tensorflow::port::MaybeAbiDemangle(a.TypeId().name());

    return ::tensorflow::errors::Internal(
        "No unary variant binary_op function found for op ADD Variant "
        "type_name: ",
        type_index_name, " for device type: ", cc_ctx->device()->name());
  };
  ::tensorflow::AddNVariant(cc_ctx, binary_add_variant);
  ::tensorflow::Set_TF_Status_from_Status(status, cc_ctx->status());
}

static Status ZerosLikeVariant(::tensorflow::OpKernelContext* cc_ctx,
                               const Variant& input, Variant* out,
                               void (*zeros_like_func)(TF_OpKernelContext* ctx,
                                                       TF_Tensor* input,
                                                       TF_Tensor* out)) {
  auto cc_zeros_like_func = [zeros_like_func](
                                ::tensorflow::OpKernelContext* cc_ctx,
                                const Tensor& cc_input, Tensor* cc_out) {
    AllocatorAttributes attr;
    if (cc_input.dtype() == ::tensorflow::DT_VARIANT) {
      attr.set_on_host(true);
    }
    TF_RETURN_IF_ERROR(cc_ctx->allocate_temp(cc_input.dtype(), cc_input.shape(),
                                             cc_out, attr));

    switch (cc_input.dtype()) {
      case ::tensorflow::DT_INVALID: {
        *cc_out = Tensor(::tensorflow::DT_INVALID);
        break;
      }
      case ::tensorflow::DT_VARIANT: {
        // If the wrapped tensor is also a variant, recursively call
        // ZerosLikeVariant to unwrap it the same way
        Variant* out_variant = cc_out->scalar<Variant>().data();
        TF_RETURN_IF_ERROR(ZerosLikeVariant(cc_ctx,
                                            cc_input.scalar<Variant>()(),
                                            out_variant, zeros_like_func));
        break;
      }
      default: {
        Status status;
        TF_Tensor* input = TF_TensorFromTensor(cc_input, &status);
        TF_RETURN_IF_ERROR(status);

        TF_Tensor* out = TF_TensorFromTensor(*cc_out, &status);
        if (!status.ok()) {
          TF_DeleteTensor(input);
          return status;
        }

        auto* ctx = reinterpret_cast<TF_OpKernelContext*>(cc_ctx);
        zeros_like_func(ctx, input, out);
      }
    }
    return cc_ctx->status();
  };

  if (out == nullptr) {
    return ::tensorflow::errors::Internal(
        "The output variant hasn't been initialized");
  }

  if (input.TypeId() ==
      tensorflow::TypeIndex::Make<::tensorflow::TensorList>()) {
    TF_RETURN_IF_ERROR(ValidateVariantType<::tensorflow::TensorList>(input));
    *out = ::tensorflow::TensorList();

    return ::tensorflow::TensorListZerosLike(
        cc_ctx, *input.get<::tensorflow::TensorList>(),
        out->get<::tensorflow::TensorList>(), cc_zeros_like_func);
  } else if (input.TypeId() == tensorflow::TypeIndex::Make<
                                   ::tensorflow::data::OptionalVariant>()) {
    TF_RETURN_IF_ERROR(
        ValidateVariantType<::tensorflow::data::OptionalVariant>(input));
    *out = ::tensorflow::data::OptionalVariant();

    return ::tensorflow::data::OptionalZerosLike(
        cc_ctx, *input.get<::tensorflow::data::OptionalVariant>(),
        out->get<::tensorflow::data::OptionalVariant>(), cc_zeros_like_func);
  }

  const std::string type_index_name =
      ::tensorflow::port::MaybeAbiDemangle(input.TypeId().name());

  return ::tensorflow::errors::Internal(
      "No unary variant unary_op function found for op ZEROS_LIKE Variant "
      "type_name: ",
      type_index_name, " for device type: ", cc_ctx->device()->name());
}

void TF_ZerosLikeVariant(TF_OpKernelContext* ctx,
                         void (*zeros_like_func)(TF_OpKernelContext* ctx,
                                                 TF_Tensor* input,
                                                 TF_Tensor* out),
                         TF_Status* status) {
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);

  const Tensor& input = cc_ctx->input(0);
  OP_REQUIRES(cc_ctx, input.dims() == 0,
              InvalidArgument(
                  "ZerosLike non-scalar Tensor with dtype=DT_VARIANT is not "
                  "supported."));
  const Variant& v = input.scalar<Variant>()();
  // DT_VARIANT tensors must be allocated on CPU since they wrap C++
  // objects which can not be efficiently represented in GPU memory.
  int numa_node = cc_ctx->device()->NumaNode();
  Tensor out(::tensorflow::cpu_allocator(numa_node), ::tensorflow::DT_VARIANT,
             ::tensorflow::TensorShape({}));
  Variant* out_v = &(out.scalar<Variant>()());
  Status cc_status = ZerosLikeVariant(cc_ctx, v, out_v, zeros_like_func);
  ::tensorflow::Set_TF_Status_from_Status(status, cc_status);
  OP_REQUIRES_OK(cc_ctx, cc_status);
  cc_ctx->set_output(0, out);
}
#endif  // IS_MOBILE_PLATFORM
