/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/eager/execute_node.h"

#include "xla/tsl/util/env_var.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

#if !defined(IS_MOBILE_PLATFORM)
bool ExecuteNodeArgs::IsRemote(EagerContext* ctx, Device* input_device,
                               TensorHandle* handle) {
  uint64 context_view_id = ctx->GetContextViewId();
  if (handle->Type() == TensorHandle::REMOTE ||
      handle->HasRemoteMirror(input_device, context_view_id)) {
    if (!has_remote_inputs_) {
      has_remote_inputs_ = true;
    }
    return true;
  }
  return false;
}
#endif  // IS_MOBILE_PLATFORM

absl::Status ExecuteNodeArgs::InitPackedHandle(const int index,
                                               EagerContext* ctx,
                                               Device* input_device,
                                               TensorHandle* packed_handle) {
  int num_handles = packed_handle->NumPackedHandles();
  packed_args_.emplace(index,
                       absl::InlinedVector<TensorValue, 4UL>(num_handles));
  TensorValue* packed_arg_flat = &(packed_args_[index][0]);
  for (int i = 0; i < num_handles; ++i) {
    TensorHandle* h = nullptr;
    TF_RETURN_IF_ERROR(packed_handle->ExtractPackedHandle(i, &h));
    // We have validated that h->device() is not a CustomDevice when
    // constructing a pack TensorHandle.
    const absl::Status status =
        h->TensorValue(h->device(), &packed_arg_flat[i]);
    if (!status.ok()) {
#if !defined(IS_MOBILE_PLATFORM)
      if (IsRemote(ctx, input_device, h)) {
        continue;
      }
#endif  // IS_MOBILE_PLATFORM
      if (h->Type() == TensorHandle::PACKED) {
        return errors::InvalidArgument(
            "Nested packed handles are not supported");
      }
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status ExecuteNodeArgs::Init(
    EagerContext* ctx, const absl::InlinedVector<TensorHandle*, 4UL>& op_inputs,
    const core::RefCountPtr<KernelAndDevice>& kernel) {
  // If there are multiple references to a TensorHandle in 'op_inputs' we must
  // increment the reference count of the corresponding Tensor or risk it being
  // overwritten during kernel execution. The reference count is incremented
  // below when we insert a copy of the Tensor into protected_tensors, and will
  // be decremented once execution is complete.
  const int n_inputs = op_inputs.size();
  if (n_inputs > 0) {
    TensorHandle* const* op_inputs_flat = &op_inputs[0];
    TensorValue* tensor_args_flat = &tensor_args_[0];
    for (int i = 0; i < n_inputs; ++i) {
      TensorHandle* in = op_inputs_flat[i];
      Device* d = kernel->InputDevice(i);
      absl::Status s =
          in->TensorValue(ctx->CanonicalDevice(d), &tensor_args_flat[i]);
      if (!s.ok()) {
#if !defined(IS_MOBILE_PLATFORM)
        if (IsRemote(ctx, d, in)) {
          continue;
        }
#endif
        if (in->Type() != TensorHandle::PACKED) {
          return s;
        }
        if (!has_packed_inputs_) {
          has_packed_inputs_ = true;
        }
        TF_RETURN_IF_ERROR(InitPackedHandle(i, ctx, d, in));
      }
    }
  }

#if !defined(IS_MOBILE_PLATFORM)
  if (has_remote_inputs_) {
    const bool is_function = kernel->IsFunction();
    serialize_remote_handle_ =
        [ctx, &op_inputs, is_function](
            const FunctionArgIndex& index,
            eager::RemoteTensorHandle* handle) -> absl::Status {
      TensorHandle* h = op_inputs[index.index];
      if (op_inputs[index.index]->Type() == TensorHandle::PACKED) {
        TF_RETURN_IF_ERROR(
            op_inputs[index.index]->ExtractPackedHandle(index.sub_index, &h));
      }
      Device* device = h->device();
      // For a multi-device function, a remote RunComponentFunction request is
      // not sent through StreamingEnqueueAsync. It could arrive at a remote
      // worker before a remote execution request which produces an input of the
      // component function. So we wait until the remote input is ready before
      // serializing it.
      bool wait_until_ready = SkipRemoteHandleWaitReady() ? false : is_function;
      return ctx->RemoteMgr()->SerializeRemoteTensorHandle(h, wait_until_ready,
                                                           handle, device);
    };
  }
#endif  // !IS_MOBILE_PLATFORM
  return absl::OkStatus();
}

absl::Status ExecuteNodeArgs::GetLocalArg(const FunctionArgIndex& index,
                                          Tensor* val) const {
  absl::Status s = EagerKernelArgs::GetLocalArg(index, val);
  if (s.ok()) {
    return absl::OkStatus();
  }
  if (packed_args_.contains(index.index)) {
    Tensor* arg = packed_args_.at(index.index).at(index.sub_index).tensor;
    if (arg) {
      *val = *arg;
      return absl::OkStatus();
    } else {
      return errors::NotFound("Argument (", index.index, ",", index.sub_index,
                              ") has no local tensor.");
    }
  } else {
    return s;
  }
}

}  // namespace tensorflow
