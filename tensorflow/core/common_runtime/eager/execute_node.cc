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

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
Status ExecuteNodeArgs::Init(
    EagerContext* ctx, const gtl::InlinedVector<TensorHandle*, 4>& op_inputs,
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
      Status s = in->TensorValue(ctx->CanonicalDevice(d), &tensor_args_flat[i]);
      if (!s.ok()) {
#if !defined(IS_MOBILE_PLATFORM)
        uint64 context_view_id = ctx->GetContextViewId();
        if (in->IsRemote() || in->HasRemoteMirror(d, context_view_id)) {
          if (!has_remote_inputs_) {
            has_remote_inputs_ = true;
          }
          continue;
        }
#endif
        return s;
      }
    }
  }

#if !defined(IS_MOBILE_PLATFORM)
  if (has_remote_inputs_) {
    serialize_remote_handle_ =
        [ctx, &op_inputs](const int i,
                          eager::RemoteTensorHandle* handle) -> Status {
      absl::variant<Device*, CustomDevice*> variant_device =
          op_inputs[i]->device();
      if (VariantDeviceIsCustom(variant_device)) {
        return errors::Internal(
            "Custom devices and remote execution are currently not supported "
            "together.");
      }
      Device* device = absl::get<Device*>(variant_device);
      return ctx->RemoteMgr()->SerializeRemoteTensorHandle(
          op_inputs[i], handle, device, device->name());
    };
  }
#endif  // !IS_MOBILE_PLATFORM
  return Status::OK();
}

}  // namespace tensorflow
