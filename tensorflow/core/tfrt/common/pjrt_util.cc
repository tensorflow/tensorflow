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
#include "tensorflow/core/tfrt/common/pjrt_util.h"

#include <memory>
#include <optional>
#include <set>

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/common/global_state.h"
#include "tensorflow/core/tfrt/common/pjrt_state.h"

namespace tensorflow {

Status SetPjRtClientInTFGlobalResourceManager(
    const DeviceType& device_type, std::unique_ptr<xla::PjRtClient> client) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  if (client == nullptr) {
    return errors::InvalidArgument("PJRT client is nullptr.");
  }
  TF_RETURN_IF_ERROR(pjrt_state->SetPjRtClient(device_type, std::move(client)));
  return OkStatus();
}

Status DeletePjRtClientFromTFGlobalResourceManagerIfResourceExists(
    const DeviceType& device_type) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  auto status = rmgr->Lookup(rmgr->default_container(), kPjRtStateResourceName,
                             &pjrt_state);
  if (!status.ok() && status.code() != error::NOT_FOUND) {
    return errors::Internal(
        "Failed to find PjRtState Resource when deleting PJRT client is "
        "requested: ",
        status.error_message());
  }
  // This method may be called before PJRT resource is created. It is OK to
  // receive NOT_FOUND in the resource look up.
  if (status.code() == error::NOT_FOUND) {
    LOG(INFO) << "PjRtState Resource is not found in TF GlobalResourceManager.";
    return OkStatus();
  }
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  TF_RETURN_IF_ERROR(pjrt_state->DeletePjRtClientIfExists(device_type));
  return OkStatus();
}

StatusOr<xla::PjRtClient*> GetOrCreatePjRtClient(
    const DeviceType& device_type,
    std::optional<std::set<int>> allowed_devices) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  StatusOr<xla::PjRtClient*> existing_pjrt_client =
      pjrt_state->GetPjRtClient(device_type);
  // Checks whether a PJRT client is found first as the DeviceType can choose to
  // create the PJRT client explicitly (e.g. in ops).
  if (existing_pjrt_client.ok()) {
    return *existing_pjrt_client;
  }
  // Returns directly if the error is not NotFound.
  if (!tsl::errors::IsNotFound(existing_pjrt_client.status())) {
    return existing_pjrt_client;
  }
  // TODO(b/260799193): use XlaPlatformInfo to pass device-specific options.
  // This info should be set in the plugin init for next pluggable device.
  // TODO(b/265435743): add GetStreamExecutorGpuClient for DEVICE_GPU when the
  // cuda_platform dependency in se_gpu_pjrt_client is changed to be compatible
  // with tf_cuda_cc_test.
  return errors::Unimplemented(
      "The PJRT client for ", device_type,
      " is not created explicitly before its first use and creating this "
      "PJRT client on the first use is not implemented.");
}

}  // namespace tensorflow
