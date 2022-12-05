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

StatusOr<xla::PjRtClient*> GetPjRtClientFromTFGlobalResourceManager(
    const DeviceType& device_type) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->Lookup(rmgr->default_container(),
                                  kPjRtStateResourceName, &pjrt_state));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  TF_ASSIGN_OR_RETURN(auto pjrt_client, pjrt_state->GetPjRtClient(device_type));
  return pjrt_client;
}

}  // namespace tensorflow
