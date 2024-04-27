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
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
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
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  if (client == nullptr) {
    return errors::InvalidArgument("PJRT client is nullptr.");
  }
  TF_RETURN_IF_ERROR(pjrt_state->SetPjRtClient(device_type, std::move(client)));
  return absl::OkStatus();
}

absl::StatusOr<xla::PjRtClient*> GetPjRtClient(const DeviceType& device_type) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  return pjrt_state->GetPjRtClient(device_type);
}

absl::Status SetPjRtGpuClientCreationInfoInTFGlobalResourceManager(
    std::unique_ptr<PjRtGpuClientCreationInfo> info) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  if (info == nullptr) {
    return absl::InvalidArgumentError("PJRT client creation info is nullptr.");
  }
  TF_RETURN_IF_ERROR(pjrt_state->SetPjRtGpuClientCreationInfo(std::move(info)));
  return absl::OkStatus();
}

absl::StatusOr<PjRtGpuClientCreationInfo*> GetPjRtGpuClientCreationInfo() {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<PjRtState>(
      rmgr->default_container(), kPjRtStateResourceName, &pjrt_state,
      [&](PjRtState** ret) {
        *ret = PjRtState::Create();
        return absl::OkStatus();
      }));
  core::ScopedUnref pjrt_state_ref(pjrt_state);
  return pjrt_state->GetPjRtGpuClientCreationInfo();
}
}  // namespace tensorflow
