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
#include "tensorflow/core/tfrt/common/pjrt_state.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

PjRtState* PjRtState::Create() { return new PjRtState(); }

StatusOr<xla::PjRtClient*> PjRtState::GetPjRtClient(
    const DeviceType& device_type) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    return it->second.get();
  }
  return errors::NotFound("PjRt client not found for device type ",
                          device_type);
}

Status PjRtState::SetPjRtClient(const DeviceType& device_type,
                                std::unique_ptr<xla::PjRtClient> client) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    unused_.push_back(std::move(it->second));
  }
  clients_[device_type] = std::move(client);
  return OkStatus();
}

Status PjRtState::DeletePjRtClientIfExists(const DeviceType& device_type) {
  absl::MutexLock lock(&mu_);
  if (auto it = clients_.find(device_type); it != clients_.end()) {
    clients_.erase(it);
  }
  return OkStatus();
}

string PjRtState::DebugString() const { return "PjRtState"; }

}  // namespace tensorflow
