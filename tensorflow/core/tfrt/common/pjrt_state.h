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
#ifndef TENSORFLOW_CORE_TFRT_COMMON_PJRT_STATE_H_
#define TENSORFLOW_CORE_TFRT_COMMON_PJRT_STATE_H_

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

const char kPjRtStateResourceName[] = "pjrt_state";
using PjRtClientsMap = std::map<DeviceType, std::unique_ptr<xla::PjRtClient>>;

// The class for the state related to PjRt. It contains a map from `DeviceType`
// to `PjRtClient`. It will be stored in the global `ResourceManager`.
class PjRtState : public ResourceBase {
 public:
  static PjRtState* Create();
  StatusOr<xla::PjRtClient*> GetPjRtClient(const DeviceType& device_type);
  Status SetPjRtClient(const DeviceType& device_type,
                       std::unique_ptr<xla::PjRtClient> client);
  Status DeletePjRtClientIfExists(const DeviceType& device_type);
  string DebugString() const override;

 private:
  explicit PjRtState() {}
  absl::Mutex mu_;
  PjRtClientsMap clients_ ABSL_GUARDED_BY(mu_);
  // Store the PJRT clients that are no longer used to guarantee that PJRT
  // clients outlive PJRT buffers.
  std::vector<std::unique_ptr<xla::PjRtClient>> unused_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_COMMON_PJRT_STATE_H_
