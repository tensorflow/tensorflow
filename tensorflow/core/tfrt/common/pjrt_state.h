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
#include <set>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/tsl/framework/allocator.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

const char kPjRtStateResourceName[] = "pjrt_state";
using PjRtClientsMap = std::map<DeviceType, std::unique_ptr<xla::PjRtClient>>;

// Information needed to create a PjRt GPU Client which is used when creating
// a client after after information about remote devices is available.
struct PjRtGpuClientCreationInfo {
  std::set<int> allowed_devices;
  std::unique_ptr<se::MultiDeviceAdapter> allocator;
  std::unique_ptr<tsl::Allocator> host_memory_allocator;
  std::map<int, std::unique_ptr<xla::LocalDeviceState>> local_device_states;
  xla::LocalClient* local_client;
};

// The class for the state related to PjRt. It contains a map from `DeviceType`
// to `PjRtClient`. It will be stored in the global `ResourceManager`.
class PjRtState : public ResourceBase {
 public:
  static PjRtState* Create();
  absl::StatusOr<xla::PjRtClient*> GetPjRtClient(const DeviceType& device_type);
  absl::StatusOr<xla::PjRtClient*> GetOrCreatePjRtClient(
      const DeviceType& device_type);
  Status SetPjRtClient(const DeviceType& device_type,
                       std::unique_ptr<xla::PjRtClient> client);
  // Moves PJRT client to `unused_`. The PJRT client moved to `unused_` will not
  // be returned by `GetPjRtClient`.
  Status MovePjRtClientToUnused(const DeviceType& device_type);
  string DebugString() const override;

  // Saves information needed to create a PJRT client (to enable creating a
  // client with remote devices).
  absl::Status SetPjRtGpuClientCreationInfo(
      std::unique_ptr<PjRtGpuClientCreationInfo> info);

  // Retrieves information needed to create a PJRT client (for creating a
  // client with remote devices).
  PjRtGpuClientCreationInfo* GetPjRtGpuClientCreationInfo();

 private:
  explicit PjRtState() {}
  absl::Mutex mu_;
  PjRtClientsMap clients_ ABSL_GUARDED_BY(mu_);
  // Store the PJRT clients that are no longer used to guarantee that PJRT
  // clients outlive PJRT buffers.
  std::vector<std::unique_ptr<xla::PjRtClient>> unused_ ABSL_GUARDED_BY(mu_);

  std::unique_ptr<PjRtGpuClientCreationInfo> pjrt_gpu_client_creation_info_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_COMMON_PJRT_STATE_H_
