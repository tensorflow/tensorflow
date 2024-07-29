/*
 * Copyright 2024 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xla/python/ifrt_proxy/client/device_allocation.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/topology.h"

namespace xla {
namespace ifrt {
namespace proxy {

char DeviceAllocation::ID = 0;

DeviceAllocation::~DeviceAllocation() {
  auto req = std::make_unique<DestructDeviceAllocationRequest>();
  req->set_device_allocation_handle(handle_);

  rpc_helper_->DestructDeviceAllocation(std::move(req))
      .OnReady(
          [](absl::StatusOr<std::shared_ptr<DestructDeviceAllocationResponse>>
                 response) {
            if (!response.ok()) {
              LOG(ERROR) << "Failed to destroy `DeviceAllocation`: "
                         << response.status();
            }
          });
}

absl::StatusOr<std::shared_ptr<Topology>> DeviceAllocation::GetTopology()
    const {
  return absl::UnimplementedError(
      "GetTopologyForDevices is not supported for the IFRT proxy client.");
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
