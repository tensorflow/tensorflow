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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_DEVICE_ALLOCATION_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_DEVICE_ALLOCATION_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_allocation.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"

namespace xla {
namespace ifrt {
namespace proxy {

class DeviceAllocation final
    : public llvm::RTTIExtends<DeviceAllocation, xla::ifrt::DeviceAllocation> {
 public:
  DeviceAllocation(Client* client, std::shared_ptr<RpcHelper> rpc_helper,
                   std::string name, uint64_t handle, DeviceList devices,
                   DeviceList addressable_devices,
                   MemoryKind default_memory_kind,
                   std::vector<MemoryKind> all_memory_kinds,
                   AttributeMap attributes, std::string debug_string)
      : client_(client),
        rpc_helper_(std::move(rpc_helper)),
        name_(std::move(name)),
        handle_(handle),
        devices_(std::move(devices)),
        addressable_devices_(std::move(addressable_devices)),
        default_memory_kind_(default_memory_kind),
        all_memory_kinds_(std::move(all_memory_kinds)),
        attributes_(std::move(attributes)),
        debug_string_(std::move(debug_string)) {}

  ~DeviceAllocation() override;

  uint64_t handle() const { return handle_; }

  // DeviceAllocation implementation.

  Client* client() const override { return client_; }

  absl::string_view name() const override { return name_; }

  DeviceList GetDeviceList() const override { return devices_; }

  DeviceList GetAddressableDeviceList() const override {
    return addressable_devices_;
  }

  MemoryKind GetDefaultMemoryKind() const override {
    return default_memory_kind_;
  }

  std::vector<MemoryKind> GetAllMemoryKinds() const override {
    return all_memory_kinds_;
  }

  absl::StatusOr<std::shared_ptr<Topology>> GetTopology() const override;

  const AttributeMap& Attributes() const override { return attributes_; }

  std::string DebugString() const override { return debug_string_; }

  static char ID;  // NOLINT

 private:
  Client* client_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::string name_;

  uint64_t handle_;
  DeviceList devices_;
  DeviceList addressable_devices_;
  MemoryKind default_memory_kind_;
  std::vector<MemoryKind> all_memory_kinds_;
  AttributeMap attributes_;
  std::string debug_string_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_DEVICE_ALLOCATION_H_
