/*
 * Copyright 2023 The OpenXLA Authors.
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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_DEVICE_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_DEVICE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"

namespace xla {
namespace ifrt {
namespace proxy {

class Client;

class DeviceDescription final : public xla::PjRtDeviceDescription {
 public:
  DeviceDescription(
      int id, int process_index, std::string device_kind,
      std::string debug_string, std::string to_string,
      absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes)
      : id_(id),
        process_index_(process_index),
        device_kind_(device_kind),
        debug_string_(std::move(debug_string)),
        to_string_(std::move(to_string)),
        attributes_(std::move(attributes)) {}

  int id() const override { return id_; }

  int process_index() const override { return process_index_; }

  absl::string_view device_kind() const override { return device_kind_; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::string_view ToString() const override { return to_string_; }

  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  int id_;
  int process_index_;
  std::string device_kind_;
  std::string debug_string_;
  std::string to_string_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

class Device final : public llvm::RTTIExtends<Device, xla::ifrt::Device> {
 public:
  Device(DeviceDescription description, int local_device_id,
         int local_hardware_id, bool is_addressable);

  ifrt::Client* client() const override;
  bool IsAddressable() const override;

  DeviceId Id() const override;
  absl::string_view Kind() const override;
  absl::string_view ToString() const override;
  absl::string_view DebugString() const override;
  int ProcessIndex() const override;

  const AttributeMap& Attributes() const override;

  absl::Span<ifrt::Memory* const> Memories() const override;
  absl::StatusOr<ifrt::Memory*> DefaultMemory() const override;

  static char ID;  // NOLINT

 private:
  friend class Client;  // For `memories_` initialization.

  ifrt::Client* client_;
  const DeviceDescription description_;

  const AttributeMap attributes_;

  const int local_device_id_;
  const int local_hardware_id_;
  const bool is_addressable_;

  std::vector<xla::ifrt::Memory*> memories_;
  xla::ifrt::Memory* default_memory_ = nullptr;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_DEVICE_H_
