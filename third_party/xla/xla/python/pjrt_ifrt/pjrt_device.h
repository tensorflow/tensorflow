/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_DEVICE_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_DEVICE_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"

namespace xla {
namespace ifrt {

class PjRtCompatibleDevice : public llvm::RTTIExtends<PjRtDevice, Device> {
 public:
  virtual xla::PjRtDevice* pjrt_device() const = 0;

  static char ID;  // NOLINT
};

class PjRtDevice final
    : public llvm::RTTIExtends<PjRtDevice, PjRtCompatibleDevice> {
 public:
  PjRtDevice(PjRtClient* client, DeviceId id, std::string kind,
             std::string to_string, std::string debug_string, int process_index,
             absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes,
             xla::PjRtDevice* pjrt_device);

  // Non-null only for addressable devices. nullptr for non-addressable devices.
  xla::PjRtDevice* pjrt_device() const override { return pjrt_device_; }

  // Device implementation.

  PjRtClient* client() const override { return client_; }

  DeviceId Id() const final;
  const AttributeMap& Attributes() const final;
  absl::string_view Kind() const final;
  absl::string_view ToString() const final;
  absl::string_view DebugString() const final;
  bool IsAddressable() const final;
  absl::StatusOr<Memory*> DefaultMemory() const final;
  absl::Span<Memory* const> Memories() const final;
  int ProcessIndex() const final;

  static char ID;  // NOLINT

 private:
  friend class PjRtClient;

  PjRtClient* client_;

  DeviceId id_;
  AttributeMap attributes_;
  std::string kind_;
  std::string to_string_;
  std::string debug_string_;
  absl::StatusOr<Memory*> default_memory_;
  std::vector<Memory*> memories_;
  int process_index_;

  xla::PjRtDevice* pjrt_device_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_DEVICE_H_
