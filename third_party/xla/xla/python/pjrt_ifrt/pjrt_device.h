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

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_client.h"
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
  PjRtDevice(PjRtClient* client, xla::PjRtDevice* pjrt_device);

  xla::PjRtDevice* pjrt_device() const override { return pjrt_device_; }

  // Device implementation.

  PjRtClient* client() const override { return client_; }

  DeviceId Id() const override;
  absl::string_view Kind() const override;
  absl::string_view ToString() const override;
  absl::string_view DebugString() const override;
  bool IsAddressable() const override;
  absl::StatusOr<Memory*> DefaultMemory() const override;
  absl::Span<Memory* const> Memories() const override;
  int ProcessIndex() const override;
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override;

  static char ID;  // NOLINT

 private:
  friend class PjRtClient;

  PjRtClient* client_;
  xla::PjRtDevice* pjrt_device_;
  std::vector<Memory*> memories_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_DEVICE_H_
