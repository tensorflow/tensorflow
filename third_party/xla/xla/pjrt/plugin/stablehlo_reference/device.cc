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

#include "xla/pjrt/plugin/stablehlo_reference/device.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/plugin/stablehlo_reference/logging.h"
#include "xla/util.h"

#define UNIMPLEMENTED(name) \
  xla::Unimplemented("StablehloReferenceDevice::" #name " is not implemented")

namespace mlir::stablehlo {

using xla::LiteralSlice;
using xla::MutableBorrowingLiteral;
using xla::PjRtClient;
using xla::PjRtDevice;
using xla::PjRtDeviceAttribute;
using xla::PjRtDeviceDescription;
using xla::PjRtGlobalDeviceId;
using xla::PjRtLocalDeviceId;
using xla::PjRtLocalHardwareId;
using xla::PjRtMemorySpace;
using xla::ScopedAsyncTrackingEvent;
using xla::Unimplemented;

// Devices need a device description.
class StablehloReferenceDeviceDescription final : public PjRtDeviceDescription {
 public:
  explicit StablehloReferenceDeviceDescription(int process_id,
                                               int local_device_id)
      : id_(local_device_id),
        process_index_(process_id),
        local_hardware_id_(local_device_id),
        debug_string_("StablehloReferenceDeviceDescription"),
        to_string_(absl::StrFormat("%s(id=%d,pid=%d)", debug_string_,
                                   id_.value(), process_index_)) {
    TRACE_ME_MEMBER;
  }

  int id() const override {
    TRACE_ME_MEMBER;
    return id_.value();
  }
  int process_index() const override {
    TRACE_ME_MEMBER;
    return process_index_;
  }
  int local_hardware_id() const {
    TRACE_ME_MEMBER;
    return local_hardware_id_;
  }

  absl::string_view device_kind() const override {
    TRACE_ME_MEMBER;
    return "StablehloReferenceDeviceDescription";
  }

  absl::string_view DebugString() const override {
    TRACE_ME_MEMBER;
    return debug_string_;
  }

  absl::string_view ToString() const override {
    TRACE_ME_MEMBER;
    return to_string_;
  }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    TRACE_ME_MEMBER;
    return attributes_;
  }

 private:
  PjRtGlobalDeviceId id_;
  int process_index_;
  int local_hardware_id_;
  std::string debug_string_;
  std::string to_string_;
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
};

// Clients need devices, and clients own the devices.
class StablehloReferenceDevice : public PjRtDevice {
 public:
  explicit StablehloReferenceDevice(PjRtClient* client)
      : PjRtDevice(), client_(client), description_(0, 0) {
    TRACE_ME_MEMBER;
  }

  absl::string_view DebugString() const override {
    TRACE_ME_MEMBER;
    return "StablehloReferenceDevice";
  }

  PjRtLocalDeviceId local_device_id() const override {
    TRACE_ME_MEMBER;
    return PjRtLocalDeviceId(local_hardware_id().value());
  }

  PjRtLocalHardwareId local_hardware_id() const override {
    TRACE_ME_MEMBER;
    return PjRtLocalHardwareId(description_.local_hardware_id());
  }

  PjRtClient* client() const override {
    TRACE_ME_MEMBER;
    return client_;
  }

  bool IsAddressable() const override {
    TRACE_ME_MEMBER;
    return process_index() == client()->process_index();
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override {
    TRACE_ME_MEMBER;
    return UNIMPLEMENTED(TransferToInfeed);
  }

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    TRACE_ME_MEMBER;
    return UNIMPLEMENTED(TransferFromOutfeed);
  }

  void AttachDefaultMemorySpace(PjRtMemorySpace* memory_space) {
    TRACE_ME_MEMBER;
    memory_space_ = memory_space;
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    TRACE_ME_MEMBER;
    return absl::MakeSpan(&memory_space_, 1);
  }

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override {
    TRACE_ME_MEMBER;
    if (!memory_space_)
      return absl::InternalError("Plugin memory space unset.");

    return memory_space_;
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    TRACE_ME_MEMBER;
    LOG(FATAL) << "Plugin does not implement CreateAsyncTrackingEvent.";
    return nullptr;
  }

  const PjRtDeviceDescription& description() const override {
    TRACE_ME_MEMBER;
    return description_;
  }

 private:
  PjRtClient* client_;
  PjRtMemorySpace* memory_space_;  // unpinned memory owned by client
  StablehloReferenceDeviceDescription description_;
};

// Device Description
std::unique_ptr<PjRtDeviceDescription> GetStablehloReferenceDeviceDescription(
    int process_id, int local_device_id) {
  return std::make_unique<StablehloReferenceDeviceDescription>(process_id,
                                                               local_device_id);
}

// Reference Device
std::unique_ptr<PjRtDevice> GetStablehloReferenceDevice(PjRtClient* client) {
  return std::make_unique<StablehloReferenceDevice>(client);
}

void AttachStablehloReferenceMemorySpace(PjRtDevice* device,
                                         PjRtMemorySpace* memory_space) {
  auto stablehlo_device = dynamic_cast<StablehloReferenceDevice*>(device);
  if (stablehlo_device == nullptr) {
    LOG(FATAL) << "Plugin cannot attach memory space to device of kind "
               << device->device_kind();
    return;
  }
  stablehlo_device->AttachDefaultMemorySpace(memory_space);
}

}  // namespace mlir::stablehlo
