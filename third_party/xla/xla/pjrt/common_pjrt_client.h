/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_COMMON_PJRT_CLIENT_H_
#define XLA_PJRT_COMMON_PJRT_CLIENT_H_

#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"

namespace xla {

// A common base class for Pjrt clients based on raw buffers.
class CommonPjRtClient : public PjRtClient {
 public:
  using PjRtClient::PjRtClient;

  // A thread pool for dispatching background work.
  // TODO(parkers): make pure virtual and update all clients.
  virtual AsyncWorkRunner* async_work_runner() const { return nullptr; }

  // Computes the memory requirements for storing shape on memory_space.
  // TODO(parkers): make pure virtual and update all clients.
  virtual absl::StatusOr<int64_t> GetOnDeviceBytesCount(
      PjRtMemorySpace* memory_space, const xla::Shape& shape) const {
    return absl::UnimplementedError("GetOnDeviceBytesCount is not supported.");
  }

  // Allocates a raw buffer of a particular size after an optional
  // allocate_after.
  virtual absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
  AllocateRawBuffer(PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
                    tsl::AsyncValueRef<bool> allocate_after) {
    return absl::UnimplementedError("AllocateRawBuffer is not supported");
  }

  // Imports foreign memory as a raw buffer.
  virtual absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
  ImportForeignMemory(void* device_ptr,
                      absl::AnyInvocable<void() &&> on_delete_callback,
                      size_t on_device_bytes_count,
                      PjRtMemorySpace* memory_space) {
    return absl::UnimplementedError("ImportForeignMemory is not supported");
  }

  // Linearizes a literal into a raw buffer and returns a DeviceEvent
  // for when the linearization is complete.
  virtual absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeInto(
      const LiteralSlice& literal, const xla::Layout& layout,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
    return absl::UnimplementedError("LinearizeInto is not supported");
  }

  // Defines a pjrt buffer from a shape, raw_buffer and definition events.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> DefineBuffer(
      const Shape& on_device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
          definition_device_events,
      bool raw_buffer_is_mutable) {
    return absl::UnimplementedError("DefineBuffer is not supported");
  }

  // When calling APIs that take extra debug information, we may want
  // to omit this debug information if it is not going to be used.
  virtual bool event_tracking_enabled() { return false; }

  // Create a linked device-event and device-event-promise such that
  // setting an event into the event promise populates the device-event.
  virtual absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                                   tsl::RCReference<PjRtDeviceEvent>>>
  CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                           absl::string_view debug_info) {
    return absl::UnimplementedError(
        "CreateLinkedEventPromise is not supported");
  }

  // Registers the necessary debug information for an allocation event.
  // TODO(parkers): Once everything is unified this should be controlled
  // by a non-device-specific config instead of delegating this control
  // to a device-specific config.
  virtual tsl::AsyncValueRef<bool> CreateAllocationEventForTransfers(
      PjRtMemorySpace* memory_space,
      const std::optional<std::string>& debug_info);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  // Applies memory-space normalization logic on top of
  // GetTopologyDescription()->GetDefaultLayout() to select the default
  // device layout (if not provided).
  virtual absl::StatusOr<xla::Shape> MakeDefaultShapeForMemorySpace(
      PjRtMemorySpace* memory_space, xla::Shape shape,
      const xla::Layout* layout) const;

  virtual bool BufferFromHostBufferSupportsZeroCopy(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides, const Shape& shape,
      PjRtMemorySpace* memory_space, const Layout* device_layout) const {
    return false;
  }

  virtual absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  LinearizeHostBufferInto(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const xla::Shape& device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
    return absl::UnimplementedError("LinearizeHostBufferInto is not supported");
  }
};

}  // namespace xla

#endif  // XLA_PJRT_COMMON_PJRT_CLIENT_H_
