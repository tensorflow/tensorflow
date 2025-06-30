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

#ifndef XLA_PJRT_RAW_BUFFER_H_
#define XLA_PJRT_RAW_BUFFER_H_

#include <optional>

#include "absl/status/statusor.h"
#include "xla/literal.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

class PjRtMemorySpace;
class PjRtBuffer;

// Experimental. Don't use unless you know what you're doing.
// A raw buffer is an unsafe API for directly transferring into device
// memory while existing processes are consuming or mutating the same buffer.
class PjRtRawBuffer : public tsl::ReferenceCounted<PjRtRawBuffer> {
 public:
  virtual ~PjRtRawBuffer() = default;

  static absl::StatusOr<tsl::RCReference<PjRtRawBuffer>> CreateRawAliasOfBuffer(
      PjRtBuffer* buffer);

  // Memory space that the raw buffer lives on.
  virtual PjRtMemorySpace* memory_space() const = 0;

  // If visible to the host, returns the base pointer for direct access.
  virtual void* GetHostPointer() const { return nullptr; }

  // Returns the number of bytes of the buffer storage on the device.
  virtual size_t GetOnDeviceSizeInBytes() const = 0;

  // Transfers the buffer to a sub-range of the on-device representation.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `src` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  virtual PjRtFuture<> CopyRawHostToDevice(const void* src, int64_t offset,
                                           int64_t transfer_size) = 0;

  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `dst` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  virtual PjRtFuture<> CopyRawDeviceToHost(void* dst, int64_t offset,
                                           int64_t transfer_size) = 0;
};

// Adds methods common to all implementations of PjRtRawBuffer based on device
// events.
class CommonPjRtRawBuffer : public PjRtRawBuffer {
 public:
  // Transfers the buffer to a sub-range of the on-device representation.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `src` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  virtual absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  CopyRawHostToDeviceAndReturnEvent(const void* src, int64_t offset,
                                    int64_t transfer_size) = 0;

  PjRtFuture<> CopyRawHostToDevice(const void* src, int64_t offset,
                                   int64_t transfer_size) override;

  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `dst` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  virtual absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  CopyRawDeviceToHostAndReturnEvent(void* dst, int64_t offset,
                                    int64_t transfer_size) = 0;

  PjRtFuture<> CopyRawDeviceToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) override;

  // Creates an event which signals when the allocation is complete.
  virtual absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  MakeAllocationReadyEvent() = 0;

  // Slices out any dynamic shape information (if present).
  virtual absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
  RemoveDynamicShapeMetadataIfPresent(const xla::Shape& logical_shape);

  // Reads the dynamic shape for a raw buffer. output_shape must be a
  // constructed AsyncValueRef which will have its dimensions updated.
  virtual void ReadDynamicShape(tsl::AsyncValueRef<xla::Shape> output_shape,
                                xla::Shape shape) = 0;

  // Interprets buffer contents as having shape and linearizes these contents
  // async into the provided literal.
  virtual void CopyToLiteralAsync(
      PjRtFuture<>::Promise promise,
      tsl::RCReference<PjRtDeviceEventPromise> device_promise,
      MutableLiteralBase* literal, xla::Shape shape) = 0;

  // Copies directly into dst_raw_buffer. Must set definition_event_promise,
  // when dst_raw_buffer is ready, allocation_event before using dst_raw_buffer
  // and src_usage_event_promise when done using this buffer.
  virtual void CopyTo(
      tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer,
      tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
      tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
      ::tsl::AsyncValueRef<bool> allocation_event) = 0;
};

class RegisterRawBufferFactory {
 public:
  using FactoryFuncT =
      std::optional<absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>> (*)(
          PjRtBuffer* buffer);
  explicit RegisterRawBufferFactory(FactoryFuncT func);
};

#define REGISTER_PJRT_RAW_BUFFER_FACTORY(func) \
  static ::xla::RegisterRawBufferFactory __raw_buffer_factory(func)

}  // namespace xla

#endif  // XLA_PJRT_RAW_BUFFER_H_
