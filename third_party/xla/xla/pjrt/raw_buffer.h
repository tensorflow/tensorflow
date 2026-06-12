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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/staging_buffer.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

class PjRtMemorySpace;
class PjRtBuffer;

// Experimental. Don't use unless you know what you're doing.
// A raw buffer is an unsafe API for directly transferring into device
// memory while existing processes are consuming or mutating the same buffer.
class PjRtRawBuffer : public PJRT_RawBuffer,
                      public tsl::ReferenceCounted<PjRtRawBuffer> {
 public:
  PjRtRawBuffer();
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
  virtual Future<> CopyRawHostToDevice(const void* src, int64_t offset,
                                       int64_t transfer_size) = 0;

  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `dst` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  virtual Future<> CopyRawDeviceToHost(void* dst, int64_t offset,
                                       int64_t transfer_size) = 0;

 private:
  static const PJRT_RawBuffer_FunctionTable kRawBufferVtable;
};

class CommonPjRtRawBuffer;
using PjRtRawBufferRef = tsl::RCReference<CommonPjRtRawBuffer>;

// Adds methods common to all implementations of PjRtRawBuffer based on device
// events.
class CommonPjRtRawBuffer : public PjRtRawBuffer {
 public:
  // Return opaque device memory pointer to the underlying memory.
  virtual void* OpaqueDeviceMemoryDataPointer() const = 0;

  // Transfers the buffer to a sub-range of the on-device representation.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `src` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  virtual absl::StatusOr<PjRtDeviceEventRef> CopyRawHostToDeviceAndReturnEvent(
      const void* src, int64_t offset, int64_t transfer_size) = 0;

  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `dst` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  virtual absl::StatusOr<PjRtDeviceEventRef> CopyRawDeviceToHostAndReturnEvent(
      void* dst, int64_t offset, int64_t transfer_size) = 0;

  // Copies the buffer to a remote device.
  // The serialized_descriptor contains metadata about the buffer on the remote
  // device. The on_done callback is called when the transfer is complete or
  // on error. The transfer_dependency_avs are dependencies that must be
  // ready before the transfer can start. The returned PjRtDeviceEventRef is
  // ready when the transfer is complete or on error.
  using RemoteSendCallback =
      std::function<void(absl::Status status, bool sends_were_enqueued)>;
  virtual absl::StatusOr<PjRtDeviceEventRef> CopyRawToRemoteDevice(
      Future<std::string> serialized_descriptor, RemoteSendCallback on_done,
      PjRtDeviceEventRefVector transfer_dependency_avs) = 0;

  // A sliced buffer is a view into the offset and range of this buffer.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `offset`. Look at implementations of
  // this method for specific alignment requirements.
  virtual absl::StatusOr<PjRtRawBufferRef> Slice(int64_t offset,
                                                 int64_t size) = 0;

  struct SliceInfo {
    int64_t offset;
    int64_t size;
  };

  // Batched version of Slice(). May be faster on some implementations.
  virtual absl::StatusOr<std::vector<PjRtRawBufferRef>> MultiSlice(
      absl::Span<const SliceInfo> slices);

  // Creates an event which signals when the allocation is complete.
  virtual absl::StatusOr<PjRtDeviceEventRef> MakeAllocationReadyEvent() = 0;

  // Copies directly into dst_raw_buffer. Must set definition_event_promise,
  // when dst_raw_buffer is ready, allocation_event before using dst_raw_buffer
  // and src_usage_event_promise when done using this buffer.
  virtual void CopyTo(
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) = 0;

  // Blocks on a list of dependencies and then copies directly into
  // dst_raw_buffer. Must set definition_event_promise,
  // when dst_raw_buffer is ready, allocation_event before using dst_raw_buffer
  // and src_usage_event_promise when done using this buffer.
  virtual void ScheduleCopyTo(
      AsyncWorkRunner* async_work_runner,
      PjRtDeviceEventRefVector transfer_dependency_events,
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event);

  // Returns the async value associated with the buffer.
  virtual PjRtDeviceEventPtr GetRawBufferAsyncValue() = 0;

  virtual bool is_mutable() const { return true; }

  // TODO(parkers): This should not be needed, but some backends
  // require deleting after all events.
  virtual void DecrefAfter(PjRtDeviceEventRefVector avs);
};

tsl::AsyncValueRef<PjRtStagingBuffer> ToStagingBuffer(
    PjRtRawBufferRef raw_buffer, PjRtDeviceEventPromiseRef usage_promise,
    absl::FunctionRef<tsl::AsyncValueRef<PjRtStagingBuffer>(size_t,
                                                            PjRtMemorySpace*)>
        allocate_staging_buffer);

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
