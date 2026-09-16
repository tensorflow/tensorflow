/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_UNDONATABLE_COMMON_PJRT_BUFFER_H_
#define XLA_PJRT_UNDONATABLE_COMMON_PJRT_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A PjRtBuffer for serving/inference workloads whose buffers are never
// donated: written once (host-to-device), read by executions, then deleted.
// Intended for serving paths (request inputs, loaded variables); not for
// execution outputs (`ToLiteral` is Unimplemented, see below).
//
// Unlike `CommonPjRtBuffer`, there is no ScopedHold bookkeeping and no usage
// events; no method blocks on another thread. Memory lifetime is enforced
// by reference counting alone: whoever obtains the raw buffer — via
// `AcquireRawBufferRef()` or an ExternalReference — must keep it alive until
// all initiated accesses (enqueued device work or host access) finish
// executing. Destroying the object drops only the buffer's own reference
// without waiting for readers; it defers the drop on unsignaled definition
// events, since the producer is the one consumer that may not hold its own
// reference. Early release via `Delete()` is a no-op. The class's
// own methods follow this rule internally by capturing the reference into their
// continuations.
//
// Because no usage events are recorded, the class cannot grant exclusive
// in-place access to its memory. Consequently, `DonateWithControlDependency()`,
// `Bitcast()` and `ReleaseDeviceMemoryOwnership()` currently return
// InvalidArgument. These could be supported without usage tracking: copying the
// underlying buffer provides exclusivity (though at a higher cost than
// `CommonPjRtBuffer`), or they could even alias the refcounted raw buffer since
// contents are immutable. `ToLiteral()`, `LazyToLiteral()`,
// `CopyToMemorySpace()`, and `CopyToRemoteDevice()` are Unimplemented, as
// serving never calls them on inputs.
//

// Thread-safety: all methods are safe to call concurrently; `mu_` guards
// the only mutable state (the cached ready future),
// everything else is immutable (`raw_buffer_` stays valid until destruction).
// As with any PjRtBuffer, destroying the object must be synchronized with all
// other method calls by the owner.
class UndonatableCommonPjRtBuffer : public PjRtBuffer {
 public:
  // Requires `raw_buffer` to be non-null. `raw_buffer_` remains valid until the
  // buffer object itself is destroyed; `Delete()` is a no-op.
  UndonatableCommonPjRtBuffer(
      std::shared_ptr<const Shape> on_device_shape, PjRtRawBufferRef raw_buffer,
      absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events,
      PjRtMemorySpace* memory_space);

  ~UndonatableCommonPjRtBuffer() override;

  static absl::StatusOr<std::unique_ptr<PjRtBuffer>> Create(
      std::unique_ptr<PjRtBuffer> buffer);

  // Hold-Free Inference Extensions

  // Acquires a reference to the underlying device memory that the caller MUST
  // hold until their enqueued device work completes.
  PjRtRawBufferRef AcquireRawBufferRef(
      const char* caller_name = "GetRawBufferNoHold") const;

  // Metadata Accessors

  const Shape& on_device_shape() const override { return *on_device_shape_; }
  PjRtMemorySpace* memory_space() const override { return memory_space_; }
  PjRtDevice* device() const override;
  PjRtClient* client() const override;
  absl::Span<const PjRtDeviceEventRef> definition_events() const {
    return definition_events_;
  }

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;
  absl::StatusOr<Shape> logical_on_device_shape() override;
  bool IsOnCpu() const override;

  // Lifecycle & Readiness

  Future<> GetReadyFuture() override ABSL_LOCKS_EXCLUDED(mu_);
  bool IsDeleted() const override { return false; };
  // Device memory is released when the object is destroyed (deferred on any
  // still-pending definition events while the producer may not hold its own
  // reference). Early release via `Delete()` is unsupported and the buffer
  // remains valid. Calling `Delete()` is a no-op and `IsDeleted()` always
  // returns false.
  void Delete() override {};

  // Data Transfers & External Interop

  Future<> CopyRawToHost(void* dst, int64_t offset,
                         int64_t transfer_size) override;
  Future<> CopyRawToHostFuture(Future<void*> dst, int64_t offset,
                               int64_t transfer_size) override;

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override ABSL_LOCKS_EXCLUDED(mu_);

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override
      ABSL_LOCKS_EXCLUDED(mu_);

  // Disabled / Unsupported Operations (Inference Safety Restrictions)

  // Mutation and donation operations are invariant violations to protect
  // external lifetime guarantees.
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DonateWithControlDependency(
      Future<> dependency) override;
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtBuffer* donated_dst) override;

  Future<> ToLiteral(MutableLiteralBase* literal) override;
  Future<> LazyToLiteral(
      absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) override;

  void CopyToRemoteDevice(Future<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override;

  // Bypasses `Bitcast` to prevent ownership transfers from invalidating
  // external usage leases prematurely.
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> Bitcast(
      xla::PrimitiveType element_type, absl::Span<const int64_t> dims,
      const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

 protected:
  std::shared_ptr<const Shape> on_device_shape_;
  mutable absl::Mutex mu_;
  Future<> definition_future_ ABSL_GUARDED_BY(mu_);
  PjRtMemorySpace* const memory_space_;

 private:
  PjRtRawBufferRef raw_buffer_;
  absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events_;
};

}  // namespace xla

#endif  // XLA_PJRT_UNDONATABLE_COMMON_PJRT_BUFFER_H_
