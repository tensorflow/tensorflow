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

#include "xla/pjrt/undonatable_common_pjrt_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/device_event_utils.h"
#include "xla/pjrt/dynamic_shapes.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/xla_data.pb.h"

namespace xla {

UndonatableCommonPjRtBuffer::UndonatableCommonPjRtBuffer(
    std::shared_ptr<const Shape> on_device_shape, PjRtRawBufferRef raw_buffer,
    absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events,
    PjRtMemorySpace* memory_space)
    : on_device_shape_(std::move(on_device_shape)),
      memory_space_(memory_space),
      raw_buffer_(std::move(raw_buffer)),
      definition_events_(std::move(definition_events)) {
  CHECK(raw_buffer_.get() != nullptr) << "raw_buffer cannot be null";
}

UndonatableCommonPjRtBuffer::~UndonatableCommonPjRtBuffer() {
  PjRtDeviceEventRefVector events;
  events.reserve(definition_events_.size());
  for (const auto& ev : definition_events_) {
    if (!ev.async_value()->IsConcrete()) {
      events.push_back(ev);
    }
  }
  definition_events_.clear();
  // Defer dropping the reference to the raw buffer until all pending definition
  // events are satisfied to prevent premature reclamation. In-flight readers
  // are managed by external semantics holding direct references.
  raw_buffer_.release()->DecrefAfter(std::move(events));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> UndonatableCommonPjRtBuffer::Create(
    std::unique_ptr<PjRtBuffer> buffer) {
  if (!buffer) {
    return absl::InvalidArgumentError("buffer cannot be null");
  }
  auto* client = dynamic_cast<CommonPjRtClient*>(buffer->client());
  if (!client) {
    return absl::InvalidArgumentError(
        "MakeUndonatable requires a client derived from CommonPjRtClient");
  }
  return client->MakeUndonatable(std::move(buffer));
}

// Hold-Free Inference Extensions

PjRtRawBufferRef UndonatableCommonPjRtBuffer::AcquireRawBufferRef(
    const char* caller_name) const {
  return raw_buffer_;
}

// Metadata Accessors

PjRtDevice* UndonatableCommonPjRtBuffer::device() const {
  CHECK_EQ(memory_space_->devices().size(), 1);
  return absl::down_cast<PjRtDevice*>(memory_space_->devices()[0]);
}

PjRtClient* UndonatableCommonPjRtBuffer::client() const {
  return memory_space_->client();
}

absl::StatusOr<size_t> UndonatableCommonPjRtBuffer::GetOnDeviceSizeInBytes()
    const {
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client());
  return common_client->GetOnDeviceBytesCount(memory_space_, *on_device_shape_);
}

absl::StatusOr<Shape> UndonatableCommonPjRtBuffer::logical_on_device_shape() {
  Shape device_shape = *on_device_shape_;
  if (device_shape.is_static()) {
    StripMetadataForLogicalShape(device_shape);
    return device_shape;
  }
  return absl::UnimplementedError(
      "Dynamic logical_on_device_shape is not yet supported.");
}

bool UndonatableCommonPjRtBuffer::IsOnCpu() const {
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client());
  return common_client->IsOnCpu(memory_space_);
}

// Lifecycle & Readiness

Future<> UndonatableCommonPjRtBuffer::GetReadyFuture() {
  absl::MutexLock lock(&mu_);
  if (definition_future_) {
    return definition_future_;
  }

  // Lazily construct the readiness Future by chaining hardware allocation
  // signals and computational definition events. This runs exactly once
  // due to the outer mu_ lock.
  auto* common_client =
      absl::down_cast<CommonPjRtClient*>(memory_space_->client());
  auto [computed_promise, computed_future] = tsl::MakePromise<void>();
  common_client->TrackFuture(memory_space_, "BufferDefinitionEvent",
                             computed_future);

  PjRtDeviceEventRefVector dependencies;
  bool first_event_is_buffer_alloc = false;
  dependencies.reserve(definition_events_.size() + 1);

  if (common_client->include_raw_buffer_in_ready_event()) {
    PjRtDeviceEventPtr alloc_event = raw_buffer_->GetRawBufferAsyncValue();
    if (alloc_event && !alloc_event.async_value()->IsConcrete()) {
      first_event_is_buffer_alloc = true;
      dependencies.push_back(alloc_event.CopyRef());
    }
  }
  for (const auto& ev : definition_events_) {
    if (!ev.async_value()->IsConcrete()) {
      dependencies.push_back(ev);
    }
  }

  if (common_client->event_tracking_enabled()) {
    common_client->AddEventDependencies(
        memory_space_,
        PjRtDeviceEventPtr::FromAsyncValue(computed_future.async_value()),
        dependencies);
  }

  PjRtDeviceEventSpan deps_span(dependencies);
  xla::RunWhenReady(deps_span, [definition_event = std::move(computed_promise),
                                first_event_is_buffer_alloc,
                                dependencies =
                                    std::move(dependencies)]() mutable {
    absl::Status status;
    for (size_t i = 0; i < dependencies.size(); ++i) {
      const auto& e = dependencies[i];
      if (auto error = e.GetErrorIfPresent()) {
        if (i == 0 && first_event_is_buffer_alloc) {
          status.Update(absl::Status(
              absl::StatusCode::kFailedPrecondition,
              absl::StrCat("Error in buffer allocation: ", error->message())));
        } else {
          status.Update(*error);
        }
      }
    }
    definition_event.Set(std::move(status));
  });

  definition_future_ = std::move(computed_future);
  return definition_future_;
}

// Data Transfers & External Interop

Future<> UndonatableCommonPjRtBuffer::CopyRawToHost(void* dst, int64_t offset,
                                                    int64_t transfer_size) {
  return CopyRawToHostFuture(Future<void*>(dst), offset, transfer_size);
}

Future<> UndonatableCommonPjRtBuffer::CopyRawToHostFuture(
    Future<void*> dst, int64_t offset, int64_t transfer_size) {
  return Future<>(
      absl::UnimplementedError("CopyRawToHost is not yet supported."));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
UndonatableCommonPjRtBuffer::AcquireExternalReference() {
  return absl::UnimplementedError(
      "AcquireExternalReference is not yet supported.");
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
UndonatableCommonPjRtBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  return absl::UnimplementedError(
      "ReleaseDeviceMemoryOwnership is not yet supported.");
}

// Disabled / Unsupported Operations (Inference Safety Restrictions)

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
UndonatableCommonPjRtBuffer::DonateWithControlDependency(Future<> dependency) {
  return absl::InvalidArgumentError(
      "Donation is not supported by UndonatableCommonPjRtBuffer (inference "
      "only).");
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
UndonatableCommonPjRtBuffer::CopyToMemorySpace(PjRtBuffer* donated_dst) {
  return absl::InvalidArgumentError(
      "Donation is not supported by UndonatableCommonPjRtBuffer (inference "
      "only).");
}

Future<> UndonatableCommonPjRtBuffer::ToLiteral(MutableLiteralBase* literal) {
  return Future<>(absl::UnimplementedError(
      "ToLiteral is not supported on UndonatableCommonPjRtBuffer; use "
      "CopyRawToHost or use the standard CommonPjRtBuffer instead"));
}

Future<> UndonatableCommonPjRtBuffer::LazyToLiteral(
    absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) {
  return Future<>(absl::UnimplementedError(
      "LazyToLiteral is not supported on UndonatableCommonPjRtBuffer; use "
      "CopyRawToHost or use the standard CommonPjRtBuffer instead"));
}

void UndonatableCommonPjRtBuffer::CopyToRemoteDevice(
    Future<std::string> serialized_descriptor, RemoteSendCallback on_done) {
  on_done(absl::UnimplementedError(
              "CopyToRemoteDevice is not supported on "
              "UndonatableCommonPjRtBuffer; use the standard CommonPjRtBuffer "
              "instead"),
          /*sends_were_enqueued=*/false);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
UndonatableCommonPjRtBuffer::Bitcast(xla::PrimitiveType element_type,
                                     absl::Span<const int64_t> dims,
                                     const Layout* device_layout) {
  return absl::UnimplementedError(
      "Bitcast is not supported on UndonatableCommonPjRtBuffer because "
      "Bitcast donates the source buffer according to the PJRT contract, which "
      "conflicts with external usage holds and serving lifetimes.");
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
UndonatableCommonPjRtBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  return absl::UnimplementedError(
      "CopyToMemorySpace is not supported on UndonatableCommonPjRtBuffer; "
      "use the standard CommonPjRtBuffer instead");
}

// Downstream alias factory enabling dispatchers (e.g., IFRT) to extract the
// raw hardware handle, bypassing heavyweight PJRT holding abstractions.
static std::optional<absl::StatusOr<PjRtRawBufferRef>>
UndonatableCommonPjRtBuffer_CreateRawAliasOfBuffer(PjRtBuffer* buffer) {
  if (auto* undonatable_buffer =
          dynamic_cast<UndonatableCommonPjRtBuffer*>(buffer)) {
    return undonatable_buffer->AcquireRawBufferRef();
  }
  return std::nullopt;
}

REGISTER_PJRT_RAW_BUFFER_FACTORY(
    UndonatableCommonPjRtBuffer_CreateRawAliasOfBuffer);

}  // namespace xla
