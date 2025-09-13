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

#include "xla/pjrt/raw_buffer.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

std::vector<RegisterRawBufferFactory::FactoryFuncT>& GetFactoryFuncs() {
  static auto* const funcs =
      new std::vector<RegisterRawBufferFactory::FactoryFuncT>;
  return *funcs;
}

PjRtFuture<> CommonPjRtRawBuffer::CopyRawHostToDevice(const void* src,
                                                      int64_t offset,
                                                      int64_t transfer_size) {
  auto event = CopyRawHostToDeviceAndReturnEvent(src, offset, transfer_size);
  if (!event.ok()) {
    return PjRtFuture<>(event.status());
  }
  return (*event)->GetReadyFuture();
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
CommonPjRtRawBuffer::RemoveDynamicShapeMetadataIfPresent(
    const xla::Shape& logical_shape) {
  return absl::InvalidArgumentError(absl::StrCat(
      "Dynamic shapes are not supported for ", memory_space()->DebugString()));
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
CommonPjRtRawBuffer::Slice(int64_t offset, int64_t size) {
  TF_ASSIGN_OR_RETURN(auto results, MultiSlice({{offset, size}}));
  return results[0];
}

absl::StatusOr<std::vector<tsl::RCReference<CommonPjRtRawBuffer>>>
CommonPjRtRawBuffer::MultiSlice(absl::Span<const SliceInfo> slices) {
  return absl::UnimplementedError(absl::StrCat("Slicing is not supported for ",
                                               memory_space()->DebugString()));
}

PjRtFuture<> CommonPjRtRawBuffer::CopyRawDeviceToHost(void* dst, int64_t offset,
                                                      int64_t transfer_size) {
  auto event = CopyRawDeviceToHostAndReturnEvent(dst, offset, transfer_size);
  if (!event.ok()) {
    return PjRtFuture<>(event.status());
  }
  return (*event)->GetReadyFuture();
}

void CommonPjRtRawBuffer::ScheduleCopyTo(
    AsyncWorkRunner* async_work_runner,
    std::vector<tsl::RCReference<tsl::AsyncValue>> transfer_dependency_avs,
    tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer,
    tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
    tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
    ::tsl::AsyncValueRef<bool> allocation_event) {
  absl::Span<const tsl::RCReference<tsl::AsyncValue>> definition_events_span =
      transfer_dependency_avs;
  async_work_runner->ScheduleWhenReady(
      definition_events_span,
      [src_raw_buffer = tsl::FormRef(this),
       dst_raw_buffer = std::move(dst_raw_buffer),
       transfer_dependency_avs = std::move(transfer_dependency_avs),
       definition_event_promise = std::move(definition_event_promise),
       src_usage_event_promise = std::move(src_usage_event_promise),
       allocation_event = std::move(allocation_event)]() {
        for (const auto& av : transfer_dependency_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            auto status = *error;
            if (allocation_event) {
              allocation_event.SetError(status);
            }
            definition_event_promise->SetError(status);
            src_usage_event_promise->SetError(status);
            return;
          }
        }

        src_raw_buffer->CopyTo(
            std::move(dst_raw_buffer), std::move(definition_event_promise),
            std::move(src_usage_event_promise), std::move(allocation_event));
      });
}

absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>
PjRtRawBuffer::CreateRawAliasOfBuffer(PjRtBuffer* buffer) {
  for (auto* func : GetFactoryFuncs()) {
    auto res = (*func)(buffer);
    if (res.has_value()) {
      return *res;
    }
  }
  if (buffer == nullptr) {
    return absl::InvalidArgumentError("Cannot create view of null buffer.");
  }
  return absl::UnimplementedError(
      absl::StrCat("CreateRawAliasOfBuffer not implemented for: ",
                   buffer->client()->platform_version()));
}

RegisterRawBufferFactory::RegisterRawBufferFactory(
    RegisterRawBufferFactory::FactoryFuncT func) {
  GetFactoryFuncs().push_back(func);
}

}  // namespace xla
