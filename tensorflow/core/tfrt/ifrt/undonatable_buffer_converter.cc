/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/undonatable_buffer_converter.h"

#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/undonatable_common_pjrt_buffer.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

absl::StatusOr<std::shared_ptr<xla::PjRtBuffer>> MakeBufferUndonatable(
    const std::shared_ptr<xla::PjRtBuffer>& pjrt_buffer) {
  auto* common_pjrt_buffer =
      dynamic_cast<xla::CommonPjRtBuffer*>(pjrt_buffer.get());
  if (common_pjrt_buffer == nullptr) {
    LOG_FIRST_N(WARNING, 1)
        << "Loaded variable buffer is not backed by CommonPjRtBuffer; "
           "keeping donatable buffers.";
    return pjrt_buffer;
  }

  xla::CommonPjRtBuffer::ScopedHold hold =
      common_pjrt_buffer->GetBufferWithHold(
          xla::CommonPjRtBuffer::ScopedHold::kDonation);
  TF_RETURN_IF_ERROR(hold.status());

  // The raw buffer and definition events must be copied before
  // ConfirmDonation(), which releases the tracked buffer's reference to the
  // device memory.
  xla::PjRtRawBufferRef raw_buffer = hold.buffer()->raw_buffer();
  absl::InlinedVector<xla::PjRtDeviceEventRef, 2> definition_events(
      hold.buffer()->definition_events().begin(),
      hold.buffer()->definition_events().end());
  hold.ConfirmDonation();

  return std::make_shared<xla::UndonatableCommonPjRtBuffer>(
      std::make_shared<const xla::Shape>(common_pjrt_buffer->on_device_shape()),
      std::move(raw_buffer), std::move(definition_events),
      common_pjrt_buffer->memory_space());
}

}  // namespace

absl::Status MakeArrayBuffersUndonatable(xla::ifrt::Array* array) {
  if (array == nullptr) {
    return absl::InvalidArgumentError("array must not be null");
  }
  auto* pjrt_compatible_array =
      llvm::dyn_cast<xla::ifrt::PjRtCompatibleArray>(array);
  if (pjrt_compatible_array == nullptr) {
    LOG_FIRST_N(WARNING, 1)
        << "Loaded variable array is not a PjRt-compatible array; keeping "
           "donatable buffers.";
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(auto pjrt_buffers,
                      pjrt_compatible_array->mutable_pjrt_buffers());
  for (std::shared_ptr<xla::PjRtBuffer>& pjrt_buffer : pjrt_buffers) {
    TF_ASSIGN_OR_RETURN(pjrt_buffer, MakeBufferUndonatable(pjrt_buffer));
  }
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
