/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_COMMON_H_
#define XLA_PJRT_PJRT_COMMON_H_

#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"
#include "xla/runtime/chip_id.h"  // IWYU pragma: export
#include "xla/runtime/device_id.h"  // IWYU pragma: export
#include "xla/runtime/process_id.h"  // IWYU pragma: export

namespace xla {

// bool comes before int64_t because when pybind11 tries to convert a Python
// object to a C++ type, it will try to convert it to the first type in the list
// of possible types that it can be converted to (b/309163973).
using PjRtValueType =
    std::variant<std::string, bool, int64_t, std::vector<int64_t>, float>;

xla::PjRtValueTypeProto PjRtValueTypeToProto(const PjRtValueType& value);

PjRtValueType PjRtValueTypeFromProto(const xla::PjRtValueTypeProto& value);

absl::string_view PjRtValueTypeCaseName(
    PjRtValueTypeProto::ValueCase value_case);

template <typename T>
constexpr absl::string_view PjRtTypeName() {
  if constexpr (std::is_same_v<T, bool>) {
    return "bool";
  } else if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return "int64_t";
  } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
    return "std::vector<int64_t>";
  } else if constexpr (std::is_same_v<T, std::string>) {
    return "string";
  } else {
    return "unknown";
  }
}

template <typename Id>
using PjRtIdContainer = absl::InlinedVector<Id, 4>;

template <typename Id>
PjRtIdContainer<Id> MakeContinuousIds(int start, int size) {
  PjRtIdContainer<Id> container;
  container.reserve(size);
  for (int i = 0; i < size; ++i) {
    container.push_back(Id(start + i));
  }
  return container;
}

using PjRtPlatformId = uint64_t;

// Helper struct for cross host transfers, returned by the callback from a call
// to PjRtBuffer::MakeCrossHostReceiveBuffers.
struct PjRtCrossHostRecvDescriptors {
  // This vector contains one serialized descriptor. (For usages outside of
  // MakeCrossHostReceiveBuffers, serialized_descriptors may have length > 1).
  // The descriptor should be transmitted to the sender and passed to a call to
  // src_buffer->CopyToRemoteDevice.
  absl::InlinedVector<std::string, 1> serialized_descriptors;
};
// Function that the client should call at the receiver if it needs to cancel a
// cross-host send, for example because the buffer that the remote host wanted
// to send is not available. The serialized descriptor should match one of the
// descriptors returned in a PjRtCrossHostRecvDescriptors. on_canceled will be
// called once cancellation is complete and indicates whether cancellation was
// successful or not.
//
// For each serialized_descriptor provided in a PjRtCrossHostRecvDescriptors,
// *either* the sending host must successfully complete a CopyToRemoteDevice
// for that descriptor, *or* the receiving host must cancel. If there is a
// duplicate (e.g., both send and cancel) then the system will be left in an
// undefined state. If there is no send or cancellation then the system will
// hang indefinitely.
using PjRtCrossHostSendCancelNotifier = std::function<void(
    absl::string_view serialized_descriptor, absl::Status reason,
    std::function<void(absl::Status)> on_canceled)>;
// State asynchronously returned by MakeCrossHostReceiveBuffers. "descriptors"
// will match the returned PjRtBuffer objects 1:1. Specifically, each PjRtBuffer
// returned by MakeCrossHostReceiveBuffers will have one
// PjRtCrossHostRecvDescriptors object containing it descriptor(s).
struct PjRtCrossHostRecvState {
  std::vector<PjRtCrossHostRecvDescriptors> descriptors;
  PjRtCrossHostSendCancelNotifier cancel_notifier;
};
using PjRtCrossHostRecvNotifier =
    std::function<void(absl::StatusOr<PjRtCrossHostRecvState>)>;

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMMON_H_
