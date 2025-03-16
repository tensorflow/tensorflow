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

#ifndef XLA_SERVICE_HOST_MEMORY_OFFLOAD_ANNOTATIONS_H_
#define XLA_SERVICE_HOST_MEMORY_OFFLOAD_ANNOTATIONS_H_

#include "absl/strings/string_view.h"

namespace xla {
namespace host_memory_offload_annotations {

// External annotations:
inline const absl::string_view kDevicePlacement = "annotate_device_placement";
inline const absl::string_view kMemoryTargetPinnedHost = "pinned_host";
inline const absl::string_view kMemoryTargetUnpinnedHost = "unpinned_host";
inline const absl::string_view kMemoryTargetDevice = "device";
inline const absl::string_view kMemoryTargetDeviceSram = "vmem";
inline const absl::string_view kMemoryTargetPinnedDevice = "pinned_device";

// Internal annotations:
inline const absl::string_view kMoveToHostCustomCallTarget = "MoveToHost";
inline const absl::string_view kMoveToDeviceCustomCallTarget = "MoveToDevice";
inline const absl::string_view kPinToDeviceCustomCallTarget = "PinToDevice";
inline const absl::string_view kPinToDeviceSramCustomCallTarget =
    "PinToDeviceSram";

}  // namespace host_memory_offload_annotations
}  // namespace xla

#endif  // XLA_SERVICE_HOST_MEMORY_OFFLOAD_ANNOTATIONS_H_
