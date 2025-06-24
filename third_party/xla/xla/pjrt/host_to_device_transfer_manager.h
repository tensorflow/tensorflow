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

#ifndef XLA_PJRT_HOST_TO_DEVICE_TRANSFER_MANAGER_H_
#define XLA_PJRT_HOST_TO_DEVICE_TRANSFER_MANAGER_H_

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/common_pjrt_client.h"

namespace xla {

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
CreateAsyncHostToDeviceTransferManager(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space);

}  // namespace xla

#endif  // XLA_PJRT_HOST_TO_DEVICE_TRANSFER_MANAGER_H_
