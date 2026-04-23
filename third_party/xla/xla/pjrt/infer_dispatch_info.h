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

#ifndef XLA_PJRT_INFER_DISPATCH_INFO_H_
#define XLA_PJRT_INFER_DISPATCH_INFO_H_

#include "absl/status/statusor.h"
#include "xla/pjrt/common_pjrt_client.h"

namespace xla {

// Constructs CommonPjRtLoadedExecutable::DispatchInfo from both device lists
// and metadata extracted from the final HloModule.
absl::StatusOr<CommonPjRtLoadedExecutable::DispatchInfo> InferDispatchInfo(
    CommonPjRtClient* client, const ComputationLayout& layout,
    const HloInputOutputAliasConfig& alias_config,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<CommonPjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs);

// Constructs CommonPjRtLoadedExecutable::DispatchInfo from both device lists
// and metadata extracted from the input mlir::ModuleOp. This may fail if all
// information is not available yet.
absl::StatusOr<CommonPjRtLoadedExecutable::DispatchInfo> InferDispatchInfo(
    CommonPjRtClient* client, mlir::ModuleOp mlir_module,
    const CompileOptions& options,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<CommonPjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs);

}  // namespace xla

#endif  // XLA_PJRT_INFER_DISPATCH_INFO_H_
