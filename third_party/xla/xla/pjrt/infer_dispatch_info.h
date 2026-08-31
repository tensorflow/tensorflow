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

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_layout.h"
#include "xla/shape.h"

namespace xla {

struct PjRtLoadedExecutableDispatchInfo {
  std::vector<Shape> parameter_device_shapes;
  std::shared_ptr<const Shape> output_device_shape;
  std::vector<int> parameter_memory_space_kind_ids;
  std::vector<int> output_memory_space_kind_ids;
  std::vector<PjRtDevice*> addressable_devices;
  std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
  std::shared_ptr<DeviceAssignment> device_assignment;
  std::vector<int> parameters_that_may_be_donated;
  std::vector<int64_t> input_buffer_sizes_in_bytes;
  // Executable shape information that is computable from the PjRtExecutable*.
  struct Extras {
    std::string name;
    int num_partitions;
    int num_replicas;
    absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
        parameter_layouts;
    absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
        output_layouts;
    std::optional<std::vector<OpSharding>> parameter_shardings;
    std::optional<std::vector<OpSharding>> output_shardings;
    std::vector<absl::string_view> parameter_memory_kinds;
    std::vector<absl::string_view> output_memory_kinds;
    absl::StatusOr<std::string> fingerprint;
    HloInputOutputAliasConfig input_output_alias_config;
  };
  struct InputHloSnapshotBits {
    xla::HloModuleProto hlo_module;
    xla::DebugOptions debug_options;
  };
  std::unique_ptr<InputHloSnapshotBits> input_hlo_snapshot_bits;
  std::unique_ptr<Extras> extras;
};

// Helper for extracting parameter shapes from GetParameterShapes.
std::vector<Shape> GetParameterShapes(const ComputationLayout& layout);

// Constructs PjRtLoadedExecutableDispatchInfo from both device lists
// and metadata extracted from the final HloModule.
absl::StatusOr<PjRtLoadedExecutableDispatchInfo> InferDispatchInfo(
    const PjRtTopologyDescription* topology,
    std::vector<Shape> parameter_device_shapes, Shape output_device_shape,
    const HloInputOutputAliasConfig& alias_config,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices,
    std::unique_ptr<PjRtLoadedExecutableDispatchInfo::Extras> extras,
    bool tuple_inputs,
    std::unique_ptr<PjRtLoadedExecutableDispatchInfo::InputHloSnapshotBits>
        input_hlo_snapshot_bits = nullptr);

// Constructs PjRtLoadedExecutableDispatchInfo from both device lists
// and metadata extracted from the input mlir::ModuleOp. This may fail if all
// information is not available yet.
absl::StatusOr<PjRtLoadedExecutableDispatchInfo> InferDispatchInfo(
    const PjRtTopologyDescription* topology, mlir::ModuleOp mlir_module,
    const CompileOptions& options,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs);

}  // namespace xla

#endif  // XLA_PJRT_INFER_DISPATCH_INFO_H_
