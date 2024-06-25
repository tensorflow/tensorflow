/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/hlo_module_util.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

namespace {

absl::Status ValidateResultShape(const Shape& client_shape,
                                 const Shape& result_shape) {
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(client_shape));
  if (!ShapeUtil::Compatible(client_shape, result_shape)) {
    return InvalidArgument(
        "Shape used to set computation result layout %s is not compatible "
        "with result shape %s",
        ShapeUtil::HumanStringWithLayout(client_shape),
        ShapeUtil::HumanString(result_shape));
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const Shape* const> argument_shapes,
    const ExecutionOptions* execution_options, int default_num_replicas,
    std::optional<int> num_threads, const AotCompilationOptions* aot_options) {
  auto config = std::make_unique<HloModuleConfig>(program_shape);
  ComputationLayout* computation_layout =
      config->mutable_entry_computation_layout();
  const int64_t argument_shapes_size = argument_shapes.size();
  if (program_shape.parameters_size() != argument_shapes_size) {
    return InvalidArgument("computation takes %d parameters, but %u given",
                           program_shape.parameters_size(),
                           argument_shapes.size());
  }
  for (int i = 0, end = argument_shapes.size(); i < end; ++i) {
    // Verify that shape of arguments matches the shape of the arguments in the
    // ProgramShape.
    if (!ShapeUtil::Compatible(*argument_shapes[i],
                               program_shape.parameters(i))) {
      return InvalidArgument(
          "Argument does not match shape of computation parameter %d: want "
          "%s, got %s",
          i, ShapeUtil::HumanString(program_shape.parameters(i)),
          ShapeUtil::HumanString(*argument_shapes[i]));
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            *argument_shapes[i]));
  }
  if (execution_options != nullptr &&
      execution_options->has_shape_with_output_layout()) {
    const Shape shape_with_output_layout(
        execution_options->shape_with_output_layout());
    TF_RETURN_IF_ERROR(
        ValidateResultShape(shape_with_output_layout, program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            shape_with_output_layout));
  } else {
    // If the result layout is not set, then choose the default.
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  if (execution_options != nullptr) {
    if (execution_options->num_replicas() > 0) {
      config->set_replica_count(execution_options->num_replicas());
    } else {
      config->set_replica_count(default_num_replicas);
    }
    if (execution_options->num_partitions() > 0) {
      config->set_num_partitions(execution_options->num_partitions());
    }
    config->set_use_spmd_partitioning(
        execution_options->use_spmd_partitioning());
    if (!execution_options->allow_spmd_sharding_propagation_to_parameters()
             .empty()) {
      config->set_allow_spmd_sharding_propagation_to_parameters(
          execution_options->allow_spmd_sharding_propagation_to_parameters());
    }
    if (!execution_options->allow_spmd_sharding_propagation_to_output()
             .empty()) {
      config->set_allow_spmd_sharding_propagation_to_output(
          execution_options->allow_spmd_sharding_propagation_to_output());
    }
    config->set_use_auto_spmd_partitioning(
        execution_options->use_auto_spmd_partitioning());
    config->set_auto_spmd_partitioning_mesh_shape(std::vector<int64_t>(
        execution_options->auto_spmd_partitioning_mesh_shape().begin(),
        execution_options->auto_spmd_partitioning_mesh_shape().end()));
    config->set_auto_spmd_partitioning_mesh_ids(std::vector<int64_t>(
        execution_options->auto_spmd_partitioning_mesh_ids().begin(),
        execution_options->auto_spmd_partitioning_mesh_ids().end()));
    config->set_deduplicate_hlo(execution_options->deduplicate_hlo());
    config->set_seed(execution_options->seed());
    config->set_launch_id(execution_options->launch_id());
    config->set_debug_options(execution_options->debug_options());
    if (execution_options->has_device_assignment()) {
      TF_ASSIGN_OR_RETURN(auto device_assignment,
                          DeviceAssignment::Deserialize(
                              execution_options->device_assignment()));
      config->set_static_device_assignment(*device_assignment);
    }
    config->set_alias_passthrough_params(
        execution_options->alias_passthrough_params());
    *config->mutable_fdo_profile() = execution_options->fdo_profile();
    config->set_device_memory_size(execution_options->device_memory_size());
  } else {
    config->set_replica_count(default_num_replicas);
    config->set_debug_options(GetDebugOptionsFromFlags());
  }

  if (num_threads.has_value()) {
    config->set_intra_op_parallelism_threads(*num_threads);
  }

  if (aot_options != nullptr) {
    config->set_matrix_unit_operand_precision(
        aot_options->matrix_unit_operand_precision());
    if (aot_options->fusion_config_collection() !=
        FusionConfigCollection::kOff) {
      config->set_fusion_config_collection(
          aot_options->fusion_config_collection());
      *config->mutable_fusion_config() = aot_options->fusion_config();
    }
  }

  return std::move(config);
}

void UpdateEntryComputationLayout(
    HloModule* module, DeviceShapeRepresentationFn shape_representation_fn,
    bool empty_tiles_only) {
  CHECK(shape_representation_fn != nullptr);
  auto update_shape = [&shape_representation_fn,
                       empty_tiles_only](Shape* shape) {
    ShapeUtil::ForEachMutableSubshape(
        shape, [&shape_representation_fn, empty_tiles_only](
                   Shape* subshape, const ShapeIndex& index) {
          if (subshape->IsArray() && subshape->has_layout()) {
            if (!empty_tiles_only ||
                (empty_tiles_only && subshape->layout().tiles().empty())) {
              *subshape = shape_representation_fn(*subshape);
            }
          }
        });
  };

  for (int i = 0; i < module->entry_computation_layout().parameter_count();
       i++) {
    Shape shape =
        module->entry_computation_layout().parameter_layout(i).shape();
    update_shape(&shape);
    *module->mutable_entry_computation_layout()->mutable_parameter_layout(i) =
        ShapeLayout(shape);
  }
  Shape shape = module->entry_computation_layout().result_layout().shape();
  update_shape(&shape);
  *module->mutable_entry_computation_layout()->mutable_result_layout() =
      ShapeLayout(shape);
}

}  // namespace xla
