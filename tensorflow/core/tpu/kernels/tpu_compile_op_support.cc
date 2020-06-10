/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

namespace tensorflow {
namespace tpu {

using stream_executor::port::Status;
using stream_executor::port::StatusOr;
using xla::ComputationLayout;
using xla::DebugOptions;
using xla::DeviceAssignment;
using xla::HloModuleConfig;
using xla::HloSharding;
using xla::InvalidArgument;
using xla::ProgramShape;
using xla::Shape;
using xla::ShapeTree;
using xla::ShapeUtil;

Status ValidateResultShape(const Shape& client_shape,
                           const Shape& result_shape) {
  TF_RETURN_IF_ERROR(
      xla::ShapeUtil::ValidateShapeWithOptionalLayout(client_shape));
  if (!xla::ShapeUtil::Compatible(client_shape, result_shape)) {
    return InvalidArgument(
        "Shape used to set computation result layout %s is not compatible "
        "with result shape %s",
        xla::ShapeUtil::HumanStringWithLayout(client_shape),
        xla::ShapeUtil::HumanString(result_shape));
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
    const ProgramShape& program_shape, absl::Span<const Shape> argument_shapes,
    absl::optional<const Shape> result_layout,
    absl::optional<const DeviceAssignment> device_assignment, int replica_count,
    int num_partitions, const DebugOptions* debug_options, const int* seed,
    const int* launch_id, const bool* alias_passthrough_params,
    const xla::FusionConfigCollection* fusion_config_collection,
    const std::vector<std::vector<bool>>* fusion_config) {
  auto config = absl::make_unique<HloModuleConfig>(program_shape);
  ComputationLayout* computation_layout =
      config->mutable_entry_computation_layout();
  if (program_shape.parameters_size() != argument_shapes.size()) {
    return InvalidArgument("computation takes %d parameters, but %u given",
                           program_shape.parameters_size(),
                           argument_shapes.size());
  }
  for (int i = 0; i < argument_shapes.size(); ++i) {
    // Verify that shape of arguments matches the shape of the arguments in the
    // ProgramShape.
    if (!ShapeUtil::Compatible(argument_shapes[i],
                               program_shape.parameters(i))) {
      return InvalidArgument(
          "Argument does not match shape of computation parameter %d: want "
          "%s, got %s",
          i, ShapeUtil::HumanString(program_shape.parameters(i)),
          ShapeUtil::HumanString(argument_shapes[i]));
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            argument_shapes[i]));
  }

  if (result_layout.has_value()) {
    TF_RETURN_IF_ERROR(
        ValidateResultShape(result_layout.value(), program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            result_layout.value()));
  } else {
    // If the result layout is not set, then choose the default.
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  config->set_replica_count(replica_count);
  config->set_num_partitions(num_partitions);
  if (seed != nullptr) {
    config->set_seed(*seed);
  }
  if (launch_id != nullptr) {
    config->set_launch_id(*launch_id);
  }
  if (debug_options != nullptr) {
    config->set_debug_options(*debug_options);
  } else {
    config->set_debug_options(xla::GetDebugOptionsFromFlags());
  }

  // TODO(henrytan): set intra_op_parallelism_threads.
  // Reference:
  // tensorflow/compiler/xla/service/service.cc?l=324.

  if (device_assignment.has_value()) {
    config->set_static_device_assignment(device_assignment.value());
  }

  if (alias_passthrough_params != nullptr) {
    config->set_alias_passthrough_params(*alias_passthrough_params);
  }

  if (fusion_config_collection != nullptr && fusion_config != nullptr &&
      *fusion_config_collection != xla::FusionConfigCollection::kOff) {
    config->set_fusion_config_collection(*fusion_config_collection);
    *config->mutable_fusion_config() = *fusion_config;
  }

  return std::move(config);
}

StatusOr<std::unique_ptr<xla::HloModuleConfig>> CreateModuleConfig(
    const xla::ProgramShape& program_shape,
    absl::Span<const Shape> argument_shapes,
    absl::optional<const Shape> result_layout,
    absl::optional<const DeviceAssignment> device_assignment, int replica_count,
    int num_partitions, const DebugOptions* debug_options) {
  return CreateModuleConfig(program_shape, argument_shapes, result_layout,
                            device_assignment, replica_count, num_partitions,
                            debug_options, /*seed=*/nullptr,
                            /*launch_id=*/nullptr,
                            /*alias_passthrough_params=*/nullptr,
                            /*fusion_config_collection=*/nullptr,
                            /*fusion_config=*/nullptr);
}

ShapeTree<HloSharding> GetSubtree(
    const ShapeTree<HloSharding>& tuple_shape_tree, int element_index) {
  ShapeTree<HloSharding> element_shape_tree(
      xla::ShapeUtil::GetTupleElementShape(tuple_shape_tree.shape(),
                                           element_index),
      HloSharding::Replicate());

  xla::ShapeIndex src_index;
  src_index.push_back(element_index);
  element_shape_tree.CopySubtreeFrom(tuple_shape_tree, src_index, {});
  return element_shape_tree;
}

Shape GetPerDeviceShape(const Shape& shape, const HloSharding& sharding,
                        int64 device) {
  if (shape.IsTuple()) {
    ShapeTree<HloSharding> tuple_shape_tree = sharding.GetAsShapeTree(shape);
    std::vector<Shape> arg_shapes;
    for (int64 i = 0; i < xla::ShapeUtil::TupleElementCount(shape); ++i) {
      Shape element_shape = xla::ShapeUtil::GetTupleElementShape(shape, i);
      HloSharding element_sharding = tuple_shape_tree.element({i});
      if (element_shape.IsTuple()) {
        element_sharding = HloSharding::Tuple(GetSubtree(tuple_shape_tree, i));
      }
      if (element_sharding.UsesDevice(device)) {
        arg_shapes.push_back(
            GetPerDeviceShape(element_shape, element_sharding, device));
      }
    }
    return xla::ShapeUtil::MakeTupleShape(arg_shapes);
  }

  if (sharding.IsTileMaximal()) {
    return shape;
  }

  std::vector<int64> dimensions;
  std::vector<int64> offset = sharding.TileOffsetForDevice(shape, device);
  std::vector<int64> limit = sharding.TileLimitForDevice(shape, device);
  for (int64 i = 0; i < limit.size(); ++i) {
    dimensions.push_back(limit[i] - offset[i]);
  }
  if (shape.has_layout()) {
    return xla::ShapeUtil::MakeShapeWithLayout(shape.element_type(), dimensions,
                                               shape.layout().minor_to_major());
  }
  return xla::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

Status AddVariableUpdatesToCores(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const std::vector<ShardingAndIndex>& arg_core_mapping,
    std::vector<bool>* may_modify_variables,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes,
    std::vector<std::vector<std::pair<int, bool>>>* per_core_variable_indices) {
  // Add all variables to the corresponding core.
  may_modify_variables->resize(metadata.num_cores_per_replica(), false);
  int resource_update_pos = 0;
  for (int i = 0; i < metadata.args_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Arg& proto_arg = metadata.args(i);
    if (proto_arg.kind() == tpu::TPUCompileMetadataProto::Arg::VARIABLE) {
      const auto& sharding = proto_arg.sharding();
      bool updated = false;
      if (resource_update_pos < compilation_result.resource_updates.size()) {
        const XlaCompiler::ResourceUpdate& update =
            compilation_result.resource_updates[resource_update_pos];
        if (update.input_index == i) {
          updated = true;
          int pos = compilation_result.outputs.size() + resource_update_pos;
          xla::Shape shape = xla::ShapeUtil::GetTupleElementShape(
              compilation_result.xla_output_shape, pos);
          auto add_to_core = [&](int64 core, const xla::Shape& per_core_shape) {
            (*per_core_output_shapes)[core].push_back(per_core_shape);
            (*may_modify_variables)[core] =
                (*may_modify_variables)[core] || update.modified;
          };
          if (sharding.type() == xla::OpSharding::MAXIMAL) {
            add_to_core(sharding.tile_assignment_devices(0), shape);
          } else if (sharding.type() == xla::OpSharding::OTHER) {
            auto sharding_or =
                xla::HloSharding::FromProto(proto_arg.sharding());
            TF_RET_CHECK(sharding_or.ok());
            for (int64 core : proto_arg.sharding().tile_assignment_devices()) {
              xla::Shape per_core_shape =
                  GetPerDeviceShape(shape, sharding_or.ValueOrDie(), core);
              add_to_core(core, per_core_shape);
            }
          } else {
            TF_RET_CHECK(sharding.type() == xla::OpSharding::REPLICATED);
            for (int64 core = 0; core < metadata.num_cores_per_replica();
                 ++core) {
              add_to_core(core, shape);
            }
          }
          ++resource_update_pos;
        }
      }
      if (sharding.type() == xla::OpSharding::MAXIMAL) {
        (*per_core_variable_indices)[sharding.tile_assignment_devices(0)]
            .push_back(
                std::pair<int, bool>(arg_core_mapping[i].indices[0], updated));
      } else if (sharding.type() == xla::OpSharding::OTHER) {
        for (int core : sharding.tile_assignment_devices()) {
          (*per_core_variable_indices)[core].push_back(
              std::pair<int, bool>(arg_core_mapping[i].indices[core], updated));
        }
      } else {
        TF_RET_CHECK(sharding.type() == xla::OpSharding::REPLICATED);
        for (int64 core = 0; core < metadata.num_cores_per_replica(); ++core) {
          (*per_core_variable_indices)[core].push_back(
              std::pair<int, bool>(arg_core_mapping[i].indices[core], updated));
        }
      }
    }
  }
  return Status::OK();
}

Status ComputeOutputShapesForEachCore(
    const tpu::TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes) {
  for (int i = 0; i < metadata.retvals_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Retval& retval = metadata.retvals(i);
    TF_RET_CHECK(!compilation_result.outputs[i].is_constant)
        << "TPU compilation output " << i
        << " has a compile-time constant value. "
           "This should never happen.";

    xla::Shape shape = xla::ShapeUtil::GetTupleElementShape(
        compilation_result.xla_output_shape, i);
    auto add_shape_to_core = [&](int core, xla::Shape per_core_shape) {
      (*per_core_output_shapes)[core].push_back(std::move(per_core_shape));
    };
    if (retval.sharding().type() == xla::OpSharding::MAXIMAL) {
      add_shape_to_core(retval.sharding().tile_assignment_devices(0),
                        std::move(shape));
    } else if (retval.sharding().type() == xla::OpSharding::OTHER) {
      auto sharding_or = xla::HloSharding::FromProto(retval.sharding());
      TF_RET_CHECK(sharding_or.ok());
      for (int64 core : retval.sharding().tile_assignment_devices()) {
        xla::Shape per_core_shape =
            GetPerDeviceShape(shape, sharding_or.ValueOrDie(), core);
        add_shape_to_core(core, std::move(per_core_shape));
      }
    } else {
      TF_RET_CHECK(retval.sharding().type() == xla::OpSharding::REPLICATED)
          << "Not all of the constant tensors were consumed.";
      for (int core = 0; core < per_core_output_shapes->size(); ++core) {
        add_shape_to_core(core, shape);
      }
    }
  }
  return Status::OK();
}

Status CreateHloModules(
    const TPUCompileMetadataProto& metadata,
    const tensorflow::XlaCompiler::CompilationResult& compilation_result,
    const absl::optional<xla::DeviceAssignment>& device_assignment,
    std::vector<std::unique_ptr<xla::HloModule>>* hlo_modules) {
  TF_RET_CHECK(
      compilation_result.computation->proto().has_host_program_shape());

  auto debug_options = xla::DebugOptions();
  debug_options.set_xla_step_marker_location(metadata.step_marker_location());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModuleConfig> module_config,
      CreateModuleConfig(
          xla::ProgramShape(
              compilation_result.computation->proto().host_program_shape()),
          compilation_result.xla_input_shapes,
          compilation_result.xla_output_shape, device_assignment,
          metadata.num_replicas(), metadata.num_cores_per_replica(),
          &debug_options));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> hlo_module,
      xla::HloModule::CreateFromProto(compilation_result.computation->proto(),
                                      *module_config));
  DumpHloModuleIfEnabled(*hlo_module, "before_optimizations");
  hlo_modules->push_back(std::move(hlo_module));

  return Status::OK();
}

XlaCompilationResultProto SerializeCompilationResult(
    const XlaCompiler::CompilationResult& compilation_result) {
  XlaCompilationResultProto compilation_result_proto;
  for (int input_mapping : compilation_result.input_mapping) {
    compilation_result_proto.add_input_mappings(input_mapping);
  }

  for (const Shape& input_shape : compilation_result.xla_input_shapes) {
    *(compilation_result_proto.add_xla_input_shapes()) = input_shape.ToProto();
  }
  *(compilation_result_proto.mutable_xla_output_shape()) =
      compilation_result.xla_output_shape.ToProto();

  for (const XlaCompiler::OutputDescription& output_description :
       compilation_result.outputs) {
    auto* new_output = compilation_result_proto.add_outputs();
    new_output->set_type(output_description.type);
    output_description.shape.AsProto(new_output->mutable_shape());
    new_output->set_is_constant(output_description.is_constant);
    output_description.constant_value.AsProtoField(
        new_output->mutable_constant_value());
    new_output->set_input_index(output_description.input_index);
    new_output->set_is_tensor_list(output_description.is_tensor_list);
  }

  *compilation_result_proto.mutable_host_compute_metadata() =
      compilation_result.host_compute_metadata;

  for (const XlaCompiler::ResourceUpdate& resource_update :
       compilation_result.resource_updates) {
    auto* new_resource_update = compilation_result_proto.add_resource_updates();
    new_resource_update->set_input_index(resource_update.input_index);
    new_resource_update->set_type(resource_update.type);
    resource_update.shape.AsProto(new_resource_update->mutable_shape());
    new_resource_update->set_modified(resource_update.modified);
    for (const std::string& gradient_access :
         resource_update.tensor_array_gradients_accessed) {
      new_resource_update->mutable_tensor_array_gradients_accessed()->insert(
          {gradient_access, true});
    }
  }

  if (compilation_result.computation != nullptr) {
    *compilation_result_proto.mutable_computation() =
        compilation_result.computation->proto();
  }

  return compilation_result_proto;
}

StatusOr<TpuAotCompilationRequestProto> CreateTpuAotCompilationRequest(
    const xla::HloModuleGroup& module_group,
    const XlaCompiler::CompilationResult& compilation_result,
    const TPUCompileMetadataProto& metadata,
    const std::vector<std::vector<xla::Shape>>& per_core_arg_shapes,
    const std::vector<std::vector<xla::Shape>>& per_core_output_shapes,
    const std::vector<std::vector<std::pair<int, bool>>>&
        per_core_variable_indices,
    const absl::optional<xla::DeviceAssignment>& device_assignment) {
  VLOG(1) << "CreateTpuAotCompilationRequest.";
  TpuAotCompilationRequestProto aot_request;
  *(aot_request.mutable_hlo_module_group()) = module_group.ToProto();
  *(aot_request.mutable_metadata()) = metadata;
  if (device_assignment.has_value()) {
    xla::DeviceAssignmentProto device_assignment_proto;
    Status status = device_assignment->Serialize(&device_assignment_proto);
    if (!status.ok()) {
      return status;
    }
    *(aot_request.mutable_device_assignment()) = device_assignment_proto;
  }

  for (const auto& arg_shapes : per_core_arg_shapes) {
    auto* new_shape_list = aot_request.add_per_core_arg_shapes();
    for (const auto& arg_shape : arg_shapes) {
      *new_shape_list->add_shapes() = arg_shape.ToProto();
    }
  }

  for (const auto& output_shapes : per_core_output_shapes) {
    auto* new_shape_list = aot_request.add_per_core_output_shapes();
    for (const auto& output_shape : output_shapes) {
      *new_shape_list->add_shapes() = output_shape.ToProto();
    }
  }

  for (const auto& variable_indices : per_core_variable_indices) {
    auto* new_list = aot_request.add_per_core_variable_indices();
    for (const auto& variable_index : variable_indices) {
      auto* core_index = new_list->add_variable_indices();
      core_index->set_index(variable_index.first);
      core_index->set_updated(variable_index.second);
    }
  }

  XlaCompilationResultProto compilation_result_proto =
      SerializeCompilationResult(compilation_result);
  *aot_request.mutable_compilation_result() = compilation_result_proto;

  VLOG(1) << "TpuAotCompilationRequest:\n" << aot_request.DebugString();
  return aot_request;
}
}  // namespace tpu
}  // namespace tensorflow
