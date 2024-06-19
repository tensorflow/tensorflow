/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_executable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/layout.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/pjrt/execute_options.pb.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

void SetOptionOverride(OptionOverrideProto& option, const std::string& value) {
  option.set_string_field(value);
}

void SetOptionOverride(OptionOverrideProto& option, bool value) {
  option.set_bool_field(value);
}

void SetOptionOverride(OptionOverrideProto& option, int64_t value) {
  option.set_int_field(value);
}

void SetOptionOverride(OptionOverrideProto& option, double value) {
  option.set_double_field(value);
}

}  // namespace

absl::StatusOr<CompileOptionsProto> CompileOptions::ToProto() const {
  CompileOptionsProto output;
  if (argument_layouts.has_value()) {
    for (const auto& layout : *argument_layouts) {
      *output.add_argument_layouts() = layout.ToProto();
    }
  }
  output.set_parameter_is_tupled_arguments(parameter_is_tupled_arguments);
  TF_ASSIGN_OR_RETURN(*output.mutable_executable_build_options(),
                      executable_build_options.ToProto());
  output.set_compile_portable_executable(compile_portable_executable);
  output.set_profile_version(profile_version);
  if (multi_slice_config != nullptr) {
    output.set_serialized_multi_slice_config(multi_slice_config->Serialize());
  }
  for (auto& env_option_override : env_option_overrides) {
    auto& tmp =
        (*output.mutable_env_option_overrides())[env_option_override.first];
    std::visit([&](const auto& arg) { SetOptionOverride(tmp, arg); },
               env_option_override.second);
  }

  if (target_config.has_value()) {
    *output.mutable_target_config() = target_config->ToProto();
  }
  return output;
}

void CompileOptions::SerializeEnvOptionOverrides(
    google::protobuf::Map<std::string, xla::OptionOverrideProto>*
        output_env_option_overrides) const {
  for (auto& env_option_override : env_option_overrides) {
    auto& tmp = (*output_env_option_overrides)[env_option_override.first];
    std::visit([&](const auto& arg) { SetOptionOverride(tmp, arg); },
               env_option_override.second);
  }
}

absl::StatusOr<CompileOptions> CompileOptions::FromProto(
    const CompileOptionsProto& proto) {
  if (!proto.serialized_multi_slice_config().empty()) {
    return Unimplemented(
        "multi_slice_config not supported in CompileOptions::FromProto.");
  }

  CompileOptions output;
  if (proto.argument_layouts_size() > 0) {
    std::vector<Shape> output_argument_layouts;
    output_argument_layouts.reserve(proto.argument_layouts_size());
    for (const auto& argument_layout : proto.argument_layouts()) {
      output_argument_layouts.emplace_back(Shape(argument_layout));
    }
    output.argument_layouts = std::move(output_argument_layouts);
  }
  output.parameter_is_tupled_arguments = proto.parameter_is_tupled_arguments();
  TF_ASSIGN_OR_RETURN(
      ExecutableBuildOptions executable_build_options,
      ExecutableBuildOptionsFromProto(proto.executable_build_options()));
  output.executable_build_options = executable_build_options;
  output.compile_portable_executable = proto.compile_portable_executable();
  output.profile_version = proto.profile_version();
  TF_ASSIGN_OR_RETURN(output.env_option_overrides,
                      LoadEnvOptionOverrides(proto.env_option_overrides()));

  if (proto.has_target_config()) {
    output.target_config = xla::Compiler::TargetConfig(proto.target_config());
  }
  return output;
}

MultiSliceConfig::~MultiSliceConfig() = default;

absl::StatusOr<ExecuteOptionsProto> ExecuteOptions::ToProto() const {
  ExecuteOptionsProto proto;

  proto.set_arguments_are_tupled(arguments_are_tupled);
  proto.set_untuple_result(untuple_result);
  proto.set_launch_id(launch_id);
  if (context != nullptr) {
    return absl::UnimplementedError(
        "ExecuteOptions with non-nullptr context is not serializable");
  }
  proto.set_strict_shape_checking(strict_shape_checking);

  if (multi_slice_config != nullptr) {
    return absl::UnimplementedError(
        "ExecuteOptions with multi-slice config is not serializable");
  }

  if (!send_callbacks.empty() || !recv_callbacks.empty()) {
    return absl::UnimplementedError(
        "ExecuteOptions with send/recv calbacks is not serializable");
  }
  proto.set_use_major_to_minor_data_layout_for_callbacks(
      use_major_to_minor_data_layout_for_callbacks);

  switch (execution_mode) {
    case ExecutionMode::kDefault:
      proto.set_execution_mode(EXECUTION_MODE_DEFAULT);
      break;
    case ExecutionMode::kSynchronous:
      proto.set_execution_mode(EXECUTION_MODE_SYNCHRONOUS);
      break;
    case ExecutionMode::kAsynchronous:
      proto.set_execution_mode(EXECUTION_MODE_ASYNCHRONOUS);
      break;
  }

  proto.mutable_non_donatable_input_indices()->Add(
      non_donatable_input_indices.begin(), non_donatable_input_indices.end());

  return proto;
}

absl::StatusOr<ExecuteOptions> ExecuteOptions::FromProto(
    const ExecuteOptionsProto& proto) {
  ExecuteOptions options;

  options.arguments_are_tupled = proto.arguments_are_tupled();
  options.untuple_result = proto.untuple_result();
  options.launch_id = proto.launch_id();
  options.strict_shape_checking = proto.strict_shape_checking();
  options.use_major_to_minor_data_layout_for_callbacks =
      proto.use_major_to_minor_data_layout_for_callbacks();

  switch (proto.execution_mode()) {
    case EXECUTION_MODE_DEFAULT:
      options.execution_mode = ExecutionMode::kDefault;
      break;
    case EXECUTION_MODE_SYNCHRONOUS:
      options.execution_mode = ExecutionMode::kSynchronous;
      break;
    case EXECUTION_MODE_ASYNCHRONOUS:
      options.execution_mode = ExecutionMode::kAsynchronous;
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unknown execution mode: ", proto.execution_mode()));
  }

  options.non_donatable_input_indices.insert(
      proto.non_donatable_input_indices().begin(),
      proto.non_donatable_input_indices().end());

  return options;
}

CompiledMemoryStatsProto CompiledMemoryStats::ToProto() {
  CompiledMemoryStatsProto proto;
  proto.set_generated_code_size_in_bytes(generated_code_size_in_bytes);
  proto.set_argument_size_in_bytes(argument_size_in_bytes);
  proto.set_output_size_in_bytes(output_size_in_bytes);
  proto.set_alias_size_in_bytes(alias_size_in_bytes);
  proto.set_temp_size_in_bytes(temp_size_in_bytes);
  proto.mutable_hlo_proto()->ParseFromString(serialized_hlo_proto);
  proto.set_host_generated_code_size_in_bytes(
      host_generated_code_size_in_bytes);
  proto.set_host_argument_size_in_bytes(host_argument_size_in_bytes);
  proto.set_host_output_size_in_bytes(host_output_size_in_bytes);
  proto.set_host_alias_size_in_bytes(host_alias_size_in_bytes);
  proto.set_host_temp_size_in_bytes(host_temp_size_in_bytes);
  return proto;
}

CompiledMemoryStats CompiledMemoryStats::FromProto(
    const CompiledMemoryStatsProto& proto) {
  CompiledMemoryStats stats;
  stats.generated_code_size_in_bytes = proto.generated_code_size_in_bytes();
  stats.argument_size_in_bytes = proto.argument_size_in_bytes();
  stats.output_size_in_bytes = proto.output_size_in_bytes();
  stats.alias_size_in_bytes = proto.alias_size_in_bytes();
  stats.temp_size_in_bytes = proto.temp_size_in_bytes();
  stats.serialized_hlo_proto = proto.hlo_proto().SerializeAsString();
  stats.host_generated_code_size_in_bytes =
      proto.host_generated_code_size_in_bytes();
  stats.host_argument_size_in_bytes = proto.host_argument_size_in_bytes();
  stats.host_output_size_in_bytes = proto.host_output_size_in_bytes();
  stats.host_alias_size_in_bytes = proto.host_alias_size_in_bytes();
  stats.host_temp_size_in_bytes = proto.host_temp_size_in_bytes();
  return stats;
}

// Recomputes the memory stats from allocations. Why recompute?
// Firstly, there are cases in which gpu::Executable inherits its allocations
// from elsewhere, and no buffer assignment is available.
// Secondly, exec->buffer_assignment()->GetStats() provides the statistics we
// want, but does not distinguish between device and host memory, and does
// not account for aliased memory.
void CompiledMemoryStats::PopulateBufferStatsFromAllocations(
    absl::Span<const BufferAllocation> allocs) {
  argument_size_in_bytes = 0;
  output_size_in_bytes = 0;
  temp_size_in_bytes = 0;
  alias_size_in_bytes = 0;
  host_argument_size_in_bytes = 0;
  host_output_size_in_bytes = 0;
  host_temp_size_in_bytes = 0;
  host_alias_size_in_bytes = 0;

  for (auto& alloc : allocs) {
    // All logical buffers assigned to a buffer allocation share a color.
    // With buffer assigner's default colorer the color happens to be the
    // memory space of the underlying HLO value. Callers may choose other
    // colorers, however, e.g.:
    // https://github.com/openxla/xla/blob/50c6489cb058881cc65622605c9c55029abebc5b/xla/service/gpu/compile_module_to_llvm_ir.cc#L152
    // Until buffer allocations provide a stronger guarantee about colors,
    // we sanity-check that the default coloring behavior was used.
    int64_t alloc_memory_space = -1;
    for (const auto& [value, _] : alloc.assigned_buffers()) {
      const HloPosition& defining_position = value->defining_position();
      int64_t memory_space = Layout::kDefaultMemorySpace;
      if (defining_position.shape().has_layout()) {
        memory_space = defining_position.shape().layout().memory_space();
      }
      if (alloc_memory_space == -1) {
        alloc_memory_space = memory_space;
      } else {
        CHECK(alloc_memory_space == memory_space &&
              "expected same memory space for all assignments in allocation");
      }
    }

    bool is_host = alloc_memory_space == Layout::kHostMemorySpace;
    int64_t size = alloc.size();
    if (alloc.is_entry_computation_parameter()) {
      if (is_host) {
        host_argument_size_in_bytes += size;
      } else {
        argument_size_in_bytes += size;
      }
      if (alloc.is_parameter_aliased_with_output()) {
        if (is_host) {
          host_alias_size_in_bytes += size;
        } else {
          alias_size_in_bytes += size;
        }
      }
    }
    if (alloc.maybe_live_out()) {
      if (is_host) {
        host_output_size_in_bytes += size;
      } else {
        output_size_in_bytes += size;
      }
    }
    if (alloc.IsPreallocatedTempBuffer()) {
      if (is_host) {
        host_temp_size_in_bytes += size;
      } else {
        temp_size_in_bytes += size;
      }
    }
  }
}

void GetOpSharding(std::vector<OpSharding>& out, const OpSharding& sharding) {
  if (sharding.type() == OpSharding::TUPLE) {
    for (const OpSharding& s : sharding.tuple_shardings()) {
      GetOpSharding(out, s);
    }
  } else {
    out.push_back(sharding);
  }
}

std::optional<std::vector<OpSharding>> PjRtExecutable::GetOutputShardings()
    const {
  auto modules = GetHloModules();
  if (!modules.ok() || (*modules).empty() ||
      !(*modules)[0]->has_spmd_output_sharding()) {
    return std::nullopt;
  }

  std::vector<OpSharding> out;
  GetOpSharding(out, (*modules)[0]->spmd_output_sharding().ToProto());
  return out;
}

std::optional<std::vector<OpSharding>> PjRtExecutable::GetParameterShardings()
    const {
  auto modules = GetHloModules();
  if (!modules.ok() || (*modules).empty() ||
      !(*modules)[0]->has_spmd_parameters_shardings()) {
    return std::nullopt;
  }

  std::vector<OpSharding> out;
  for (const auto& s : (*modules)[0]->spmd_parameters_shardings()) {
    GetOpSharding(out, s.ToProto());
  }
  return out;
}

absl::StatusOr<std::vector<Shape>> PjRtExecutable::GetOutputShapes() const {
  TF_ASSIGN_OR_RETURN(auto modules, GetHloModules());
  std::vector<Shape> output_shapes;
  output_shapes.reserve(modules.size());
  for (const auto& module : modules) {
    output_shapes.push_back(module->result_shape());
  }
  return output_shapes;
}

absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
PjRtExecutable::GetOutputElementTypes() const {
  TF_ASSIGN_OR_RETURN(auto output_shapes, GetOutputShapes());
  std::vector<std::vector<PrimitiveType>> output_element_types;
  output_element_types.reserve(output_shapes.size());
  for (int i = 0; i < output_shapes.size(); ++i) {
    const Shape& output_shape = output_shapes[i];
    std::vector<PrimitiveType> element_types;
    if (output_shape.IsTuple()) {
      const auto& tuple_shapes = output_shape.tuple_shapes();
      element_types.reserve(tuple_shapes.size());
      for (int j = 0; j < tuple_shapes.size(); ++j) {
        if (tuple_shapes[j].IsTuple()) {
          return Unimplemented(
              "GetOutputElementTypes() doesn't support programs with "
              "nested-tupled outputs.");
        }
        element_types.push_back(tuple_shapes[j].element_type());
      }
    } else {
      element_types.reserve(1);
      element_types.push_back(output_shape.element_type());
    }
    output_element_types.push_back(std::move(element_types));
  }
  return output_element_types;
}

absl::StatusOr<std::vector<std::vector<DimensionVector>>>
PjRtExecutable::GetOutputDimensions() const {
  TF_ASSIGN_OR_RETURN(auto output_shapes, GetOutputShapes());
  std::vector<std::vector<DimensionVector>> output_dimensions;
  output_dimensions.reserve(output_shapes.size());
  for (int i = 0; i < output_shapes.size(); ++i) {
    const Shape& output_shape = output_shapes[i];
    std::vector<DimensionVector> dimensions;
    if (output_shape.IsTuple()) {
      const auto& tuple_shapes = output_shape.tuple_shapes();
      dimensions.reserve(tuple_shapes.size());
      for (int j = 0; j < tuple_shapes.size(); ++j) {
        if (tuple_shapes[j].IsTuple()) {
          return Unimplemented(
              "GetOutputDimensions() doesn't support programs with "
              "nested-tupled outputs.");
        }
        dimensions.push_back(
            ShapeUtil::CreateDimensionVectorFromShape(tuple_shapes[j]));
      }
    } else {
      dimensions.reserve(1);
      dimensions.push_back(
          ShapeUtil::CreateDimensionVectorFromShape(output_shape));
    }
    output_dimensions.push_back(std::move(dimensions));
  }
  return output_dimensions;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtLayout>>>
PjRtExecutable::GetParameterLayouts() const {
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      GetHloModules());
  if (hlo_modules.size() > 1) {
    return Unimplemented(
        "PjRtExecutable::GetParameterLayouts doesn't support MPMD "
        "executables.");
  }
  if (hlo_modules.empty()) {
    return InvalidArgument(
        "PjRtExecutable::GetParameterLayouts: couldn't retrieve HLO module "
        "from executable.");
  }
  ComputationLayout comp_layout = hlo_modules[0]->entry_computation_layout();
  TF_ASSIGN_OR_RETURN(std::vector<Layout> layouts,
                      comp_layout.FlattenedParameterLayouts());
  std::vector<std::unique_ptr<PjRtLayout>> result;
  result.reserve(layouts.size());
  for (const Layout& layout : layouts) {
    result.push_back(std::make_unique<PjRtXlaLayout>(layout));
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtLayout>>>
PjRtExecutable::GetOutputLayouts() const {
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      GetHloModules());
  if (hlo_modules.size() > 1) {
    return Unimplemented(
        "PjRtExecutable::GetOutputLayouts doesn't support MPMD "
        "executables.");
  }
  if (hlo_modules.empty()) {
    return InvalidArgument(
        "PjRtExecutable::GetOutputLayouts: couldn't retrieve HLO module "
        "from executable.");
  }
  ComputationLayout comp_layout = hlo_modules[0]->entry_computation_layout();
  TF_ASSIGN_OR_RETURN(std::vector<Layout> layouts,
                      comp_layout.FlattenedResultLayouts());
  std::vector<std::unique_ptr<PjRtLayout>> result;
  result.reserve(layouts.size());
  for (const Layout& layout : layouts) {
    result.push_back(std::make_unique<PjRtXlaLayout>(layout));
  }
  return result;
}

absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
PjRtExecutableUtil::RunHloCostAnalysis(const PjRtExecutable& executable,
                                       HloCostAnalysis* hlo_cost_analysis) {
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> modules,
                      executable.GetHloModules());
  if (modules.empty()) {
    return NotFound(
        "Executable '%s' did not have an HloModule to generate "
        "cost analysis with.",
        executable.name());
  } else if (modules.size() > 1) {
    return Unimplemented(
        "GetCostAnalysis() doesn't support multiple program "
        "multiple data executables (from executable '%s').",
        executable.name());
  }
  return RunHloCostAnalysis(modules, hlo_cost_analysis);
}

absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
PjRtExecutableUtil::RunHloCostAnalysis(
    const std::vector<std::shared_ptr<xla::HloModule>>& hlo_modules,
    HloCostAnalysis* hlo_cost_analysis) {
  if (hlo_modules.empty()) {
    return NotFound("RunHloCostAnalysis called with empty hlo_modules");
  } else if (hlo_modules.size() > 1) {
    return Unimplemented(
        "GetCostAnalysis() doesn't support multiple program "
        "multiple data executables.");
  }

  TF_RETURN_IF_ERROR(
      hlo_modules[0]->entry_computation()->Accept(hlo_cost_analysis));

  // Return cost properties
  absl::flat_hash_map<std::string, PjRtValueType> ret;
  hlo_cost_analysis->properties().ForEach(
      [&](absl::string_view key, float val) { ret[key] = val; });
  return ret;
}

absl::StatusOr<
    std::vector<std::pair<std::string, CompileOptions::OptionOverride>>>
CompileOptions::LoadEnvOptionOverrides(
    const google::protobuf::Map<std::string, xla::OptionOverrideProto>&
        env_option_overrides) {
  std::vector<std::pair<std::string, CompileOptions::OptionOverride>> result;
  for (auto& env_option_override : env_option_overrides) {
    switch (env_option_override.second.value_case()) {
      case OptionOverrideProto::kStringField:
        result.push_back({env_option_override.first,
                          CompileOptions::OptionOverride(
                              env_option_override.second.string_field())});
        break;
      case OptionOverrideProto::kBoolField:
        result.push_back({env_option_override.first,
                          CompileOptions::OptionOverride(
                              env_option_override.second.bool_field())});
        break;
      case OptionOverrideProto::kIntField:
        result.push_back({env_option_override.first,
                          CompileOptions::OptionOverride(
                              env_option_override.second.int_field())});
        break;
      case OptionOverrideProto::kDoubleField:
        result.push_back({env_option_override.first,
                          CompileOptions::OptionOverride(
                              env_option_override.second.double_field())});
        break;
      case OptionOverrideProto::VALUE_NOT_SET:
        return Internal("OptionOverrideProto value not set.");
    }
  }
  return result;
}

absl::Status CompileOptions::ApplyOption(const std::string& key,
                                         const OptionOverride& value) {
  if (auto* xla_field = xla::DebugOptions::descriptor()->FindFieldByName(key)) {
    xla::DebugOptions& debug_options =
        *executable_build_options.mutable_debug_options();
    const tsl::protobuf::Reflection* reflection = debug_options.GetReflection();
    if (!reflection) {
      return InvalidArgument(
          "No reflection object associated with xla::DebugOptions.");
    }
    if (xla_field->type() == tsl::protobuf::FieldDescriptor::TYPE_BOOL &&
        std::holds_alternative<bool>(value)) {
      reflection->SetBool(&debug_options, xla_field, std::get<bool>(value));
      return absl::OkStatus();
    } else if (std::holds_alternative<std::string>(value)) {
      TF_RETURN_IF_ERROR(
          ApplyOptionFromString(xla_field, std::get<std::string>(value)));
      return absl::OkStatus();
    } else if (xla_field->type() ==
                   tsl::protobuf::FieldDescriptor::TYPE_INT32 &&
               std::holds_alternative<int64_t>(value)) {
      reflection->SetInt32(&debug_options, xla_field, std::get<int64_t>(value));
      return absl::OkStatus();
    } else if (xla_field->type() ==
                   tsl::protobuf::FieldDescriptor::TYPE_INT64 &&
               std::holds_alternative<int64_t>(value)) {
      reflection->SetInt64(&debug_options, xla_field, std::get<int64_t>(value));
      return absl::OkStatus();
    } else if (xla_field->type() ==
                   tsl::protobuf::FieldDescriptor::TYPE_FLOAT &&
               std::holds_alternative<double>(value)) {
      reflection->SetFloat(&debug_options, xla_field, std::get<double>(value));
      return absl::OkStatus();
    } else if (xla_field->type() ==
                   tsl::protobuf::FieldDescriptor::TYPE_DOUBLE &&
               std::holds_alternative<double>(value)) {
      reflection->SetDouble(&debug_options, xla_field, std::get<double>(value));
      return absl::OkStatus();
    } else {
      return InvalidArgument(
          "While setting option %s, '%s' is not a valid %s value.", key,
          std::visit([](auto&& arg) { return absl::StrCat(arg); }, value),
          xla_field->type_name());
    }
  } else {
    return InvalidArgument("No such compile option: '%s'", key);
  }
}

absl::Status CompileOptions::ApplyAllOptionOverrides() {
  for (auto& option : env_option_overrides) {
    TF_RETURN_IF_ERROR(ApplyOption(option.first, option.second));
  }
  return absl::OkStatus();
}

absl::Status CompileOptions::ApplyOptionFromString(
    const tsl::protobuf::FieldDescriptor* field, const std::string& value) {
  xla::DebugOptions& debug_options =
      *executable_build_options.mutable_debug_options();
  const tsl::protobuf::Reflection* reflection = debug_options.GetReflection();
  if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_STRING) {
    reflection->SetString(&debug_options, field, value);
    return absl::OkStatus();
  } else if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_INT32) {
    int int_value;
    if (absl::SimpleAtoi(value, &int_value)) {
      reflection->SetInt32(&debug_options, field, int_value);
      return absl::OkStatus();
    }
  } else if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_INT64) {
    int int_value;
    if (absl::SimpleAtoi(value, &int_value)) {
      reflection->SetInt64(&debug_options, field, int_value);
      return absl::OkStatus();
    }
  } else if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_FLOAT) {
    float float_value;
    if (absl::SimpleAtof(value, &float_value)) {
      reflection->SetFloat(&debug_options, field, float_value);
      return absl::OkStatus();
    }
  } else if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_BOOL) {
    bool bvalue = value == "True";
    if (value == "True" || value == "False") {
      reflection->SetBool(&debug_options, field, bvalue);
      return absl::OkStatus();
    }
  }
  return InvalidArgument(
      "While setting option %s, '%s' is not a valid %s value.", field->name(),
      value, field->type_name());
}

}  // namespace xla
