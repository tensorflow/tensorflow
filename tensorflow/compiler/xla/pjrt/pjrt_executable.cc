/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace {

void SetOptionOverride(OptionOverrideProto& option, const std::string& value) {
  option.set_string_field(value);
}

void SetOptionOverride(OptionOverrideProto& option, bool value) {
  option.set_bool_field(value);
}

}  // namespace

StatusOr<CompileOptionsProto> CompileOptions::ToProto() const {
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
  return output;
}

StatusOr<CompileOptions> CompileOptions::FromProto(
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
  for (auto& env_option_override : proto.env_option_overrides()) {
    switch (env_option_override.second.value_case()) {
      case OptionOverrideProto::kStringField:
        output.env_option_overrides.push_back(
            {env_option_override.first,
             std::variant<std::string, bool>(
                 env_option_override.second.string_field())});
        break;
      case OptionOverrideProto::kBoolField:
        output.env_option_overrides.push_back(
            {env_option_override.first,
             std::variant<std::string, bool>(
                 env_option_override.second.bool_field())});
        break;
      case OptionOverrideProto::VALUE_NOT_SET:
        return InternalError("OptionOverrideProto value not set.");
    }
  }
  return output;
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

Status CompileOptions::ApplyOption(const std::string& key,
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
      return OkStatus();
    } else if (xla_field->type() ==
                   tsl::protobuf::FieldDescriptor::TYPE_STRING &&
               std::holds_alternative<std::string>(value)) {
      reflection->SetString(&debug_options, xla_field,
                            std::get<std::string>(value));
      return OkStatus();
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

Status CompileOptions::ApplyAllOptionOverrides() {
  for (auto& option : env_option_overrides) {
    TF_RETURN_IF_ERROR(ApplyOption(option.first, option.second));
  }
  return OkStatus();
}

}  // namespace xla
