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

#include "tensorflow/compiler/xla/service/hlo_runner_interface.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

namespace xla {

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::CreateModuleFromString(const absl::string_view hlo_string,
                                           const DebugOptions& debug_options) {
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return ParseAndReturnUnverifiedModule(hlo_string, config);
}

namespace {

// Creates an HloModule from the given proto.
StatusOr<std::unique_ptr<HloModule>> HloProtoToModule(
    const HloProto& proto, const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      HloModule::CreateModuleConfigFromProto(proto.hlo_module(),
                                                             debug_options));
  TF_ASSIGN_OR_RETURN(auto module,
                      HloModule::CreateFromProto(proto.hlo_module(), config));
  return std::move(module);
}

}  // namespace

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromBinaryProtoFile(
    const std::string& filename, const DebugOptions& debug_options) {
  HloProto proto;
  TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 filename, &proto));
  return HloProtoToModule(proto, debug_options);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromTextProtoFile(
    const std::string& filename, const DebugOptions& debug_options) {
  HloProto proto;
  TF_RETURN_IF_ERROR(
      tensorflow::ReadTextProto(tensorflow::Env::Default(), filename, &proto));
  return HloProtoToModule(proto, debug_options);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromHloTextFile(
    const std::string& filename, const DebugOptions& debug_options) {
  string hlo_string;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  filename, &hlo_string));
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return ParseAndReturnUnverifiedModule(hlo_string, config);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromModuleBinaryProtofile(
    const std::string& filename, const DebugOptions& debug_options) {
  HloModuleProto module_proto;
  TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 filename, &module_proto));

  TF_ASSIGN_OR_RETURN(
      HloModuleConfig module_config,
      HloModule::CreateModuleConfigFromProto(module_proto, debug_options));

  return HloModule::CreateFromProto(module_proto, module_config);
}

StatusOr<Literal> HloRunnerInterface::Execute(
    std::unique_ptr<HloModule> module, absl::Span<const Literal> arguments,
    bool run_hlo_passes, ExecutionProfile* profile) {
  // Construct a vector of plain pointers for the arguments.
  std::vector<const Literal*> argument_pointers;
  argument_pointers.reserve(arguments.size());
  for (const auto& argument : arguments) {
    argument_pointers.push_back(&argument);
  }
  return Execute(
      /*module=*/std::move(module),
      /*arguments=*/argument_pointers,
      /*run_hlo_passes=*/run_hlo_passes,
      /*profile=*/profile);
}

StatusOr<Literal> HloRunnerInterface::ExecuteWithExecutable(
    Executable* executable, absl::Span<const Literal> arguments,
    ExecutionProfile* profile) {
  // Construct a vector of plain pointers for the arguments.
  std::vector<const Literal*> argument_pointers;
  argument_pointers.reserve(arguments.size());
  for (const auto& argument : arguments) {
    argument_pointers.push_back(&argument);
  }
  return ExecuteWithExecutable(executable, argument_pointers, nullptr);
}

void HloRunnerInterface::UpdateEntryComputationLayout(
    HloModule* module, DeviceShapeRepresentationFn shape_representation_fn) {
  CHECK(shape_representation_fn != nullptr);
  // Make sure entry computation shapes are in device representation.
  for (int i = 0; i < module->entry_computation_layout().parameter_count();
       i++) {
    Shape shape =
        module->entry_computation_layout().parameter_layout(i).shape();
    *module->mutable_entry_computation_layout()->mutable_parameter_layout(i) =
        ShapeLayout(shape_representation_fn(shape));
  }
  *module->mutable_entry_computation_layout()->mutable_result_layout() =
      ShapeLayout(shape_representation_fn(
          module->entry_computation_layout().result_layout().shape()));
}

}  // namespace xla
