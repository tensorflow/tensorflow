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

#include "xla/service/hlo_runner_interface.h"

#include "xla/service/hlo_parser.h"

namespace xla {

/*static*/ absl::StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::CreateModuleFromString(const absl::string_view hlo_string,
                                           const DebugOptions& debug_options) {
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return ParseAndReturnUnverifiedModule(hlo_string, config);
}

namespace {

// Creates an HloModule from the given proto.
absl::StatusOr<std::unique_ptr<HloModule>> HloProtoToModule(
    const HloProto& proto, const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      HloModule::CreateModuleConfigFromProto(proto.hlo_module(),
                                                             debug_options));
  TF_ASSIGN_OR_RETURN(auto module,
                      HloModule::CreateFromProto(proto.hlo_module(), config));
  return std::move(module);
}
template <class T>
std::vector<T*> MakePointerVector(absl::Span<T> input_vec) {
  std::vector<T*> output_pointers;
  output_pointers.reserve(input_vec.size());
  for (auto& input : input_vec) {
    output_pointers.push_back(&input);
  }
  return output_pointers;
}

}  // namespace

/*static*/ absl::StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromBinaryProtoFile(
    const std::string& filename, const DebugOptions& debug_options) {
  HloProto proto;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), filename, &proto));
  return HloProtoToModule(proto, debug_options);
}

/*static*/ absl::StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromTextProtoFile(
    const std::string& filename, const DebugOptions& debug_options) {
  HloProto proto;
  TF_RETURN_IF_ERROR(tsl::ReadTextProto(tsl::Env::Default(), filename, &proto));
  return HloProtoToModule(proto, debug_options);
}

/*static*/ absl::StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromHloTextFile(
    const std::string& filename, const DebugOptions& debug_options) {
  std::string hlo_string;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), filename, &hlo_string));
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return ParseAndReturnUnverifiedModule(hlo_string, config);
}

/*static*/ absl::StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromModuleBinaryProtofile(
    const std::string& filename, const DebugOptions& debug_options) {
  HloModuleProto module_proto;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), filename, &module_proto));

  TF_ASSIGN_OR_RETURN(
      HloModuleConfig module_config,
      HloModule::CreateModuleConfigFromProto(module_proto, debug_options));

  return HloModule::CreateFromProto(module_proto, module_config);
}

absl::StatusOr<Literal> HloRunnerInterface::Execute(
    std::unique_ptr<HloModule> module, absl::Span<const Literal> arguments,
    bool run_hlo_passes, ExecutionProfile* profile) {
  // Construct a vector of plain pointers for the arguments.
  auto argument_pointers = MakePointerVector<const Literal>(arguments);
  return Execute(
      /*module=*/std::move(module),
      /*arguments=*/argument_pointers,
      /*run_hlo_passes=*/run_hlo_passes,
      /*profile=*/profile);
}

absl::StatusOr<Literal> HloRunnerInterface::ExecuteWithBufferAssignment(
    std::unique_ptr<HloModule> module,
    const BufferAssignmentProto* buffer_assignment_proto,
    absl::Span<const Literal> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  // Construct a vector of plain pointers for the arguments.
  auto argument_pointers = MakePointerVector<const Literal>(arguments);
  return ExecuteWithBufferAssignment(
      /*module=*/std::move(module),
      /*buffer_assignment_proto=*/buffer_assignment_proto,
      /*arguments=*/argument_pointers,
      /*run_hlo_passes=*/run_hlo_passes,
      /*profile=*/profile);
}

absl::StatusOr<Literal> HloRunnerInterface::ExecuteWithExecutable(
    Executable* executable, absl::Span<const Literal> arguments,
    ExecutionProfile* profile) {
  // Construct a vector of plain pointers for the arguments.
  auto argument_pointers = MakePointerVector<const Literal>(arguments);
  return ExecuteWithExecutable(executable, argument_pointers, profile);
}

}  // namespace xla
