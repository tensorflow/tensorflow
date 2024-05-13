/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu_compilation_environment.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "xla/parse_flags_from_env.h"
#include "xla/service/compilation_environments.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {

void InitializeFlagsForGpuCompEnv(std::vector<tsl::Flag>* flag_list,
                                  GpuCompilationEnvironment* gpu_comp_env) {
  auto int64_setter_for =
      [gpu_comp_env](
          void (GpuCompilationEnvironment::*member_setter)(int64_t)) {
        return [gpu_comp_env, member_setter](int64_t value) {
          (gpu_comp_env->*member_setter)(value);
          return true;
        };
      };
  flag_list->push_back(tsl::Flag(
      "dummy_flag",
      int64_setter_for(&GpuCompilationEnvironment::set_dummy_flag),
      gpu_comp_env->dummy_flag(), "Dummy flag to demonstrate the flow"));
}

absl::StatusOr<GpuCompilationEnvironment> CreateGpuCompEnvFromFlagStrings(
    std::vector<std::string>& flags, bool strict) {
  GpuCompilationEnvironment gpu_comp_env;
  std::vector<tsl::Flag> flag_objects;
  InitializeFlagsForGpuCompEnv(&flag_objects, &gpu_comp_env);
  bool result = tsl::Flags::Parse(flags, flag_objects);
  if (!result || (strict && !flags.empty())) {
    return InvalidArgument("Could not parse flags: %s",
                           absl::StrJoin(flags, ", "));
  }
  return gpu_comp_env;
}

absl::StatusOr<GpuCompilationEnvironment> CreateGpuCompEnvFromEnvVar() {
  GpuCompilationEnvironment env;
  std::vector<tsl::Flag> flag_objects;
  InitializeFlagsForGpuCompEnv(&flag_objects, &env);
  bool result = ParseFlagsFromEnvAndIgnoreUnknown("XLA_FLAGS", flag_objects);
  if (!result) {
    return InvalidArgument("Could not parse XLA_FLAGS.");
  }
  return env;
}

GpuCompilationEnvironment CreateGpuCompEnvWithDefaultValues() {
  GpuCompilationEnvironment env;
  env.set_dummy_flag(1);
  return env;
}

Status InitializeMissingFieldsFromXLAFlags(GpuCompilationEnvironment& env) {
  TF_ASSIGN_OR_RETURN(GpuCompilationEnvironment from_env,
                      CreateGpuCompEnvFromEnvVar());

  auto default_env = CreateGpuCompEnvWithDefaultValues();

  auto reflection = env.GetReflection();
  auto reflection_from_env = from_env.GetReflection();
  auto descriptor = GpuCompilationEnvironment::descriptor();
  std::vector<const tsl::protobuf::FieldDescriptor*> missing_fields;

  for (int j = 0; j < descriptor->field_count(); ++j) {
    const tsl::protobuf::FieldDescriptor* field = descriptor->field(j);
    if (reflection->HasField(env, field) &&
        reflection_from_env->HasField(from_env, field)) {
      return InvalidArgument(
          "Flag %s is set in both XLA_FLAGS env var and "
          "GpuCompilationEnvironment.",
          field->name());
    } else if (!reflection->HasField(env, field) &&
               !reflection_from_env->HasField(from_env, field)) {
      missing_fields.push_back(field);
    }
  }
  env.MergeFrom(from_env);

  if (!missing_fields.empty()) {
    reflection->SwapFields(&env, &default_env, missing_fields);
  }
  return OkStatus();
}

namespace {

// Implement a CompilationEnvironment::ProcessNewEnvFn for
// GpuCompilationEnvironment, so that we can add GpuCompilationEnvironments
// to CompilationEnvironments.
//
// The implementation returns Empty env if one doesn't exist already.
// NOLINTNEXTLINE
absl::StatusOr<std::unique_ptr<tsl::protobuf::Message>>
ProcessNewGpuCompilationEnvironment(
    std::unique_ptr<tsl::protobuf::Message> env) {  // NOLINT
  if (!env) {
    env = std::make_unique<GpuCompilationEnvironment>();
  }
  return env;
}

}  // namespace

}  // namespace xla

static bool InitModule() {
  xla::CompilationEnvironments::RegisterProcessNewEnvFn(
      xla::GpuCompilationEnvironment::descriptor(),
      xla::ProcessNewGpuCompilationEnvironment);
  return true;
}
static bool module_initialized = InitModule();
