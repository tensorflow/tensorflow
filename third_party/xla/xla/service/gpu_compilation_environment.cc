/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "xla/service/compilation_environments.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {

// TODO(b/284274097): Create flags with default values when flags
// are moved from DebugOptions to GpuCompilationEnvironment.
std::unique_ptr<GpuCompilationEnvironment> CreateDefaultGpuCompEnv() {
  return std::make_unique<GpuCompilationEnvironment>();
}

namespace {

// Implement a CompilationEnvironment::ProcessNewEnvFn for
// GpuCompilationEnvironment, so that we can add GpuCompilationEnvironments
// to CompilationEnvironments.
//
// The implementation returns Default env if one doesn't exist already.
// NOLINTNEXTLINE
std::unique_ptr<tsl::protobuf::Message> ProcessNewGpuCompilationEnvironment(
    std::unique_ptr<tsl::protobuf::Message> env) {  // NOLINT
  if (!env) {
    return xla::CreateDefaultGpuCompEnv();
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
