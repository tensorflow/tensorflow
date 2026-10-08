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

#include "xla/pjrt/interpreter/interpreter_pjrt_compiler.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/interpreter/interpreter_executable.h"
#include "xla/pjrt/interpreter/interpreter_topology_description.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/stream_executor/platform/initialize.h"

namespace xla {
namespace {

absl::StatusOr<const InterpreterTopologyDescription*> GetInterpreterTopology(
    const PjRtTopologyDescription& topology) {
  if (topology.platform_id() != xla::InterpreterId()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid platform ID: expected Interpreter platform, got ",
                     topology.platform_name()));
  }
  return &absl::down_cast<const xla::InterpreterTopologyDescription&>(topology);
}

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
InterpreterPjRtCompiler::Compile(CompileOptions options,
                                 const XlaComputation& computation,
                                 const PjRtTopologyDescription& topology,
                                 PjRtClient* client) {
  ABSL_ASSIGN_OR_RETURN(const InterpreterTopologyDescription* interpreter_topology,
                   GetInterpreterTopology(topology));

  ABSL_ASSIGN_OR_RETURN(auto executable,
                   CompileInterpreterExecutable(computation, std::move(options),
                                                *interpreter_topology));
  return std::unique_ptr<PjRtExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
InterpreterPjRtCompiler::Compile(CompileOptions options,
                                 MaybeOwningMlirModule module,
                                 const PjRtTopologyDescription& topology,
                                 PjRtClient* client) {
  ABSL_ASSIGN_OR_RETURN(const InterpreterTopologyDescription* interpreter_topology,
                   GetInterpreterTopology(topology));

  ABSL_ASSIGN_OR_RETURN(auto executable, CompileInterpreterExecutable(
                                        std::move(module), std::move(options),
                                        *interpreter_topology));
  return std::unique_ptr<PjRtExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
InterpreterPjRtCompiler::DeserializePjRtTopologyDescription(
    const std::string& serialized_topology) {
  xla::PjRtTopologyDescriptionProto proto;
  if (!proto.ParseFromString(serialized_topology)) {
    return absl::InvalidArgumentError(
        "Failed to parse InterpreterTopologyDescription from string.");
  }
  return InterpreterTopologyDescription::FromProto(proto);
}

}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    pjrt_register_interpreter_compiler, {
      std::unique_ptr<xla::PjRtCompiler> compiler =
          std::make_unique<xla::InterpreterPjRtCompiler>();
      PjRtRegisterDefaultCompiler(xla::InterpreterName(), std::move(compiler));
    });
