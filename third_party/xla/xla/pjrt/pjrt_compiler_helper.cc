/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_compiler_helper.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompileMlirModule(
    mlir::ModuleOp module, const CompileOptionsProto& options_proto,
    const PjRtTopologyDescriptionProto& topology_proto) {
  TF_ASSIGN_OR_RETURN(CompileOptions compile_options,
                      CompileOptions::FromProto(options_proto));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtCompiler> compiler,
                      xla::GetCApiCompiler(topology_proto.platform_name()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtTopologyDescription> topology_description,
      compiler->DeserializePjRtTopologyDescription(
          topology_proto.SerializeAsString()));

  TF_ASSIGN_OR_RETURN(auto executable,
                      compiler->Compile(compile_options, module,
                                        *topology_description, nullptr));
  if (executable == nullptr) {
    return absl::InternalError(
        "PjRtCompiler::Compile() returned a nullptr, even with OK status.");
  }
  return executable;
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompileHloModule(
    const HloModuleProto& hlo_module_proto, const CompileOptionsProto& options,
    const PjRtTopologyDescriptionProto& topology_proto) {
  TF_ASSIGN_OR_RETURN(CompileOptions compile_options,
                      CompileOptions::FromProto(options));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtCompiler> compiler,
                      xla::GetCApiCompiler(topology_proto.platform_name()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtTopologyDescription> topology_description,
      compiler->DeserializePjRtTopologyDescription(
          topology_proto.SerializeAsString()));
  xla::XlaComputation xla_computation(hlo_module_proto);
  TF_ASSIGN_OR_RETURN(auto executable,
                      compiler->Compile(compile_options, xla_computation,
                                        *topology_description, nullptr));
  if (executable == nullptr) {
    return absl::InternalError(
        "PjRtCompiler::Compile() returned a nullptr, even with OK status.");
  }
  return executable;
}

}  // namespace xla
