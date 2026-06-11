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

#include "xla/pjrt/cpu/cpu_pjrt_compiler.h"

#include <memory>
#include <utility>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/stream_executor/platform/initialize.h"

namespace xla::cpu {
namespace {

absl::StatusOr<const CpuTopologyDescription*> GetCpuTopology(
    const PjRtTopologyDescription& topology) {
  if (topology.platform_id() != xla::CpuPlatformId()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid platform ID: expected CPU platform, got ",
                     topology.platform_name()));
  }
  return &absl::down_cast<const xla::CpuTopologyDescription&>(topology);
}

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtExecutable>> CpuPjRtCompiler::Compile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  ASSIGN_OR_RETURN(const CpuTopologyDescription* cpu_topology,
                   GetCpuTopology(topology));

  ASSIGN_OR_RETURN(
      auto executable,
      CompileCpuExecutable(computation, std::move(options), *cpu_topology));
  return std::unique_ptr<PjRtExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> CpuPjRtCompiler::Compile(
    CompileOptions options, MaybeOwningMlirModule module,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  ASSIGN_OR_RETURN(const CpuTopologyDescription* cpu_topology,
                   GetCpuTopology(topology));

  ASSIGN_OR_RETURN(auto executable,
                   CompileCpuExecutable(std::move(module), std::move(options),
                                        *cpu_topology));
  return std::unique_ptr<PjRtExecutable>(std::move(executable));
}

}  // namespace xla::cpu

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(pjrt_register_cpu_compiler, {
  std::unique_ptr<xla::PjRtCompiler> compiler =
      std::make_unique<xla::cpu::CpuPjRtCompiler>();
  PjRtRegisterDefaultCompiler(xla::CpuName(), std::move(compiler));
});
