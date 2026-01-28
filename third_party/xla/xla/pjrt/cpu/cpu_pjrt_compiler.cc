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
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

namespace {

// A dummy CpuCollectives implementation used for compilation.
class DummyCpuCollectives : public xla::cpu::CpuCollectives {
 public:
  absl::StatusOr<std::vector<std::unique_ptr<xla::Communicator>>>
  CreateCommunicators(const xla::CliqueKey& clique_key,
                      const std::optional<xla::CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Config& config) final {
    return absl::UnimplementedError(
        "DummyCpuCollectives::CreateCommunicators is not implemented");
  }
};

// Creates a PjRt CPU client from the given topology description.
//
absl::StatusOr<std::unique_ptr<xla::PjRtClient>>
CreatePjRtCpuClientFromTopology(
    const xla::PjRtTopologyDescription& topology_description) {
  xla::CpuClientOptions options;
  TF_ASSIGN_OR_RETURN(options.cpu_device_count,
                      topology_description.CoreCountOfDefaultTypePerProcess());
  CHECK_GE(*options.cpu_device_count, 1);
  auto cpu_topology_description =
      absl::down_cast<const CpuTopologyDescription*>(&topology_description);
  if (cpu_topology_description == nullptr) {
    return absl::InvalidArgumentError(
        "Topology description is not a CpuTopologyDescription");
  }
  options.topology = cpu_topology_description;
  // We need to provide `CpuCollectives` to be able to compile multi-host/-slice
  // CPU computations. The details of the collectives is not important because
  // the compilation only checks if any `CpuCollectives` exists.
  options.collectives = std::make_shared<DummyCpuCollectives>();
  return xla::GetXlaPjrtCpuClient(options);
}

template <typename T>
absl::StatusOr<std::unique_ptr<PjRtExecutable>> CompileInternal(
    const T& computation, CompileOptions options,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  TF_ASSIGN_OR_RETURN(auto cpu_client,
                      CreatePjRtCpuClientFromTopology(topology));

  return cpu_client->Compile(computation, options);
}

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtExecutable>> CpuPjRtCompiler::Compile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  return CompileInternal(computation, options, topology, client);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> CpuPjRtCompiler::Compile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  return CompileInternal(module, options, topology, client);
}

}  // namespace xla::cpu

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(pjrt_register_cpu_compiler, {
  std::unique_ptr<xla::PjRtCompiler> compiler =
      std::make_unique<xla::cpu::CpuPjRtCompiler>();
  PjRtRegisterCompiler(xla::CpuName(), std::move(compiler));
});
