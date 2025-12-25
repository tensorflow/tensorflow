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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/stream_executor/platform/initialize.h"

namespace xla {
namespace {

// CPU Compiler class wraps a CPU Client.
// In general  this is bad practice since we shouldn't require a device for
// compilation (GPU, TPU), but for CPU this is natural since we'll always
// require a CPU to do the compilation.
class CpuCompiler : public PjRtCompiler {
 public:
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override {
    auto compiler = xla::GetPjRtCpuClient({});
    return compiler->get()->Compile(computation, options);
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override {
    auto compiler = xla::GetPjRtCpuClient({});
    return compiler->get()->Compile(module, options);
  }

  absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
  DeserializePjRtTopologyDescription(
      const std::string& serialized_topology) override {
    PjRtTopologyDescriptionProto proto;
    if (!proto.ParseFromString(serialized_topology)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse PjRtTopologyDescriptionProto from "
                       "binary string of size: ",
                       serialized_topology.size()));
    }

    // Only allow unset or empty `{}` platform specific topology.
    if (proto.has_platform_specific_topology() &&
        !proto.platform_specific_topology().type_url().empty()) {
      // TODO(b/442725339): Consider deleting CPU topology proto.
      // If it turns out we need this, we can support this using:
      // `CpuTopologyProto -> UnpackTo`.
      return absl::InvalidArgumentError(
          "CPU topology description should not have platform specific "
          "topology.");
    }
    return std::make_unique<CpuTopologyDescription>(
        proto.platform_id(), proto.platform_name(), proto.platform_version(),
        std::vector<xla::CpuTopology::CpuDevice>{},
        absl::Span<const std::string>{});
  }

 private:
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const CompileOptions& input_options, const CompileOptions& options,
      std::unique_ptr<xla::HloModule> hlo_module,
      const PjRtTopologyDescription& topology, PjRtClient* client);
};

}  // namespace
}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(pjrt_register_cpu_compiler, {
  std::unique_ptr<xla::PjRtCompiler> compiler =
      std::make_unique<xla::CpuCompiler>();
  PjRtRegisterCompiler(xla::CpuName(), std::move(compiler));
});
