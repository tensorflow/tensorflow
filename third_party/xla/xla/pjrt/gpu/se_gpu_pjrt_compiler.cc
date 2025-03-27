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

#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/layout_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/utils.h"
#include "xla/service/compiler.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/local_service_utils.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

bool IsGpuClient(const PjRtClient& client) {
  return client.platform_id() == CudaId() || client.platform_id() == RocmId() ||
         client.platform_id() == SyclId();
}

bool IsSameTopology(const PjRtTopologyDescription& topology1,
                    const PjRtTopologyDescription& topology2) {
  const StreamExecutorGpuTopologyDescription& gpu_topology1 =
      tensorflow::down_cast<const StreamExecutorGpuTopologyDescription&>(
          topology1);
  const StreamExecutorGpuTopologyDescription& gpu_topology2 =
      tensorflow::down_cast<const StreamExecutorGpuTopologyDescription&>(
          topology2);
  return gpu_topology1 == gpu_topology2;
}

absl::Status IsValidTopologyAndClientForCompile(
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  if (client == nullptr) {
    return absl::UnimplementedError(
        "SE:GPU compiler requires non-null client.");
  }
  if (!IsGpuClient(*client)) {
    return absl::InvalidArgumentError(
        "SE:GPU compiler requires a GPU PjRtClient.");
  }
  TF_ASSIGN_OR_RETURN(auto client_topology, client->GetTopologyDescription());

  if (!IsSameTopology(topology, *client_topology)) {
    return absl::UnimplementedError(
        "SE:GPU compiler requires the topology same as the one in the client.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<xla::Compiler>>
GetCompilerForDefaultGpuPlatform() {
  TF_ASSIGN_OR_RETURN(stream_executor::Platform * platform,
                      PlatformUtil::GetPlatform("gpu"));
  return Compiler::GetForPlatform(platform);
}

absl::StatusOr<std::unique_ptr<xla::Compiler>> GetCompilerForPlatform(
    std::optional<stream_executor::Platform::Id> platform_id) {
  if (!platform_id.has_value()) {
    return GetCompilerForDefaultGpuPlatform();
  }

  TF_ASSIGN_OR_RETURN(
      stream_executor::Platform * platform,
      stream_executor::PlatformManager::PlatformWithId(platform_id.value()));
  return Compiler::GetForPlatform(platform);
}

}  // namespace

StreamExecutorGpuCompiler::StreamExecutorGpuCompiler(
    stream_executor::Platform::Id platform_id)
    : requested_platform_id_(platform_id) {}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(CompileOptions options,
                                   const XlaComputation& computation,
                                   const PjRtTopologyDescription& topology,
                                   PjRtClient* client) {
  // We get the compiler here because doing so in the constructor might fail due
  // to static initialization order shenanigans. Also we can't fail construction
  // of this class because it's also statically constructed.
  // TODO(b/382417973): Use factories instead of static initialization of
  // singletons.
  TF_ASSIGN_OR_RETURN(auto gpu_compiler,
                      GetCompilerForPlatform(requested_platform_id_));

  CompileOptions input_options = options;
  if (!options.target_config) {
    if (client != nullptr) {
      TF_RETURN_IF_ERROR(IsValidTopologyAndClientForCompile(topology, client));
      return client->CompileAndLoad(computation, options);
    }
    const auto& gpu_topology =
        tensorflow::down_cast<const xla::StreamExecutorGpuTopologyDescription&>(
            topology);
    if (gpu_topology.target_config().has_value()) {
      options.target_config.emplace(
          Compiler::TargetConfig(*gpu_topology.target_config()));
    } else {
      return absl::UnimplementedError(
          "Compilation without client and without target_config specified is "
          "not implemented");
    }
  }
  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());
  std::vector<const Shape*> argument_layout_pointers;
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [](Shape shape) { return LayoutUtil::GetWithDefaultLayout(shape); },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> hlo_config,
                      GetHloModuleConfig(computation, argument_layout_pointers,
                                         options.executable_build_options));

  HloModuleProto hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(hlo_module_proto, *hlo_config));
  UpdateEntryComputationLayout(
      hlo_module.get(), std::bind(&Compiler::DefaultDeviceShapeRepresentation,
                                  gpu_compiler.get(), std::placeholders::_1));
  DumpHloModuleIfEnabled(*hlo_module, kBeforeOptimizationsDumpName);
  Compiler::CompileOptions opts;
  opts.target_config = options.target_config;

  AotCompilationOptions aot_options(gpu_compiler->PlatformId());
  aot_options.set_target_config(*options.target_config);
  aot_options.set_run_backend_only(
      options.executable_build_options.run_backend_only());

  const int num_replicas = hlo_module->config().replica_count();
  const int num_partitions = hlo_module->config().num_partitions();
  const std::string name = hlo_module->name();
  const std::string fingerprint = hlo_module->GetFingerprint128();
  const int num_outputs = hlo_module->result_shape().IsTuple()
                              ? hlo_module->result_shape().tuple_shapes_size()
                              : 1;
  auto unique_module_group =
      std::make_unique<HloModuleGroup>(std::move(hlo_module));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      gpu_compiler->CompileAheadOfTime(std::move(unique_module_group),
                                       aot_options));
  std::vector<std::vector<absl::string_view>> output_memory_kinds(1);
  output_memory_kinds[0].resize(num_outputs,
                                StreamExecutorGpuHbmMemorySpace::kKind);
  return std::make_unique<StreamExecutorExecutable>(
      std::move(input_options), std::move(aot_results), num_replicas,
      num_partitions, name, fingerprint, std::move(output_memory_kinds));
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(CompileOptions options,
                                   mlir::ModuleOp module,
                                   const PjRtTopologyDescription& topology,
                                   PjRtClient* client) {
  CompileOptions input_options = options;
  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false,
      /*use_shardy=*/false));
  return Compile(std::move(input_options), xla_computation, topology, client);
}
}  // namespace xla
