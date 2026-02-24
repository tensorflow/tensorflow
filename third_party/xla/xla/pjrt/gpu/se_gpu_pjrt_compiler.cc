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

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/utils.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/dump.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/local_service_utils.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

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
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<xla::Compiler>>
GetCompilerForDefaultGpuPlatform() {
  TF_ASSIGN_OR_RETURN(stream_executor::Platform * platform,
                      PlatformUtil::GetPlatform("gpu"));
  return Compiler::GetForPlatform(platform->id());
}

absl::StatusOr<std::unique_ptr<xla::Compiler>> GetCompilerForPlatform(
    std::optional<stream_executor::Platform::Id> platform_id) {
  if (!platform_id.has_value()) {
    return GetCompilerForDefaultGpuPlatform();
  }

  TF_ASSIGN_OR_RETURN(
      stream_executor::Platform * platform,
      stream_executor::PlatformManager::PlatformWithId(platform_id.value()));
  return Compiler::GetForPlatform(platform->id());
}

}  // namespace

StreamExecutorGpuCompiler::StreamExecutorGpuCompiler(
    stream_executor::Platform::Id platform_id)
    : requested_platform_id_(platform_id) {}

absl::StatusOr<Compiler*> StreamExecutorGpuCompiler::GetOrCreateCompiler() {
  absl::MutexLock lock(compiler_mutex_);
  if (compiler_ == nullptr) {
    // We get the compiler here because doing so in the constructor might fail
    // due to static initialization order shenanigans (An instance of this class
    // is initialized statically and this might happen before the compiler is
    // registered with Compiler::RegisterCompilerFactory). For the same reason,
    // we can't fail construction of this class, therefore we have this
    // GetOrCreate function and we can return on error when calling Compile.
    TF_ASSIGN_OR_RETURN(compiler_,
                        GetCompilerForPlatform(requested_platform_id_));
  }
  return compiler_.get();
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(CompileOptions options,
                                   const XlaComputation& computation,
                                   const PjRtTopologyDescription& topology,
                                   PjRtClient* client) {
  TF_ASSIGN_OR_RETURN(Compiler * gpu_compiler, GetOrCreateCompiler());

  CompileOptions input_options = options;
  if (xla::IsEarlyExitCompilation(options)) {
    auto* se_gpu_topology =
        tsl::down_cast<const xla::StreamExecutorGpuTopologyDescription*>(
            &topology);
    const xla::GpuTopology& gpu_topology = se_gpu_topology->gpu_topology();
    TF_RET_CHECK(gpu_topology.has_gpu_target_config())
        << "GPU cross-compile is not yet implemented for topology "
        << se_gpu_topology->ToProto()->ShortDebugString();
    options.gpu_target_config = gpu_topology.gpu_target_config();
  }
  if (!options.gpu_target_config) {
    if (client != nullptr) {
      TF_RET_CHECK(IsGpuClient(*client))
          << "GPU compilation requires a GPU PjRt client.";
      TF_RETURN_IF_ERROR(IsValidTopologyAndClientForCompile(topology, client));
      TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                          client->Compile(computation, options));
      return executable;
    }
    const auto& gpu_topology =
        tensorflow::down_cast<const xla::StreamExecutorGpuTopologyDescription&>(
            topology);
    if (gpu_topology.target_config().has_value()) {
      TF_ASSIGN_OR_RETURN(
          Compiler::GpuTargetConfig target_config,
          Compiler::GpuTargetConfig::FromProto(*gpu_topology.target_config()));
      options.gpu_target_config.emplace(std::move(target_config));
    } else {
      return absl::UnimplementedError(
          "Compilation without client and without target_config specified is "
          "not implemented");
    }
  }
  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [allow_auto_layout](Shape shape) {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return LayoutUtil::GetWithDefaultLayout(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> hlo_config,
                      GetHloModuleConfig(computation, argument_layout_pointers,
                                         options.executable_build_options));

  HloModuleProto hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(hlo_module_proto, *hlo_config));
  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_pjrt_allow_auto_layout_in_hlo(true);
  UpdateEntryComputationLayout(
      hlo_module.get(), std::bind(&Compiler::DefaultDeviceShapeRepresentation,
                                  gpu_compiler, std::placeholders::_1));
  DumpHloModuleIfEnabled(*hlo_module, kBeforeOptimizationsDumpName);

  AotCompilationOptions aot_options(gpu_compiler->PlatformId());
  GpuTopology xla_gpu_topology = GetSingleDeviceGpuTopology(
      /*platform_version=*/"", *options.gpu_target_config);
  aot_options.set_gpu_topology(xla_gpu_topology);
  aot_options.set_run_backend_only(
      options.executable_build_options.run_backend_only());
  if (IsEarlyExitCompilation(options)) {
    aot_options.set_early_exit_point(
        AotCompilationOptions::EarlyExitPoint::kAfterLayoutAssignment);
    aot_options.set_executor(nullptr);
  }
  const int num_replicas = hlo_module->config().replica_count();
  const int num_partitions = hlo_module->config().num_partitions();
  const std::string name = hlo_module->name();
  const std::string fingerprint = hlo_module->GetFingerprint128();
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      gpu_compiler->CompileAheadOfTime(std::move(hlo_module), aot_options));
  return std::make_unique<StreamExecutorExecutable>(
      std::move(input_options), std::move(aot_results), num_replicas,
      num_partitions, name, fingerprint,
      /*default_memory_kind=*/StreamExecutorGpuHbmMemorySpace::kKind);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(CompileOptions options,
                                   mlir::ModuleOp module,
                                   const PjRtTopologyDescription& topology,
                                   PjRtClient* client) {
  if (!options.gpu_target_config && client != nullptr) {
    TF_RET_CHECK(IsGpuClient(*client))
        << "GPU compilation requires a GPU PjRt client.";
    TF_RETURN_IF_ERROR(IsValidTopologyAndClientForCompile(topology, client));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                        client->Compile(module, options));
    return executable;
  }

  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false,
      /*exec_build_options=*/&options.executable_build_options,
      mlir::mhlo::getGpuChloToHighLevelMhloOptions()));
  return Compile(std::move(options), xla_computation, topology, client);
}
}  // namespace xla
