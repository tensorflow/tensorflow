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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_runtime_abi_version.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
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
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace {

bool IsGpuClient(const PjRtClient& client) {
  return IsGpuId(client.platform_id());
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
    PjRtPlatformId pjrt_platform_id, stream_executor::Platform::Id platform_id)
    : requested_platform_id_(platform_id),
      pjrt_platform_id_(pjrt_platform_id) {}

StreamExecutorGpuCompiler::StreamExecutorGpuCompiler(
    PjRtPlatformId pjrt_platform_id, std::unique_ptr<Compiler> compiler)
    : compiler_(std::move(compiler)), pjrt_platform_id_(pjrt_platform_id) {}

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

namespace {
// Returns a `GpuTopology` populated with a target config obtained from the
// given parameters. Will fail if no target_config is found.
absl::StatusOr<GpuTopology> GetTopologyWithTargetConfig(
    const PjRtTopologyDescription& topology, const CompileOptions& options) {
  const auto gpu_topology_description =
      dynamic_cast<const xla::StreamExecutorGpuTopologyDescription*>(&topology);
  if (gpu_topology_description == nullptr) {
    return absl::InvalidArgumentError(
        "The PjRtTopologyDescription must be a "
        "StreamExecutorGpuTopologyDescription.");
  }
  // TODO: b/491510579 - The gpu_topology_description has 2 fields for a target
  // config, we should fold them and we can get rid of this if branch.
  if (gpu_topology_description->target_config().has_value()) {
    VLOG(2) << "Found GPU target config in PjRt topology description.";
    ASSIGN_OR_RETURN(gpu::GpuTargetConfig gpu_target_config,
                     Compiler::GpuTargetConfig::FromProto(
                         *gpu_topology_description->target_config()));
    return gpu_topology_description->gpu_topology().CopyWithNewTargetConfig(
        gpu_target_config);
  }
  if (gpu_topology_description->gpu_topology().has_gpu_target_config()) {
    VLOG(2) << "Found GPU target config in GpuTopology.";
    return gpu_topology_description->gpu_topology();
  }
  if (options.gpu_target_config.has_value()) {
    VLOG(2) << "Found GPU target config in compile options.";
    return gpu_topology_description->gpu_topology().CopyWithNewTargetConfig(
        options.gpu_target_config.value());
  }
  return absl::InvalidArgumentError(
      "No GPU target config found in topology description or compile options.");
}
}  // namespace

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client,
    LayoutCanonicalizationCallback layout_callback) {
  TF_ASSIGN_OR_RETURN(Compiler * gpu_compiler, GetOrCreateCompiler());

  // This function does a bunch of temporary modifications to the CompileOptions
  // which should not be reflected in the options that we keep with the
  // resulting executable. Therefore we make a copy here. `input_options` are
  // the options as provided by the caller.
  CompileOptions input_options = options;

  absl::StatusOr<GpuTopology> topology_with_target_config =
      GetTopologyWithTargetConfig(topology, options);
  if (!topology_with_target_config.ok() && client == nullptr) {
    // Note that we have code that depends on this being an UnimplementedError,
    // therefore we have this explicit early return here.
    return absl::UnimplementedError(absl::StrCat(
        "Compilation without client and without target config specified is not "
        "implemented. Details: ",
        topology_with_target_config.status().ToString()));
  }
  if (!topology_with_target_config.ok() && client != nullptr) {
    LOG(INFO) << "Found PjRtClient and no GPU target config. Performing a JIT "
                 "compilation. Details: "
              << topology_with_target_config.status();
    TF_RET_CHECK(IsGpuClient(*client))
        << "JIT compilation requires a GPU PjRt client.";
    TF_RETURN_IF_ERROR(IsValidTopologyAndClientForCompile(topology, client));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                        client->Compile(computation, input_options));
    return executable;
  }

  ASSIGN_OR_RETURN(GpuTopology xla_gpu_topology, topology_with_target_config);
  options.gpu_target_config = xla_gpu_topology.gpu_target_config();
  if (layout_callback != nullptr) {
    options.executable_build_options.set_layout_canonicalization_callback(
        std::move(layout_callback));
  }

  if (client != nullptr) {
    LOG(INFO) << "Found GPU target config and a PjRtClient. Performing a cross "
                 "compilation.";
  } else {
    LOG(INFO) << "Found GPU target config and no PjRtClient. Performing a "
                 "deviceless compilation.";
  }
  if (xla::IsEarlyExitCompilation(options)) {
    LOG(INFO) << "Early exit compilation is enabled. Note that this is always "
                 "a deviceless compilation.";
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
  aot_options.set_gpu_topology(xla_gpu_topology);
  aot_options.set_run_backend_only(
      options.executable_build_options.run_backend_only());
  if (IsEarlyExitCompilation(options)) {
    aot_options.set_early_exit_point(
        AotCompilationOptions::EarlyExitPoint::kAfterLayoutAssignment);
    aot_options.set_executor(nullptr);
  } else if (client != nullptr) {
    const StreamExecutorGpuClient* gpu_client =
        dynamic_cast<const StreamExecutorGpuClient*>(client);
    TF_RET_CHECK(gpu_client != nullptr)
        << "Given PjRtClient is not a StreamExecutorGpuClient.";
    aot_options.set_executor(
        gpu_client->client()->backend().default_stream_executor());
  }
  const int num_replicas = hlo_module->config().replica_count();
  const int num_partitions = hlo_module->config().num_partitions();
  const std::string name = hlo_module->name();
  const std::string fingerprint = hlo_module->GetFingerprint128();
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      gpu_compiler->CompileAheadOfTime(std::move(hlo_module), aot_options));
  return std::make_unique<StreamExecutorExecutable>(
      pjrt_platform_id_, std::move(input_options), std::move(aot_results),
      num_replicas, num_partitions, name, fingerprint,
      /*default_memory_kind=*/StreamExecutorGpuHbmMemorySpace::kKind);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(CompileOptions options,
                                   const XlaComputation& computation,
                                   const PjRtTopologyDescription& topology,
                                   PjRtClient* client) {
  return Compile(std::move(options), computation, topology, client,
                 /*layout_callback=*/nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
StreamExecutorGpuCompiler::Compile(CompileOptions options,
                                   MaybeOwningMlirModule module,
                                   const PjRtTopologyDescription& topology,
                                   PjRtClient* client) {
  absl::StatusOr<GpuTopology> topology_with_target_config =
      GetTopologyWithTargetConfig(topology, options);

  if (!topology_with_target_config.ok() && client != nullptr) {
    TF_RET_CHECK(IsGpuClient(*client))
        << "GPU compilation requires a GPU PjRt client.";
    TF_RETURN_IF_ERROR(IsValidTopologyAndClientForCompile(topology, client));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                        client->Compile(std::move(module), options));
    return executable;
  }

  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module.mlir_module(), xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false,
      /*exec_build_options=*/&options.executable_build_options,
      mlir::mhlo::getGpuChloToHighLevelMhloOptions()));

  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes,
                      GetArgLayoutModes(module.mlir_module()));
  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes,
                      GetOutputLayoutModes(module.mlir_module()));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces,
                      GetArgMemoryKinds(module.mlir_module()));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces,
                      GetOutputMemoryKinds(module.mlir_module()));

  // MLIR module no longer required - release any memory if owned.
  module = MaybeOwningMlirModule();

  const auto choose_compact_layout_for_shape =
      [](const Shape& shape) -> absl::StatusOr<Shape> {
    Shape compact_shape = LayoutUtil::GetWithDefaultLayout(shape);
    if (primitive_util::IsSubByteNonPredType(compact_shape.element_type())) {
      compact_shape.mutable_layout()->set_element_size_in_bits(
          primitive_util::BitWidth(compact_shape.element_type()));
    }
    return compact_shape;
  };

  // If auto-sharding modifies shapes of arguments and/or result,
  // we get a callback to restore the layouts. Let us restore the layouts
  // according to the attributes we parsed from MLIR.
  auto layout_callback = [&choose_compact_layout_for_shape, &arg_layout_modes,
                          &out_layout_modes, &arg_memory_spaces,
                          &out_memory_spaces](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    XlaComputation xla_computation(XlaComputation(module.ToProto()));
    return LayoutModesToXlaShapes(
        xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
        out_memory_spaces, choose_compact_layout_for_shape);
  };

  TF_ASSIGN_OR_RETURN(
      auto arg_layouts_and_pointers,
      LayoutModesToXla(xla_computation, arg_layout_modes, out_layout_modes,
                       arg_memory_spaces, out_memory_spaces,
                       choose_compact_layout_for_shape,
                       options.executable_build_options));

  options.argument_layouts = std::move(arg_layouts_and_pointers.first);
  return Compile(std::move(options), xla_computation, topology, client,
                 std::move(layout_callback));
}

absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>>
StreamExecutorGpuCompiler::GetTargetRuntimeAbiVersion() {
  ASSIGN_OR_RETURN(Compiler * compiler, GetOrCreateCompiler());
  ASSIGN_OR_RETURN(
      stream_executor::Platform * platform,
      stream_executor::PlatformManager::PlatformWithId(compiler->PlatformId()));
  ASSIGN_OR_RETURN(
      std::unique_ptr<stream_executor::RuntimeAbiVersion> runtime_abi_version,
      platform->GetRuntimeAbiVersion());
  return std::make_unique<StreamExecutorGpuPjRtRuntimeAbiVersion>(
      pjrt_platform_id_, std::move(runtime_abi_version));
}
}  // namespace xla
