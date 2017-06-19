/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/compile_only_service.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/computation_tracker.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<CompileOnlyService>>
CompileOnlyService::NewService(perftools::gputools::Platform* platform) {
  ServiceOptions default_options;
  default_options.set_platform(platform);
  return NewService(default_options);
}

/* static */ StatusOr<std::unique_ptr<CompileOnlyService>>
CompileOnlyService::NewService(const ServiceOptions& options) {
  perftools::gputools::Platform* platform = options.platform();
  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Backend> compute_constant_backend,
                      CreateComputeConstantBackend());
  std::unique_ptr<CompileOnlyService> service(
      new CompileOnlyService(compiler, std::move(compute_constant_backend)));
  return std::move(service);
}

CompileOnlyService::CompileOnlyService(
    Compiler* compiler, std::unique_ptr<Backend> compute_constant_backend)
    : Service(/*backend=*/nullptr, std::move(compute_constant_backend)),
      compiler_(compiler) {
  runs_in_client_process_ = true;
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CompileOnlyService::CompileAheadOfTime(
    const tensorflow::gtl::ArraySlice<AotComputationInstance> computations,
    const AotCompilationOptions& options) {
  std::vector<std::unique_ptr<HloModule>> hlo_modules;
  for (const AotComputationInstance& instance : computations) {
    TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                        computation_tracker_.Resolve(instance.computation));
    VersionedComputationHandle versioned_handle =
        user_computation->GetVersionedHandle();

    // Dump computation proto state if flag is set.
    legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
    const string& directory_path = flags->xla_dump_computations_to;
    if (!directory_path.empty()) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<SessionModule> session_module,
          computation_tracker_.SnapshotComputation(versioned_handle.handle));
      string filename = tensorflow::strings::StrCat(
          "computation_", versioned_handle.handle.handle(), "__",
          session_module->entry().name(), "__version_",
          versioned_handle.version);
      TF_RETURN_IF_ERROR(Executable::DumpToDirectory(directory_path, filename,
                                                     *session_module));
    }

    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<const ProgramShape> program_shape,
        user_computation->ComputeProgramShape(versioned_handle.version));

    HloModuleConfig hlo_module_config(*program_shape);
    hlo_module_config.set_debug_options(
        legacy_flags::GetDebugOptionsFromFlags());
    auto* computation_layout =
        hlo_module_config.mutable_entry_computation_layout();
    if (flags->xla_hlo_profile) {
      hlo_module_config.enable_hlo_profiling(true);
    }
    for (int i = 0; i < instance.argument_layouts.size(); ++i) {
      const Shape& argument_layout = *instance.argument_layouts[i];
      if (ShapeUtil::IsTuple(argument_layout)) {
        return Unimplemented("tuple arguments not supported yet");
      }
      TF_RETURN_IF_ERROR(
          computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
              argument_layout));
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            *instance.result_layout));

    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                        computation_tracker_.BuildHloModule(
                            versioned_handle, hlo_module_config,
                            /*include_unreachable_instructions=*/true));
    hlo_modules.push_back(std::move(hlo_module));
  }

  return compiler_->CompileAheadOfTime(std::move(hlo_modules),
                                       MakeHloDumper(), options);
}

}  // namespace xla
