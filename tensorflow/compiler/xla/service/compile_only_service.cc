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
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<CompileOnlyService>>
CompileOnlyService::NewService(se::Platform* platform) {
  ServiceOptions default_options;
  default_options.set_platform(platform);
  return NewService(default_options);
}

/* static */ StatusOr<std::unique_ptr<CompileOnlyService>>
CompileOnlyService::NewService(const ServiceOptions& options) {
  se::Platform* platform = options.platform();
  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));

  std::unique_ptr<CompileOnlyService> service(
      new CompileOnlyService(options, compiler));
  return std::move(service);
}

CompileOnlyService::CompileOnlyService(const ServiceOptions& options,
                                       Compiler* compiler)
    : Service(options, /*execute_backend=*/nullptr), compiler_(compiler) {}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CompileOnlyService::CompileAheadOfTime(
    const tensorflow::gtl::ArraySlice<AotXlaComputationInstance> computations,
    const AotCompilationOptions& options,
    std::unique_ptr<AotCompilationMetadata>* metadata) {
  std::vector<std::unique_ptr<HloModule>> hlo_modules;
  for (const AotXlaComputationInstance& instance : computations) {
    TF_RET_CHECK(instance.computation.has_program_shape());

    const DebugOptions& debug_options = options.debug_options();

    // Dump computation proto if flag is set.
    const string& directory_path = debug_options.xla_dump_computations_to();
    if (!directory_path.empty()) {
      HloSnapshot hlo_snapshot;
      *hlo_snapshot.mutable_hlo()->mutable_hlo_module() = instance.computation;
      string filename = tensorflow::strings::StrCat(
          "computation_", instance.computation.id(), "__",
          instance.computation.entry_computation_name());
      const string& per_host_path = tensorflow::io::JoinPath(
          directory_path, tensorflow::port::Hostname());

      TF_RETURN_IF_ERROR(
          Executable::DumpToDirectory(per_host_path, filename, hlo_snapshot));
    }

    const auto& program_shape = instance.computation.program_shape();
    ExecutionOptions execution_options;
    *execution_options.mutable_debug_options() = debug_options;
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleConfig> module_config,
        CreateModuleConfig(program_shape, instance.argument_layouts,
                           &execution_options));

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> hlo_module,
        HloModule::CreateFromProto(instance.computation, *module_config));
    TF_RETURN_IF_ERROR(MaybeDumpHloModule(*hlo_module));
    hlo_modules.push_back(std::move(hlo_module));
  }

  return compiler_->CompileAheadOfTime(std::move(hlo_modules), options,
                                       metadata);
}

}  // namespace xla
