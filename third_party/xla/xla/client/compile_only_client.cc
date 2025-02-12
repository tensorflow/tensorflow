/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/client/compile_only_client.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/service/compile_only_service.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/xla.pb.h"

namespace xla {

absl::StatusOr<std::unique_ptr<HloModuleConfig>>
CompileOnlyClient::CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const Shape* const> argument_shapes,
    const ExecutionOptions* execution_options) {
  return compiler_service_->CreateModuleConfig(program_shape, argument_shapes,
                                               execution_options);
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CompileOnlyClient::CompileAheadOfTime(
    const absl::Span<const AotXlaComputationInstance> computations,
    const AotCompilationOptions& options,
    std::unique_ptr<AotCompilationMetadata>* metadata) {
  std::vector<CompileOnlyService::AotXlaComputationInstance> service_instances;
  service_instances.reserve(computations.size());
  for (const AotXlaComputationInstance& instance : computations) {
    service_instances.emplace_back();
    CompileOnlyService::AotXlaComputationInstance& service_instance =
        service_instances.back();
    TF_RET_CHECK(instance.computation != nullptr);
    service_instance.computation = instance.computation->proto();
    service_instance.argument_layouts = instance.argument_layouts;
    service_instance.result_layout = *instance.result_layout;
  }
  return compiler_service_->CompileAheadOfTime(service_instances, options,
                                               metadata);
}

int64_t CompileOnlyClient::PointerSizeForTriple(absl::string_view triple) {
  llvm::Triple llvm_triple(
      llvm::Triple::normalize(llvm::StringRef(triple.data(), triple.size())));
  if (llvm_triple.isArch64Bit()) {
    return 8;
  } else if (llvm_triple.isArch32Bit()) {
    return 4;
  } else {
    CHECK(llvm_triple.isArch16Bit());
    return 2;
  }
}

}  // namespace xla
