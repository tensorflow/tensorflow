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

#include "xla/service/compiler_base.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/debug_options_flags.h"
#include "xla/service/compiled_module_base.h"
#include "xla/util.h"

namespace xla {

CompilerBase::~CompilerBase() = default;

// Define a default version where metadata is not used.
absl::StatusOr<std::vector<std::unique_ptr<CompiledModuleBase>>>
CompilerBase::CompileAheadOfTime(
    std::unique_ptr<HloModule> hlo_module,
    const AotCompilationOptionsBase& options,
    std::unique_ptr<AotCompilationMetadata>* metadata) {
  if (metadata != nullptr) {
    return Unimplemented(
        "Populating AotCompilationMetadata is not implemented on this "
        "compiler.");
  }
  return CompileAheadOfTime(std::move(hlo_module), options);
}

AotCompilationOptionsBase::AotCompilationOptionsBase()
    : debug_options_(GetDebugOptionsFromFlags()) {}

}  // namespace xla
