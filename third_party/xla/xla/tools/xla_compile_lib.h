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

#ifndef XLA_TOOLS_XLA_COMPILE_LIB_H_
#define XLA_TOOLS_XLA_COMPILE_LIB_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/util.h"

namespace xla {

// Compiles the provided module for the given platform, either "cpu" or "gpu".
// When compiling for GPU, if the target config is provided, the compilation
// will be AOT. If it is not provided, an attached GPU will be used. When
// compiling for CPU, the compilation will always be AOT. If a result is
// provided, the post-optimization module will be stored in it.
//
// This is the expected entry point to the compilation functionality.
absl::StatusOr<std::string> CompileExecutable(
    std::unique_ptr<HloModule> hlo_module, absl::string_view platform,
    std::optional<Compiler::TargetConfig> target_config,
    CompilationResult& result);

// Merges the measured duration into compilation_result and writes
// compilation_result to result_output_file in the wire format.
absl::Status WriteResultFile(absl::string_view result_output_file,
                             TimerStats& stats,
                             CompilationResult& compilation_result);

// Loads the HLO, MHLO, or StableHLO module at the given file path.
absl::StatusOr<std::unique_ptr<HloModule>> LoadModule(
    absl::string_view module_path);

// Full entry point if you want to wrap a binary around this functionality.
// See flag definitions in ../service/xla_compile_main.cc for semantics.
absl::Status XlaCompileMain(
    absl::string_view module_path, absl::string_view output_path,
    absl::string_view platform, absl::string_view gpu_target_config_path,
    absl::string_view autotune_results_path, absl::string_view symbol_repo,
    absl::string_view symbol_id, bool use_attached_device,
    bool wait_for_uploads, absl::string_view result_output_file);

}  // namespace xla

#endif  // XLA_TOOLS_XLA_COMPILE_LIB_H_
