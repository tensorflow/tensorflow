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
#include "xla/service/symbol_repository.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/util.h"

namespace xla {

// Compiles the provided module for the given platform.
// When compiling for GPU, if the target config is provided, the compilation
// will be AOT. If it is not provided, an attached GPU will be used. When
// compiling for CPU, the compilation will always be AOT. If a result is
// provided, the post-optimization module will be stored in it.
//
// This is the expected entry point to the compilation functionality.
absl::StatusOr<std::string> CompileExecutable(
    std::unique_ptr<HloModule> hlo_module, BackendType backend,
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

struct XlaCompileOptions {
  // Fully backend-independent options.
  std::string module_path;
  std::string output_path;
  std::string platform;
  std::string result_output_file;

  // Options for SymbolRepository lookup.
  struct SymbolRepoOptions {
    std::string symbol_repo;
    std::string symbol_id;
    std::string optimized_symbol_id;
    bool wait_for_uploads;
  };

  // GPU-specific options.
  struct GpuOptions {
    std::string gpu_target_config_path;
    bool use_attached_device;
    std::string autotune_results_path;
  };

  SymbolRepoOptions repo_options;
  GpuOptions gpu_options;
};

// Full entry point if you want to wrap a binary around this functionality. See
// flag definitions in ../service/xla_compile_main.cc for semantics, which
// correspond to fields in XlaCompileOptions.
absl::Status XlaCompileMain(const XlaCompileOptions& compile_options);

}  // namespace xla

#endif  // XLA_TOOLS_XLA_COMPILE_LIB_H_
