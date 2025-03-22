/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/assemble_compilation_provider.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/composite_compilation_provider.h"
#include "xla/stream_executor/cuda/defer_relocatable_compilation_compilation_provider.h"
#include "xla/stream_executor/cuda/driver_compilation_provider.h"
#include "xla/stream_executor/cuda/nvjitlink_compilation_provider.h"
#include "xla/stream_executor/cuda/nvjitlink_known_issues.h"
#include "xla/stream_executor/cuda/nvjitlink_support.h"
#include "xla/stream_executor/cuda/nvptxcompiler_compilation_provider.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/cuda/subprocess_compilation_provider.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::cuda {
namespace {

// Returns true if NvJitLink is supported and should be used.
absl::Status HasNvJitLinkSupport(const CompilationProviderOptions& options) {
  if (!IsLibNvJitLinkSupported()) {
    return absl::UnavailableError(
        "LibNvJitLink is not supported (disabled during compilation).");
  }

  if (options.nvjitlink_mode() ==
      CompilationProviderOptions::NvJitLinkMode::kDisabled) {
    return absl::UnavailableError(
        "LibNvJitLink is disabled (explicitly disabled via flag).");
  }

  if (options.nvjitlink_mode() ==
      CompilationProviderOptions::NvJitLinkMode::kEnabled) {
    VLOG(4) << "Considering NvJitLink since it was explicitly enabled.";
    return absl::OkStatus();
  }

  if (LoadedNvJitLinkHasKnownIssues()) {
    return absl::UnavailableError(
        "LibNvJitLink is disabled since the loaded library version has known "
        "issues.");
  }

  VLOG(4)
      << "Considering NvJitLink since the loaded library version has no known "
         "issues.";
  return absl::OkStatus();
}

// Returns true if LibNvPtxCompiler is supported and should be used.
absl::Status HasNvptxCompilerSupport(
    const CompilationProviderOptions& options) {
  if (!IsLibNvPtxCompilerSupported()) {
    return absl::UnavailableError(
        "LibNvPtxCompiler is not supported (disabled during compilation).");
  }

  if (!options.enable_libnvptxcompiler()) {
    return absl::UnavailableError(
        "LibNvPtxCompiler is disabled (explicitly disabled via flag).");
  }

  VLOG(4) << "Considering NvPtxCompiler since it was supported and enabled.";
  return absl::OkStatus();
}

// Returns an error if the user-set flags are not compatible with each other and
// the build of XLA.
absl::Status CheckIncompatibleFlagSettings(
    const CompilationProviderOptions& options) {
  if (options.nvjitlink_mode() ==
          CompilationProviderOptions::NvJitLinkMode::kEnabled &&
      !IsLibNvJitLinkSupported()) {
    return absl::UnavailableError("LibNvJitLink is not supported.");
  }

  if (options.enable_libnvptxcompiler() && !IsLibNvPtxCompilerSupported()) {
    return absl::UnavailableError("LibNvPtxCompiler is not supported.");
  }

  return absl::OkStatus();
}

// Calls `GetToolVersion` on the given path if it's OK. Otherwise returns the
// error status.
absl::StatusOr<SemanticVersion> GetToolVersionIfToolAvailable(
    const absl::StatusOr<std::string>& path) {
  if (!path.ok()) {
    return path.status();
  }

  return GetToolVersion(path.value());
}

// Returns the given non-OK status or the value as a string.
template <typename T>
std::string ToDebugString(const absl::StatusOr<T>& status_or) {
  if (status_or.ok()) {
    return absl::StrCat(status_or.value());
  }
  return std::string{status_or.status().message()};
}

}  // namespace

absl::StatusOr<std::unique_ptr<CompilationProvider>>
AssembleCompilationProvider(const CompilationProviderOptions& options) {
  // TODO(b/381059098): Simplify this logic

  TF_RETURN_IF_ERROR(CheckIncompatibleFlagSettings(options));

  std::string decision_log;
  const auto append_to_decision_log = [&](absl::string_view decision) {
    VLOG(4) << decision;
    absl::StrAppend(&decision_log, " - ", decision, "\n");
  };

  const absl::Status has_nvjitlink = HasNvJitLinkSupport(options);
  append_to_decision_log(
      absl::StrCat("Has NvJitLink support: ", has_nvjitlink.message()));

  const absl::Status has_nvptxcompiler = HasNvptxCompilerSupport(options);
  append_to_decision_log(
      absl::StrCat("Has NvPtxCompiler support: ", has_nvptxcompiler.message()));

  const bool parallel_compilation_support_is_desired =
      options.enable_llvm_module_compilation_parallelism();
  append_to_decision_log(
      absl::StrCat("Parallel compilation support is desired: ",
                   parallel_compilation_support_is_desired));

  if (has_nvjitlink.ok() && has_nvptxcompiler.ok()) {
    // If both libraries are supported, we will use them together. This setup
    // supports parallel compilation and we have the most control over the
    // versions being used.
    VLOG(3) << "Using libnvptxcompiler for compilation and libnvjitlink for "
               "linking.";
    std::vector<std::unique_ptr<CompilationProvider>> providers;
    providers.reserve(2);
    providers.push_back(std::make_unique<NvptxcompilerCompilationProvider>());
    providers.push_back(std::make_unique<NvJitLinkCompilationProvider>());
    return CompositeCompilationProvider::Create(std::move(providers));
  }

  if (has_nvjitlink.ok() && !has_nvptxcompiler.ok()) {
    // If we only have libnvjitlink, we use it for both compilation and
    // linking. To support parallel compilation we defer compilation into
    // relocatable modules to the linking step by using the
    // DeferRelocatableCompilationCompilationProvider.
    VLOG(3) << "Using libnvjitlink for compilation and linking.";
    return DeferRelocatableCompilationCompilationProvider::Create(
        std::make_unique<NvJitLinkCompilationProvider>());
  }

  if (has_nvptxcompiler.ok() && !parallel_compilation_support_is_desired) {
    // If we only have libnvptxcompiler, but don't need parallel compilation, we
    // can just use the library on its own - no linking required.
    VLOG(3) << "Using only libnvptxcompiler for compilation - no parallel "
               "compilation support needed.";
    return std::make_unique<NvptxcompilerCompilationProvider>();
  }

  absl::StatusOr<std::string> ptxas_path =
      FindPtxAsExecutable(options.cuda_data_dir());
  absl::StatusOr<SemanticVersion> ptxas_version =
      GetToolVersionIfToolAvailable(ptxas_path);

  absl::StatusOr<std::string> nvlink_path =
      FindNvlinkExecutable(options.cuda_data_dir());
  absl::StatusOr<SemanticVersion> nvlink_version =
      GetToolVersionIfToolAvailable(nvlink_path);

  append_to_decision_log(
      absl::StrCat("ptxas_path: ", ToDebugString(ptxas_path)));
  append_to_decision_log(
      absl::StrCat("ptxas_version: ", ToDebugString(ptxas_version)));
  append_to_decision_log(
      absl::StrCat("nvlink_path: ", ToDebugString(nvlink_path)));
  append_to_decision_log(
      absl::StrCat("nvlink_version: ", ToDebugString(nvlink_version)));

  const bool has_subprocess_compilation_support =
      ptxas_path.ok() && nvlink_path.ok();

  if (has_subprocess_compilation_support) {
    VLOG(3) << "Using ptxas(path=" << ptxas_path.value()
            << ", version=" << ptxas_version.value() << ") and "
            << "nvlink(path=" << nvlink_path.value()
            << ", version=" << nvlink_version.value()
            << ") for compilation and linking.";
    return std::make_unique<SubprocessCompilationProvider>(ptxas_path.value(),
                                                           nvlink_path.value());
  }

  const bool has_driver_compilation_support =
      options.enable_driver_compilation();
  append_to_decision_log(absl::StrCat("Driver compilation is enabled: ",
                                      has_driver_compilation_support));

  if (parallel_compilation_support_is_desired && has_nvptxcompiler.ok() &&
      has_driver_compilation_support) {
    // It's possible to use libnvptxcompiler for compilation and the driver for
    // linking. This setup supports parallel compilation but is less desired
    // because we don't control the driver version. A too old driver might lead
    // to linking errors.
    VLOG(3) << "Using libnvptxcompiler for compilation and the driver for "
               "linking.";
    std::vector<std::unique_ptr<CompilationProvider>> providers;
    providers.reserve(2);
    providers.push_back(std::make_unique<NvptxcompilerCompilationProvider>());
    providers.push_back(std::make_unique<DriverCompilationProvider>());
    return CompositeCompilationProvider::Create(std::move(providers));
  }

  if (ptxas_path.ok() && has_driver_compilation_support) {
    // It's possible to use ptxas for compilation and the driver for linking.
    // This setup supports parallel compilation but is less desired because we
    // don't control the driver version. A too old driver might lead to linking
    // errors.
    VLOG(3) << "Using libnvptxcompiler for compilation and the driver for "
               "linking.";
    std::vector<std::unique_ptr<CompilationProvider>> providers;
    auto ptxas_provider = std::make_unique<SubprocessCompilationProvider>(
        ptxas_path.value(), std::string{});
    providers.reserve(2);
    providers.push_back(std::move(ptxas_provider));
    providers.push_back(std::make_unique<DriverCompilationProvider>());
    return CompositeCompilationProvider::Create(std::move(providers));
  }

  // Passed this point we won't be able to support parallel compilation, so we
  // error out if it was requested.
  if (parallel_compilation_support_is_desired) {
    return absl::UnavailableError(
        absl::StrCat("Parallel compilation was requested, but no available "
                     "compilation provider supports it. Details: \n",
                     decision_log));
  }

  if (ptxas_path.ok()) {
    VLOG(3) << "Using ptxas(path=" << ptxas_path.value()
            << ", version=" << ptxas_version.value()
            << ") for compilation. nvlink is not available.";
    return std::make_unique<SubprocessCompilationProvider>(ptxas_path.value(),
                                                           std::string{});
  }

  if (has_driver_compilation_support) {
    VLOG(3) << "Using the driver for compilation.";
    return std::make_unique<DriverCompilationProvider>();
  }

  return absl::UnavailableError(absl::StrCat(
      "No PTX compilation provider is available. Neither ptxas/nvlink nor "
      "nvjtlink is available. As a fallback you can enable JIT compilation "
      "in the CUDA driver via the flag "
      "`--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found`. Details: \n",
      decision_log));
}

}  // namespace stream_executor::cuda
