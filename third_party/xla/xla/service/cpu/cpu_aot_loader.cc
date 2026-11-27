/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/cpu/cpu_aot_loader.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/service/compiled_module.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/function_library_loader.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/util.h"

#ifdef ABSL_HAVE_MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#endif

namespace xla::cpu {
namespace {

// The msan instrumentation of this process, which AOT results must match.
#ifdef ABSL_HAVE_MEMORY_SANITIZER
constexpr bool kMsanEnabled = true;
int MsanTrackOrigins() { return __msan_get_track_origins(); }
#else
constexpr bool kMsanEnabled = false;
int MsanTrackOrigins() { return 0; }
#endif

}  // namespace

llvm::TargetOptions CompilerTargetOptions(
    const HloModuleConfig& module_config) {
  llvm::TargetOptions target_options;
  return target_options;
}

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    const std::string& serialized_aot_result,
    FunctionLibraryLoader function_library_loader) {
  xla::cpu::CompilationResultProto proto;
  if (!proto.ParseFromString(serialized_aot_result)) {
    return Internal("Failed to parse serialized CpuAotCompilationResult.");
  }
  return LoadExecutable(proto, function_library_loader);
}

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    const xla::cpu::CompilationResultProto& aot_result_proto,
    FunctionLibraryLoader function_library_loader) {
  ABSL_ASSIGN_OR_RETURN(
      auto aot_result,
      LoadAotCompilationResult(aot_result_proto, function_library_loader));
  return LoadExecutable(std::move(*aot_result));
}

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    CompiledModule&& compilation_result) {
  return std::move(compilation_result).LoadExecutable();
}

absl::StatusOr<std::unique_ptr<CompiledModule>>
CpuAotLoader::LoadAotCompilationResult(
    const std::string& serialized_aot_result,
    FunctionLibraryLoader function_library_loader) {
  xla::cpu::CompilationResultProto proto;
  if (!proto.ParseFromString(serialized_aot_result)) {
    return Internal("Failed to parse serialized CpuAotCompilationResult.");
  }
  return LoadAotCompilationResult(proto, function_library_loader);
}

absl::StatusOr<std::unique_ptr<CompiledModule>>
CpuAotLoader::LoadAotCompilationResult(
    const xla::cpu::CompilationResultProto& aot_result_proto,
    FunctionLibraryLoader function_library_loader) {
  if (aot_result_proto.msan_enabled() != kMsanEnabled ||
      aot_result_proto.msan_track_origins() != MsanTrackOrigins()) {
    return FailedPrecondition(
        "The AOT result has msan_enabled=%d and msan_track_origins=%d, but "
        "this process has msan_enabled=%d and msan_track_origins=%d",
        aot_result_proto.msan_enabled(), aot_result_proto.msan_track_origins(),
        kMsanEnabled, MsanTrackOrigins());
  }

  VLOG(3) << "AOT result target machine options: "
          << aot_result_proto.target_machine_options().DebugString();

  ABSL_ASSIGN_OR_RETURN(TargetMachineOptions target_machine_options,
                   TargetMachineOptions::FromProto(
                       aot_result_proto.target_machine_options()));
  llvm::Triple host_triple(llvm::sys::getDefaultTargetTriple());
  llvm::Triple expected_triple(target_machine_options.triple());
  if (host_triple.getArchName() != expected_triple.getArchName()) {
    return Internal("Target arch mismatch expected %s got %s.",
                    expected_triple.getArchName(), host_triple.getArchName());
  }

  const llvm::StringMap<bool> host_machine_features =
      llvm::sys::getHostCPUFeatures();
  const std::vector<std::string> compile_machine_features =
      target_machine_options.GetTargetMachineFeaturesVector();
  // Convert the supported features to a vector of strings.
  std::vector<std::string> host_machine_features_vector;
  for (const auto& [feature, supported] : host_machine_features) {
    if (supported) {
      host_machine_features_vector.push_back(feature.str());
    }
  }

  VLOG(3) << "Host machine options:"
          << "\nHost CPU: " << llvm::sys::getHostCPUName().str()
          << "\nHost triple: " << host_triple.str() << "\nHost features: "
          << absl::StrJoin(host_machine_features_vector, ",");

  for (const absl::string_view feature : compile_machine_features) {
    if (!absl::StartsWith(feature, "+")) {
      continue;
    }
    absl::string_view feature_name = feature.substr(1);
    // LLVM tuning options (`prefer-*` and `fast-*`) guide internal
    // microarchitectural performance decisions (cost models, combiners) rather
    // than architectural hardware ISA features. Consequently,
    // `llvm::sys::getHostCPUFeatures()` does not report them. We filter them
    // out to prevent false-positive compatibility failures.
    if (absl::StartsWith(feature_name, "prefer-") ||
        absl::StartsWith(feature_name, "fast-")) {
      continue;
    }
    if (!host_machine_features.lookup(
            llvm::StringRef(feature_name.data(), feature_name.size()))) {
      // TODO: b/477590953 - Turn this warning into an absl::Status Internal
      // error once a mechanism for passing CPU topology to host offloaded
      // programs is implemented.
      LOG(ERROR)
          << "Loading XLA:CPU AOT result. Target machine feature " << feature
          << " is not  supported on the host machine. Machine type used for "
             "XLA:CPU compilation doesn't match the machine type for "
             "execution. Compile machine features: ["
          << absl::StrJoin(compile_machine_features, ",")
          << "] vs host machine features: ["
          << absl::StrJoin(host_machine_features_vector, ",") << "]"
          << ". This could lead to execution errors such as SIGILL.";
    }
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> function_library,
                   function_library_loader(aot_result_proto));
  if (function_library == nullptr) {
    return Internal("FunctionLibraryLoader returned a null FunctionLibrary.");
  }

  return CpuAotCompilationResult::FromProto(aot_result_proto,
                                            std::move(function_library));
}

}  // namespace xla::cpu
