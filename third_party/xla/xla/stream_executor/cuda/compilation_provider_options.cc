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

#include "xla/stream_executor/cuda/compilation_provider_options.h"

#include <string>

#include "absl/strings/str_format.h"
#include "xla/xla.pb.h"

namespace stream_executor::cuda {

std::string CompilationProviderOptions::ToString() const {
  return absl::StrFormat(
      "CompilationProviderOptions{nvjitlink_mode: %d, enable_libnvptxcompiler: "
      "%v, enable_llvm_module_compilation_parallelism: %v, "
      "enable_driver_compilation: %v, cuda_data_dir: %s}",
      nvjitlink_mode_, enable_libnvptxcompiler_,
      enable_llvm_module_compilation_parallelism_, enable_driver_compilation_,
      cuda_data_dir_);
}

CompilationProviderOptions CompilationProviderOptions::FromDebugOptions(
    const xla::DebugOptions& debug_options) {
  CompilationProviderOptions options;
  options.nvjitlink_mode_ = [&] {
    if (debug_options.xla_gpu_libnvjitlink_mode() ==
        xla::DebugOptions::LIB_NV_JIT_LINK_MODE_ENABLED) {
      return NvJitLinkMode::kEnabled;
    }
    if (debug_options.xla_gpu_libnvjitlink_mode() ==
        xla::DebugOptions::LIB_NV_JIT_LINK_MODE_DISABLED) {
      return NvJitLinkMode::kDisabled;
    }
    return NvJitLinkMode::kAuto;
  }();
  options.enable_libnvptxcompiler_ =
      debug_options.xla_gpu_enable_libnvptxcompiler();
  options.enable_llvm_module_compilation_parallelism_ =
      debug_options.xla_gpu_enable_llvm_module_compilation_parallelism();
  options.enable_driver_compilation_ =
      debug_options.xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found();
  options.cuda_data_dir_ = debug_options.xla_gpu_cuda_data_dir();
  return options;
}
}  //    namespace stream_executor::cuda
