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

#ifndef XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_OPTIONS_H_
#define XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_OPTIONS_H_

#include <ostream>
#include <string>

#include "xla/xla.pb.h"

namespace stream_executor::cuda {

class CompilationProviderOptions {
 public:
  enum class NvJitLinkMode {
    kDisabled,
    kEnabled,
    kAuto,
  };

  CompilationProviderOptions() = default;
  CompilationProviderOptions(NvJitLinkMode nvjitlink_mode,
                             bool enable_libnvptxcompiler,
                             bool enable_llvm_module_compilation_parallelism,
                             bool enable_driver_compilation,
                             std::string cuda_data_dir)
      : nvjitlink_mode_(nvjitlink_mode),
        enable_libnvptxcompiler_(enable_libnvptxcompiler),
        enable_llvm_module_compilation_parallelism_(
            enable_llvm_module_compilation_parallelism),
        enable_driver_compilation_(enable_driver_compilation),
        cuda_data_dir_(cuda_data_dir) {}

  static CompilationProviderOptions FromDebugOptions(
      const xla::DebugOptions& debug_options);

  NvJitLinkMode nvjitlink_mode() const { return nvjitlink_mode_; }
  bool enable_libnvptxcompiler() const { return enable_libnvptxcompiler_; }
  bool enable_llvm_module_compilation_parallelism() const {
    return enable_llvm_module_compilation_parallelism_;
  }
  bool enable_driver_compilation() const { return enable_driver_compilation_; }
  const std::string& cuda_data_dir() const { return cuda_data_dir_; }

  friend bool operator==(const CompilationProviderOptions& lhs,
                         const CompilationProviderOptions& rhs) {
    return lhs.nvjitlink_mode_ == rhs.nvjitlink_mode_ &&
           lhs.enable_libnvptxcompiler_ == rhs.enable_libnvptxcompiler_ &&
           lhs.enable_llvm_module_compilation_parallelism_ ==
               rhs.enable_llvm_module_compilation_parallelism_ &&
           lhs.enable_driver_compilation_ == rhs.enable_driver_compilation_ &&
           lhs.cuda_data_dir_ == rhs.cuda_data_dir_;
  }

  friend bool operator!=(const CompilationProviderOptions& lhs,
                         const CompilationProviderOptions& rhs) {
    return !(lhs == rhs);
  }

  std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const CompilationProviderOptions& options) {
    sink.Append(options.ToString());
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const CompilationProviderOptions& options) {
    return os << options.ToString();
  }

  template <typename H>
  friend H AbslHashValue(H h, const CompilationProviderOptions& options) {
    return H::combine(
        std::move(h), options.nvjitlink_mode_, options.enable_libnvptxcompiler_,
        options.enable_llvm_module_compilation_parallelism_,
        options.enable_driver_compilation_, options.cuda_data_dir_);
  }

 private:
  NvJitLinkMode nvjitlink_mode_ = NvJitLinkMode::kDisabled;
  bool enable_libnvptxcompiler_ = false;
  bool enable_llvm_module_compilation_parallelism_ = false;
  bool enable_driver_compilation_ = false;
  std::string cuda_data_dir_;
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_OPTIONS_H_
