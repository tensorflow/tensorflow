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

#ifndef XLA_STREAM_EXECUTOR_CUDA_COMPILATION_OPTIONS_H_
#define XLA_STREAM_EXECUTOR_CUDA_COMPILATION_OPTIONS_H_

#include "absl/strings/str_format.h"

namespace stream_executor::cuda {

// Collects all compilation options that are used by the CompilationProvider
// interface.
struct CompilationOptions {
  // Disable all PTX compiler optimizations
  bool disable_optimizations = false;

  // If true, compilation will fail if register spilling is detected. A
  // absl::CancelledError will be returned.
  bool cancel_if_reg_spill = false;

  // If true, the PTX compiler will generate line information which is useful
  // for profiling
  bool generate_line_info = false;

  // If true, the PTX compiler will generate debug information.
  bool generate_debug_info = false;

  friend bool operator==(const CompilationOptions& lhs,
                         const CompilationOptions& rhs) {
    return lhs.disable_optimizations == rhs.disable_optimizations &&
           lhs.cancel_if_reg_spill == rhs.cancel_if_reg_spill &&
           lhs.generate_line_info == rhs.generate_line_info &&
           lhs.generate_debug_info == rhs.generate_debug_info;
  }

  friend bool operator!=(const CompilationOptions& lhs,
                         const CompilationOptions& rhs) {
    return !(lhs == rhs);
  }

  template <typename H>
  friend H AbslHashValue(H h, const CompilationOptions& options) {
    return H::combine(std::move(h), options.disable_optimizations,
                      options.cancel_if_reg_spill, options.generate_line_info,
                      options.generate_debug_info);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const CompilationOptions& options) {
    absl::Format(&sink,
                 "disable_optimizations: %v, cancel_if_reg_spill: %v, "
                 "generate_line_info: %v, generate_debug_info: %v",
                 options.disable_optimizations, options.cancel_if_reg_spill,
                 options.generate_line_info, options.generate_debug_info);
  }
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_COMPILATION_OPTIONS_H_
