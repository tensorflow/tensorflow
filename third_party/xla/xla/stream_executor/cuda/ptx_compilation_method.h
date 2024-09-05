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

#ifndef XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILATION_METHOD_H_
#define XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILATION_METHOD_H_

#include <ostream>

#include "absl/strings/str_cat.h"
namespace stream_executor {

enum class PtxCompilationMethod {
  kNvJitLink,
  kNvPtxCompiler,
  kPtxas,
};

template <typename Sink>
static void AbslStringify(Sink& sink,
                          const PtxCompilationMethod& compilation_method) {
  switch (compilation_method) {
    case PtxCompilationMethod::kNvJitLink:
      sink.Append("NvJitLink");
      break;
    case PtxCompilationMethod::kNvPtxCompiler:
      sink.Append("NvPtxCompiler");
      break;
    case PtxCompilationMethod::kPtxas:
      sink.Append("Ptxas");
      break;
  }
}

inline std::ostream& operator<<(std::ostream& os,
                                const PtxCompilationMethod& method) {
  return os << absl::StrCat(method);
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILATION_METHOD_H_
