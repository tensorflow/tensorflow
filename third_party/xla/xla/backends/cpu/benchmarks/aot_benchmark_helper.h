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

#ifndef XLA_BACKENDS_CPU_BENCHMARKS_AOT_BENCHMARK_HELPER_H_
#define XLA_BACKENDS_CPU_BENCHMARKS_AOT_BENCHMARK_HELPER_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"

namespace xla::cpu {

namespace internal {
inline constexpr absl::string_view kEntryPointNameDefault = "entry";
}  // end namespace internal

std::unique_ptr<AotCompilationOptions> GetAotCompilationOptions(
    absl::string_view entry_point_name = internal::kEntryPointNameDefault,
    CpuAotCompilationOptions::RelocationModel relocation_model =
        CpuAotCompilationOptions::RelocationModel::BigPic,
    absl::string_view features = "");

};  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_BENCHMARKS_AOT_BENCHMARK_HELPER_H_
