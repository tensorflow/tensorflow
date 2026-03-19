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

#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/test_target_triple_helper.h"

namespace xla::cpu {
std::unique_ptr<AotCompilationOptions> GetAotCompilationOptions(
    absl::string_view entry_point_name,
    CpuAotCompilationOptions::RelocationModel relocation_model,
    absl::string_view features) {
  std::string entry_point_name_str(entry_point_name);
  std::string features_str(features);
  return std::make_unique<CpuAotCompilationOptions>(
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/std::move(features_str),
      /*entry_point_name=*/std::move(entry_point_name_str),
      /*relocation_model=*/relocation_model);
}
}  // namespace xla::cpu
