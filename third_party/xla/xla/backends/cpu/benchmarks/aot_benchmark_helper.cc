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

#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"

namespace xla::cpu {
std::shared_ptr<AotCompilationOptions> GetAotCompilationOptions(
    std::string entry_point_name,
    CpuAotCompilationOptions::RelocationModel relocation_model,
    std::string triple, std::string cpu_name, std::string features) {
  return std::make_shared<CpuAotCompilationOptions>(
      /*triple=*/std::move(triple), /*cpu_name=*/std::move(cpu_name),
      /*features=*/std::move(features),
      /*entry_point_name=*/std::move(entry_point_name),
      /*relocation_model=*/relocation_model);
}
}  // namespace xla::cpu
