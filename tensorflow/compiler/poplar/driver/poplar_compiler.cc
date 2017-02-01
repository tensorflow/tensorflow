/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/poplar/driver/poplar_compiler.h"
#include "tensorflow/compiler/poplar/driver/poplar_executable.h"

#include "tensorflow/stream_executor/poplar/poplar_platform_id.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/lib/initialize.h"

#include <iostream>
namespace se = ::perftools::gputools;

namespace xla {
namespace poplar {

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::Compile(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloModuleConfig> module_config, HloDumper dump_hlo,
    se::StreamExecutor* stream_exec) {
  TF_RET_CHECK(stream_exec != nullptr);

  std::cout << "PoplarCompiler::Compile\n";

  return tensorflow::errors::Unimplemented(
          "Compilation of multiple HLO modules is not yet supported on CPU.");
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    HloDumper dump_hlos, std::vector<se::StreamExecutor*> stream_execs) {
  return tensorflow::errors::Unimplemented(
      "Compilation of multiple HLO modules is not yet supported on CPU.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PoplarCompiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    HloDumper dump_hlo, const AotCompilationOptions& aot_options) {
  TF_RET_CHECK(hlo_modules.size() == module_configs.size());

  return tensorflow::errors::InvalidArgument(
      "AOT compilation not supported on Poplar");
}

se::Platform::Id PoplarCompiler::PlatformId() const {
  return se::poplar::kPoplarPlatformId;
}

}  // namespace poplar
}  // namespace xla

REGISTER_MODULE_INITIALIZER(poplar_compiler, {
  xla::Compiler::RegisterCompilerFactory(se::poplar::kPoplarPlatformId, []() {
    return xla::MakeUnique<xla::poplar::PoplarCompiler>();
  });
});
