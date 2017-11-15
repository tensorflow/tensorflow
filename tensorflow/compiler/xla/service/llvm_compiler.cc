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

#include "tensorflow/compiler/xla/service/llvm_compiler.h"

namespace xla {
StatusOr<std::vector<std::unique_ptr<Executable>>> LLVMCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> modules,
    std::vector<std::vector<perftools::gputools::StreamExecutor*>>
        stream_execs) {
  std::vector<std::unique_ptr<Executable>> result;
  for (size_t i = 0; i < modules.size(); i++) {
    if (stream_execs[i].size() != 1) {
      return Unimplemented(
          "Model partitioning not implemented for the CPU/GPU compilers!");
    }

    TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                        Compile(std::move(modules[i]), stream_execs[i][0]));
    result.push_back(std::move(executable));
  }

  return {std::move(result)};
}
}  // namespace xla
