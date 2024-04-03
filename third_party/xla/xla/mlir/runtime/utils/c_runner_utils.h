/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_MLIR_RUNTIME_UTILS_C_RUNNER_UTILS_H_
#define XLA_MLIR_RUNTIME_UTILS_C_RUNNER_UTILS_H_

#include <string_view>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"  // from @llvm-project

namespace xla {
namespace runtime {

inline llvm::orc::SymbolMap CRunnerUtilsSymbolMap(
    llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](std::string_view name, auto symbol_ptr) {
    symbol_map[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(symbol_ptr),
                                llvm::JITSymbolFlags()};
  };

#ifndef _WIN32
  // TODO(b/246980307): fails to link on windows because it's marked dllimport.
  bind("memrefCopy", &memrefCopy);
#endif

  return symbol_map;
}

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_UTILS_C_RUNNER_UTILS_H_
