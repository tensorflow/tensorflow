/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_RUNTIME_UTILS_FLOAT_16BITS_H_
#define XLA_MLIR_RUNTIME_UTILS_FLOAT_16BITS_H_

#include <string_view>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"

// Provided by compiler-rt and MLIR.
// Converts an F32 value to a BF16.
extern "C" uint16_t __truncsfbf2(float);
// Converts an F64 value to a BF16.
extern "C" uint16_t __truncdfbf2(double);

namespace xla {
namespace runtime {

inline llvm::orc::SymbolMap Float16bitsSymbolMap(
    llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](std::string_view name, auto symbol_ptr) {
    symbol_map[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(symbol_ptr),
                                llvm::JITSymbolFlags()};
  };

  bind("__truncsfbf2", &__truncsfbf2);
  bind("__truncdfbf2", &__truncdfbf2);

  return symbol_map;
}

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_UTILS_FLOAT_16BITS_H_
