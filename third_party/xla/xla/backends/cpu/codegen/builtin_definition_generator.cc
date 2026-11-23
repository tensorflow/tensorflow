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

#include "xla/backends/cpu/codegen/builtin_definition_generator.h"

#include <optional>
#include <utility>

#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/Shared/JITDylibLookupFlags.h"
#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "xla/backends/cpu/codegen/builtin_symbols.h"

namespace xla::cpu {

static std::optional<llvm::orc::ExecutorSymbolDef> ResolveBuiltinSymbol(
    const llvm::DataLayout& data_layout, llvm::StringRef name) {
  // On Mac OS X, 'name' may have a leading underscore prefix, even though the
  // registered name may not.
  if (name.size() > 1 && name.front() == data_layout.getGlobalPrefix()) {
    name = name.drop_front();
  }

  void* address =
      FindBuiltinSymbol(absl::string_view(name.data(), name.size()));
  if (address == nullptr) {
    return std::nullopt;
  }

  // We register runtime symbols as weak, because during concurrent compilation
  // different threads may race to register their symbols in the same dylib and
  // we get spurious "symbol already defined" errors.
  return llvm::orc::ExecutorSymbolDef{llvm::orc::ExecutorAddr::fromPtr(address),
                                      llvm::JITSymbolFlags::Weak};
}

//===----------------------------------------------------------------------===//
// BuiltinDefinitionGenerator
//===----------------------------------------------------------------------===//

BuiltinDefinitionGenerator::BuiltinDefinitionGenerator(
    llvm::DataLayout data_layout)
    : data_layout_(std::move(data_layout)) {}

llvm::Error BuiltinDefinitionGenerator::tryToGenerate(
    llvm::orc::LookupState&, llvm::orc::LookupKind kind,
    llvm::orc::JITDylib& jit_dylib, llvm::orc::JITDylibLookupFlags,
    const llvm::orc::SymbolLookupSet& names) {
  llvm::orc::SymbolMap symbols;
  symbols.reserve(names.size());

  for (const auto& [name, flags] : names) {
    if (auto symbol = ResolveBuiltinSymbol(data_layout_, *name)) {
      symbols[name] = *symbol;
    }
  }

  cantFail(jit_dylib.define(llvm::orc::absoluteSymbols(std::move(symbols))));
  return llvm::Error::success();
}

}  // namespace xla::cpu
