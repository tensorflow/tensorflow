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

#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/Shared/JITDylibLookupFlags.h"
#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "xla/backends/cpu/codegen/builtin_symbols.h"

namespace xla::cpu {

#ifdef ABSL_HAVE_MEMORY_SANITIZER
static constexpr bool kHasMemorySanitizer = true;
#else
static constexpr bool kHasMemorySanitizer = false;
#endif

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
    : data_layout_(std::move(data_layout)) {
  if constexpr (kHasMemorySanitizer) {
    // Resolve msan runtime functions (e.g. __msan_warning*) from the current
    // process via dlsym. This is more future-proof than explicitly registering
    // __msan_* functions; these functions do change between LLVM versions.
    auto is_msan_symbol = [prefix = data_layout_.getGlobalPrefix()](
                              const llvm::orc::SymbolStringPtr& name) {
      llvm::StringRef str = *name;
      if (!str.empty() && str.front() == prefix) {
        str = str.drop_front();
      }
      return str.starts_with("__msan_");
    };
    // Register runtime symbols as weak -- see ResolveBuiltinSymbol above.
    auto add_weak_symbols = [](llvm::orc::JITDylib& jd,
                               llvm::orc::SymbolMap symbols) -> llvm::Error {
      for (auto& [name, def] : symbols) {
        def.setFlags(def.getFlags() | llvm::JITSymbolFlags::Weak);
      }
      return jd.define(llvm::orc::absoluteSymbols(std::move(symbols)));
    };
    auto generator =
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            data_layout_.getGlobalPrefix(), is_msan_symbol, add_weak_symbols);
    if (generator) {
      process_generator_ = std::move(*generator);
    } else {
      LOG(WARNING)
          << "Failed to initialize dynamic library generator for msan: "
          << llvm::toString(generator.takeError());
    }
  }
}

llvm::Error BuiltinDefinitionGenerator::tryToGenerate(
    llvm::orc::LookupState& ls, llvm::orc::LookupKind kind,
    llvm::orc::JITDylib& jit_dylib, llvm::orc::JITDylibLookupFlags lookup_flags,
    const llvm::orc::SymbolLookupSet& names) {
  llvm::orc::SymbolMap symbols;
  symbols.reserve(names.size());

  for (const auto& [name, flags] : names) {
    if (auto symbol = ResolveBuiltinSymbol(data_layout_, *name)) {
      symbols[name] = *symbol;
    }
  }

  cantFail(jit_dylib.define(llvm::orc::absoluteSymbols(std::move(symbols))));

  if (process_generator_ != nullptr) {
    return process_generator_->tryToGenerate(ls, kind, jit_dylib, lookup_flags,
                                             names);
  }
  return llvm::Error::success();
}

}  // namespace xla::cpu
