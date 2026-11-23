/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/cpu/function_library_loader.h"

#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/any_span.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/util.h"

namespace xla::cpu {

absl::StatusOr<std::vector<FunctionLibrary::Symbol>>
GetCompiledSymbolsFromProto(
    absl::AnySpan<const SymbolProto> compiled_symbols_proto) {
  std::vector<FunctionLibrary::Symbol> compiled_symbols;
  compiled_symbols.reserve(compiled_symbols_proto.size());
  for (const SymbolProto& symbol_proto : compiled_symbols_proto) {
    switch (symbol_proto.function_type_id()) {
      case SymbolProto::KERNEL:
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Kernel>(symbol_proto.name()));
        break;
      case SymbolProto::COMPARATOR:
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Comparator>(
                symbol_proto.name()));
        break;
      default:
        return Internal(
            "Unknown function type id %s",
            SymbolProto_FunctionTypeId_Name(symbol_proto.function_type_id()));
    }
  }
  VLOG(3) << "Collected " << compiled_symbols.size() << " compiled symbols";
  for (const FunctionLibrary::Symbol& symbol : compiled_symbols) {
    VLOG(3) << " Symbol: " << symbol.name;
  }

  return compiled_symbols;
}

}  // namespace xla::cpu
