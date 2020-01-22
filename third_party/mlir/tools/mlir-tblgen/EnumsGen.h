//===- EnumsGen.h - MLIR enum utility generator -----------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines common utilities for enum generator.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIR_TBLGEN_ENUMSGEN_H_
#define MLIR_TOOLS_MLIR_TBLGEN_ENUMSGEN_H_

#include "mlir/Support/LLVM.h"

namespace llvm {
class Record;
}

namespace mlir {
namespace tblgen {

using ExtraFnEmitter = llvm::function_ref<void(const llvm::Record &enumDef,
                                               llvm::raw_ostream &os)>;

// Emits declarations for the given EnumAttr `enumDef` into `os`.
//
// This will emit a C++ enum class and string to symbol and symbol to string
// conversion utility declarations. Additional functions can be emitted via
// the `emitExtraFns` function.
void emitEnumDecl(const llvm::Record &enumDef, ExtraFnEmitter emitExtraFns,
                  llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIR_TBLGEN_ENUMSGEN_H_
