/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_H_
#define XLA_HLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_H_

#include <memory>
#include <utility>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace xla {

mlir::LogicalResult MlirHloToHloTranslateFunction(mlir::ModuleOp module,
                                                  llvm::raw_ostream& output,
                                                  bool emit_return_tuple,
                                                  bool emit_use_tuple_arg);

mlir::LogicalResult MlirHloToHloTextTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts);

// Translate the MHLO program in in-memory file 'buffer' to a HLO program
// written in a file represented with handle 'output_stream';
mlir::LogicalResult MlirHloToHloTextMain(
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    llvm::raw_ostream& output_stream, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts);

}  // namespace xla

#endif  // XLA_HLO_TRANSLATE_MHLO_TO_HLO_TRANSLATE_H_
