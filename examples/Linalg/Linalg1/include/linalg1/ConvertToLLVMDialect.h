//===- ConvertToLLVMDialect.h - conversion from Linalg to LLVM --*- C++ -*-===//
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

#ifndef LINALG1_CONVERTTOLLVMDIALECT_H_
#define LINALG1_CONVERTTOLLVMDIALECT_H_

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Allocator.h"

#include <memory>

namespace mlir {
class ConversionPattern;
class DialectConversion;
struct LogicalResult;
class MLIRContext;
class ModuleOp;
class RewritePattern;
class Type;
using OwningRewritePatternList = std::vector<std::unique_ptr<RewritePattern>>;
namespace LLVM {
class LLVMType;
} // end namespace LLVM
} // end namespace mlir

namespace linalg {
/// Convert the given Linalg dialect type `t` into an LLVM IR dialect type.
/// Keep all other types unmodified.
mlir::Type convertLinalgType(mlir::Type t);

/// Get the conversion patterns for RangeOp, ViewOp and SliceOp from the Linalg
/// dialect to the LLVM IR dialect. The LLVM IR dialect must be registered. This
/// function can be used to apply multiple conversion patterns in the same pass.
/// It does not have to be called explicitly before the conversion.
void populateLinalg1ToLLVMConversionPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context);

/// Convert the Linalg dialect types and RangeOp, ViewOp and SliceOp operations
/// to the LLVM IR dialect types and operations in the given `module`.  This is
/// the main entry point to the conversion.
mlir::LogicalResult convertToLLVM(mlir::ModuleOp module);
} // end namespace linalg

#endif // LINALG1_CONVERTTOLLVMDIALECT_H_
