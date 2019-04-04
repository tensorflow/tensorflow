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
class DialectConversion;
class DialectOpConversion;
class MLIRContext;
class Module;
class Type;
namespace LLVM {
class LLVMType;
} // end namespace LLVM
} // end namespace mlir

namespace linalg {
/// Convert the given Linalg dialect type `t` into an LLVM IR dialect type.
/// Keep all other types unmodified.
mlir::Type convertLinalgType(mlir::Type t);

/// Allocate the conversion patterns for RangeOp, ViewOp and SliceOp from the
/// Linalg dialect to the LLVM IR dialect.  The converters are allocated in the
/// `allocator` using the provided `context`.  The latter must have the LLVM IR
/// dialect registered.
/// This function can be used to apply multiple conversion patterns in the same
/// pass.  It does not have to be called explicitly before the conversion.
llvm::DenseSet<mlir::DialectOpConversion *>
allocateDescriptorConverters(llvm::BumpPtrAllocator *allocator,
                             mlir::MLIRContext *context);

/// Create a DialectConversion from the Linalg dialect to the LLVM IR dialect.
/// The conversion is set up to convert types and function signatures using
/// `convertLinalgType` and obtains operation converters by calling `initer`.
std::unique_ptr<mlir::DialectConversion> makeLinalgToLLVMLowering(
    std::function<llvm::DenseSet<mlir::DialectOpConversion *>(
        llvm::BumpPtrAllocator *, mlir::MLIRContext *context)>
        initer);

/// Convert the Linalg dialect types and RangeOp, ViewOp and SliceOp operations
/// to the LLVM IR dialect types and operations in the given `module`.  This is
/// the main entry point to the conversion.
void convertToLLVM(mlir::Module &module);
} // end namespace linalg

#endif // LINALG1_CONVERTTOLLVMDIALECT_H_
