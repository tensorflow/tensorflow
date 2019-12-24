//===- MlirOptMain.h - MLIR Optimizer Driver main ---------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <vector>

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // end namespace llvm

namespace mlir {
struct LogicalResult;
class PassPipelineCLParser;

LogicalResult MlirOptMain(llvm::raw_ostream &os,
                          std::unique_ptr<llvm::MemoryBuffer> buffer,
                          const PassPipelineCLParser &passPipeline,
                          bool splitInputFile, bool verifyDiagnostics,
                          bool verifyPasses);

} // end namespace mlir
