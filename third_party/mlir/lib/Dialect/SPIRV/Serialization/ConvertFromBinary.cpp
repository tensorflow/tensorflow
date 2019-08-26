//===- ConvertFromBinary.cpp - MLIR SPIR-V binary to module conversion ----===//
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
// This file implements a translation from SPIR-V binary module to MLIR SPIR-V
// ModuleOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

// Deserializes the SPIR-V binary module stored in the file named as
// `inputFilename` and returns a module containing the SPIR-V module.
OwningModuleRef deserializeModule(llvm::StringRef inputFilename,
                                  MLIRContext *context) {
  Builder builder(context);

  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    emitError(UnknownLoc::get(context), errorMessage);
    return {};
  }

  // Make sure the input stream can be treated as a stream of SPIR-V words
  auto start = file->getBufferStart();
  auto size = file->getBufferSize();
  if (size % sizeof(uint32_t) != 0) {
    emitError(UnknownLoc::get(context))
        << "SPIR-V binary module must contain integral number of 32-bit words";
    return {};
  }

  auto binary = llvm::makeArrayRef(reinterpret_cast<const uint32_t *>(start),
                                   size / sizeof(uint32_t));

  auto spirvModule = spirv::deserialize(binary, context);
  if (!spirvModule)
    return {};

  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(inputFilename, /*line=*/0, /*column=*/0, context)));
  module->getBody()->push_front(spirvModule->getOperation());

  return module;
}

static TranslateToMLIRRegistration
    registration("deserialize-spirv",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return deserializeModule(inputFilename, context);
                 });
