//===- ConvertToBinary.cpp - MLIR SPIR-V module to binary conversion ------===//
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
// This file implements a translation from MLIR SPIR-V ModuleOp to SPIR-V
// binary module.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Module.h"
#include "mlir/SPIRV/SPIRVOps.h"
#include "mlir/SPIRV/Serialization.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

LogicalResult serializeModule(Module module, StringRef outputFilename) {
  if (!module)
    return failure();

  SmallVector<uint32_t, 0> binary;
  bool done = false;
  bool success = false;

  // TODO(antiagainst): we are checking there is only one SPIR-V ModuleOp in
  // this module and serialize it. This is due to the restriction of the current
  // translation infrastructure; we must take in a MLIR module here. So we are
  // wrapping the SPIR-V ModuleOp inside a MLIR module. This should be changed
  // to take in the SPIR-V ModuleOp directly after module and function are
  // migrated to be general ops.
  for (auto fn : module) {
    fn.walk<spirv::ModuleOp>([&](spirv::ModuleOp spirvModule) {
      if (done) {
        spirvModule.emitError("found more than one 'spv.module' op");
        return;
      }

      done = true;
      success = spirv::serialize(spirvModule, binary);
    });
  }

  if (!success)
    return failure();

  auto file = openOutputFile(outputFilename);
  if (!file)
    return failure();

  file->os().write(reinterpret_cast<char *>(binary.data()),
                   binary.size() * sizeof(uint32_t));
  file->keep();

  return mlir::success();
}

static TranslateFromMLIRRegistration
    registration("serialize-spirv",
                 [](Module module, StringRef outputFilename) {
                   return failed(serializeModule(module, outputFilename));
                 });
