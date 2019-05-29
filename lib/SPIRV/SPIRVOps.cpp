//===- SPIRVOps.cpp - MLIR SPIR-V operations ------------------------------===//
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
// This file defines the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/SPIRV/SPIRVOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

static ParseResult parseModule(OpAsmParser *parser, OperationState *state) {
  Region *body = state->addRegion();

  if (parser->parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
      parser->parseKeyword("attributes") ||
      parser->parseOptionalAttributeDict(state->attributes))
    return failure();

  return success();
}

static ParseResult printModule(Operation *op, OpAsmPrinter *printer) {
  *printer << op->getName();
  printer->printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                       /*printBlockTerminators=*/false);
  *printer << " attributes";
  printer->printOptionalAttrDict(op->getAttrs(),
                                 /*elidedAttrs=*/{});
  return success();
}

namespace mlir {
namespace spirv {

#define GET_OP_CLASSES
#include "mlir/SPIRV/SPIRVOps.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/SPIRV/SPIRVStructureOps.cpp.inc"

} // namespace spirv
} // namespace mlir
