//===- SPIRVGLSLOps.cpp - MLIR SPIR-V GLSL extended operations ------------===//
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
// This file defines the operations in the SPIR-V extended instructions set for
// GLSL
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVGLSLOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// spv.glsl.UnaryOp
//===----------------------------------------------------------------------===//

static ParseResult parseGLSLUnaryOp(OpAsmParser *parser,
                                    OperationState *state) {
  OpAsmParser::OperandType operandInfo;
  Type type;
  if (parser->parseOperand(operandInfo) || parser->parseColonType(type) ||
      parser->resolveOperands(operandInfo, type, state->operands)) {
    return failure();
  }
  state->addTypes(type);
  return success();
}

static void printGLSLUnaryOp(Operation *unaryOp, OpAsmPrinter *printer) {
  *printer << unaryOp->getName() << ' ' << *unaryOp->getOperand(0) << " : "
           << unaryOp->getOperand(0)->getType();
}

namespace mlir {
namespace spirv {

#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/SPIRVGLSLOps.cpp.inc"

} // namespace spirv
} // namespace mlir
