//===- FxpMathOps.cpp - Op implementation for FxpMathOps ------------------===//
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

#include "mlir/Dialect/FxpMathOps/FxpMathOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::fxpmath;

#define GET_OP_CLASSES
#include "mlir/Dialect/FxpMathOps/FxpMathOps.cpp.inc"

FxpMathOpsDialect::FxpMathOpsDialect(MLIRContext *context)
    : Dialect(/*name=*/"fxpmath", context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/FxpMathOps/FxpMathOps.cpp.inc"
      >();
}
