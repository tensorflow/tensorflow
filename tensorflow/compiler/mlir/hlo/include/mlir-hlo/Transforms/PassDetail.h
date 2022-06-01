/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_TRANSFORMS_PASSDETAIL_H
#define MLIR_HLO_TRANSFORMS_PASSDETAIL_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncDialect;
class FuncOp;
}  // end namespace func

namespace arith {
class ArithmeticDialect;
}  // end namespace arith

namespace memref {
class MemRefDialect;
}  // end namespace memref

#define GEN_PASS_CLASSES
#include "mlir-hlo/Transforms/passes.h.inc"

}  // end namespace mlir

#endif  // MLIR_HLO_TRANSFORMS_PASSDETAIL_H
