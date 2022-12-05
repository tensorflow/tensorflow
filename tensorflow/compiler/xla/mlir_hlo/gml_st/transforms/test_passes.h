/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TEST_PASSES_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TEST_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gml_st {

#define GEN_PASS_DECL
#include "gml_st/transforms/test_passes.h.inc"

std::unique_ptr<OperationPass<func::FuncOp>> createTestGmlStLoopPeelingPass();

std::unique_ptr<OperationPass<func::FuncOp>> createTestGmlStLoopTilingPass();

std::unique_ptr<OperationPass<ModuleOp>> createTestGmlStBufferizationPass();

std::unique_ptr<OperationPass<func::FuncOp>> createTestGmlStGreedyFusionPass();

#define GEN_PASS_REGISTRATION
#include "gml_st/transforms/test_passes.h.inc"

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TEST_PASSES_H
