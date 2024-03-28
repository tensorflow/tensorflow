/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_SPARSECORE_SPARSECORE_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_SPARSECORE_SPARSECORE_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace TFDevice {

// For architectures that support accelerated embedding lookups, this pass will
// rewrite the graph to use pipelining for better device utilization.
std::unique_ptr<OperationPass<ModuleOp>> CreateEmbeddingSequencingPass();

// This is a strictly sequential and formally correct fallback option for the
// embedding pipelining pass intended for debugging during pipelining
// development.
std::unique_ptr<OperationPass<ModuleOp>> CreateEmbeddingPipeliningPass();

// Passes in the program key to embedding ops, by moving the embedding ops
// after the _TPUCompileMlir op.
std::unique_ptr<OperationPass<func::FuncOp>> CreateEmbeddingProgramKeyPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_EMBEDDINGSEQUENCINGPASS
#define GEN_PASS_DECL_EMBEDDINGPIPELININGPASS
#define GEN_PASS_DECL_EMBEDDINGPROGRAMKEYPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/sparsecore/sparsecore_passes.h.inc"

}  // namespace TFDevice
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_SPARSECORE_SPARSECORE_PASSES_H_
