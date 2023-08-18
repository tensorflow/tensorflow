/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSOR_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSOR_TRANSFORMS_PASSES_H_

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {

namespace arith {
class ArithDialect;
}

namespace linalg {
class LinalgDialect;
}

namespace math {
class MathDialect;
}

namespace scf {
class SCFDialect;
}

namespace TFL {
class TFLDialect;
}

namespace tensor {
class TensorDialect;

void populateLegalizeTFLPatterns(MLIRContext* ctx, RewritePatternSet& patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFLPass();


#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#define GEN_PASS_DECL_TENSORLEGALIZETFLPASS

#include "tensorflow/compiler/mlir/tensor/transforms/passes.h.inc"

}  // namespace tensor
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSOR_TRANSFORMS_PASSES_H_

