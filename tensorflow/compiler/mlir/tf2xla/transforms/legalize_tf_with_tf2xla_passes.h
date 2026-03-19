/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_LEGALIZE_TF_WITH_TF2XLA_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_LEGALIZE_TF_WITH_TF2XLA_PASSES_H_

#include <memory>
#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {

namespace func {
class FuncOp;
}
class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;

namespace mhlo {

/// Converter to be used along with the fallback Tf2Xla patterns below.
class Tf2XlaTypeConverter : public TypeConverter {
 public:
  Tf2XlaTypeConverter();
};

/// Adds the TF to XLA via TF2XLA rewrite patterns to the pattern list.
/// `prefer_tf2xla` means an op will be included iff it is not in
/// `MlirLegalizedUnderPreferTf2XlaSet`. `!prefer_tf2xla` mean an op will be
/// included if there is no native MLIR legalization for the op.
void PopulateLegalizeTfWithTf2XlaPatterns(llvm::StringRef device_type,
                                          RewritePatternSet& patterns,
                                          MLIRContext* ctx,
                                          Tf2XlaTypeConverter& converter,
                                          bool prefer_tf2xla = false);


}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_LEGALIZE_TF_WITH_TF2XLA_PASSES_H_
