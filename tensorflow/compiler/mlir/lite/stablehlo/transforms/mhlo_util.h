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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_MHLO_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_MHLO_UTIL_H_

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace TFL {
namespace mhlo {

std::vector<std::string> GetAcceptedDialects();

// Can we find the given `dialect_name` in the `accepted_dialects`?
bool IsAcceptedDialect(llvm::StringRef dialect_name,
                       const std::vector<std::string> &accepted_dialects);

// Is MHLO op allowed in the TF to MHLO conversion result?
bool IsMhloOpAllowed(StringRef op_name);

// The consolidated logic to verify if each final op is acceptable or not.
// Also see `PrintOpStatsPass` and `CheckAcceptedOpsPass`.
bool IsAcceptedOp(llvm::StringRef dialect_name, llvm::StringRef op_name,
                  const std::vector<std::string> &accepted_dialects);

// Adds patterns which map TF Ops to MHLO Ops.
inline void PopulateTFToMhloPatterns(
    MLIRContext *context, bool legalize_chlo,
    llvm::Optional<StringRef> tf2xla_fallback_device_type, bool prefer_tf2xla,
    RewritePatternSet *patterns) {
  // Add TF->HLO legalization patterns.
  ::mlir::mhlo::PopulateLegalizeTfPatterns(context, patterns);

  // Add TF->TF lowering patterns.
  TF::PopulateTFLoweringBeforeHLOPatterns(context, patterns);

  if (tf2xla_fallback_device_type) {
    // Adding fallback Tf2XlaPatterns is needed to make the patterns work.
    // Add TF->HLO legalization patterns via TF2XLA fallback.
    ::mlir::mhlo::PopulateLegalizeTfWithTf2XlaPatterns(
        tf2xla_fallback_device_type.getValue(), *patterns, context,
        prefer_tf2xla);
  }

  // Populate with CHLO->HLO lowerings to account for TF ops legalized to
  // client HLO (CHLO) first.
  // https://github.com/tensorflow/mlir-hlo
  if (legalize_chlo) {
    chlo::populateDecomposeChloPatterns(context, patterns);
    chlo::populateChloBroadcastingPatterns(context, patterns);
  }
  // ConstantLike op is convenient to create splat constants, but is
  // canonicalized to plain HLO constant if statically shaped. Add the
  // canonicalization pattern to pattern list to enable multi-hop lowering.
  ::mlir::chlo::ConstantLikeOp::getCanonicalizationPatterns(*patterns, context);
}

}  // namespace mhlo
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_MHLO_UTIL_H_
