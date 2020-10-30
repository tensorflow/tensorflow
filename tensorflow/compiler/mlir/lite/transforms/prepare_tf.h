#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_TF_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_TF_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace TF {

// Populates TensorFlow lite prepare patterns to prepare for
// legalization.
void PopulatePrepareTfPatterns(MLIRContext *context,
                               OwningRewritePatternList *patterns);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_TF_H_
