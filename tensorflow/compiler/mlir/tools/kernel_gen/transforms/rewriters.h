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

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_REWRITERS_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_REWRITERS_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace bufferization {
class BufferizeTypeConverter;
}
class ConversionTarget;
class LLVMTypeConverter;
class MLIRContext;
class RewritePatternSet;
class TypeConverter;

namespace kernel_gen {
namespace tf_framework {

/// Collects a set of patterns to convert from the TF Framework dialect to LLVM.
void PopulateTFFrameworkToLLVMConversionPatterns(LLVMTypeConverter *converter,
                                                 RewritePatternSet *patterns);

/// Collects a set of patterns to rewrite functions for use with TF framework
/// and also replace `alloc`, `dealloc` and `assert`.
void PopulateEmbedTFFrameworkPatterns(RewritePatternSet *patterns);
void PopulateEmbedTFFrameworkAssertPattern(RewritePatternSet *patterns);

}  // namespace tf_framework

namespace transforms {

/// Collects a set of patterns that bufferize operations from the standard and
/// other dialects.
void populateExtraBufferizeDialects(DialectRegistry &registry);
void populateExtraBufferizePatterns(
    ConversionTarget &target, MLIRContext *context,
    bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns);

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_REWRITERS_H_
