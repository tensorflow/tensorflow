/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_JITRT_STUB_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_JITRT_STUB_H_

#include <memory>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"

namespace tensorflow {

class TfrtJitRtStub {
 public:
  virtual ~TfrtJitRtStub() = default;

  virtual void RegisterJitRtDialects(mlir::DialectRegistry &registry) {}

  virtual void PopulateJitRtConversionPatterns(
      mlir::ConversionTarget *target, mlir::MLIRContext *context,
      mlir::RewritePatternSet *patterns, CoreRTConverter *corert_converter) {}

  virtual mlir::Value CreateJitRtFallbackCompileKernel(
      mlir::OpBuilder &builder, mlir::ModuleOp module,
      mlir::Value chain_value) {
    return chain_value;
  }

  virtual void AddTfrtJitRtPasses(const TfrtPipelineOptions &options,
                                  mlir::OpPassManager &pm) {}
};

void RegisterTfrtJitRtStub(std::unique_ptr<TfrtJitRtStub> stub);

void RegisterJitRtDialects(mlir::DialectRegistry &registry);

// Helper function for inserting TFRT JitRt dialect conversions.
void PopulateJitRtConversionPatterns(mlir::ConversionTarget *target,
                                     mlir::MLIRContext *context,
                                     mlir::RewritePatternSet *patterns,
                                     CoreRTConverter *corert_converter);

mlir::Value CreateJitRtFallbackCompileKernel(mlir::OpBuilder &builder,
                                             mlir::ModuleOp module,
                                             mlir::Value chain_value);

void AddTfrtJitRtPasses(const TfrtPipelineOptions &options,
                        mlir::OpPassManager &pm);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_JITRT_STUB_H_
