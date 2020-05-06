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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace tensorflow {

// Create a pass that converts MLIR TF dialect to MLIR TFRT CoreRT dialect.
std::unique_ptr<mlir::Pass> CreateTFToCoreRTConversionPass();

// Run TFToCoreRTConversionPass as a free function. Useful for reusing the pass
// logic in a custom pass with additional conversions.
mlir::LogicalResult TFToCoreRTConversionPassRun(
    mlir::MLIRContext* context, mlir::ModuleOp* module,
    mlir::ConversionTarget* target, mlir::OwningRewritePatternList* patterns);

// Create the corert optimization pass.
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateCoreRTOptimizePass();

struct CoreRTPipelineOptions
    : public mlir::PassPipelineOptions<CoreRTPipelineOptions> {
  Option<std::string> default_device{
      *this, "default-device", llvm::cl::desc("default device assignment"),
      llvm::cl::init("cpu")};
  Option<bool> enable_optimizer{
      *this, "enable-optimizer",
      llvm::cl::desc("run optimization passes on corert dialect"),
      llvm::cl::init(false)};
  Option<std::string> force_data_format{
      *this, "force-data-format",
      llvm::cl::desc("force data format for all layout sensitive operations")};
};

// Creates a pipeline of passes that lowers MLIR TF Executor dialect to TF
// dialect for CoreRT purposes.
void CreateTFExecutorToTFPipeline(
    mlir::OpPassManager& pm, const CoreRTPipelineOptions& options);  // NOLINT

// Creates a pipeline of passes that converts MLIR TF Executor dialect to CoreRT
// dialect.
void CreateTFExecutorToCoreRTPipeline(
    mlir::OpPassManager& pm, const CoreRTPipelineOptions& options);  // NOLINT

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_PASSES_H_
