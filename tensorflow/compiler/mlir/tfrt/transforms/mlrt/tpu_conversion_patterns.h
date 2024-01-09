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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TPU_CONVERSION_PATTERNS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TPU_CONVERSION_PATTERNS_H_

#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/execute_op_registry.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"

namespace tensorflow {
namespace mlrt_compiler {

inline constexpr char kTfMlrtCustomDevice[] = "tf_mlrt.custom_device";
inline constexpr char kTpuHostDevice[] = "tpu_host_device";

void RegisterTpuDialect(mlir::DialectRegistry& registry);

void PopulateTpuPreParallelizationConversionPatterns(
    mlir::ConversionTarget& target, mlir::RewritePatternSet& patterns,
    const TfrtPipelineOptions& options);

void PopulateTpuConversionPatterns(mlir::ConversionTarget& target,
                                   mlir::RewritePatternSet& patterns,
                                   mlir::TypeConverter& type_converter,
                                   ExecuteOpRegistry& execute_op_registry,
                                   const TfrtPipelineOptions& options);

}  // namespace mlrt_compiler
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_TPU_CONVERSION_PATTERNS_H_
