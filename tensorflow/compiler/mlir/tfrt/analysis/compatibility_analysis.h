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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_ANALYSIS_COMPATIBILITY_ANALYSIS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_ANALYSIS_COMPATIBILITY_ANALYSIS_H_

#include "mlir/IR/Module.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/analysis/analysis.pb.h"

namespace tensorflow {

// Analyze a MLIR module in tf dialect.
mlir::tfrt::CompatibilityAnalysisProto AnalyzeTFCompatibility(
    mlir::ModuleOp op);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_ANALYSIS_COMPATIBILITY_ANALYSIS_H_
