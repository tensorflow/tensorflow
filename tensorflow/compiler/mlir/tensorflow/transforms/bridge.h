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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_BRIDGE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_BRIDGE_H_

#include <string>

#include "absl/base/attributes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/lib/core/status.h"

namespace mlir {
namespace TF {

inline constexpr char kStandardPipelineBefore[] = "standard_pipeline_before";
inline constexpr char kStandardPipelineAfter[] = "standard_pipeline_after";

// Runs all passes involved in transforming or optimizing an MLIR graph without
// any target specialization. When enable_logging is true, enables
// tensorflow::BridgeLogger. When enable_inliner is true, enables the inliner
// pass.
ABSL_DEPRECATED(
    "This is legacy code and is unsupported. Use at your own risk. Use "
    "tf2xla/api/v2/* for specific functionality")
absl::Status RunBridgeWithStandardPipeline(ModuleOp module, bool enable_logging,
                                           bool enable_inliner);
}  // namespace TF

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_BRIDGE_H_
