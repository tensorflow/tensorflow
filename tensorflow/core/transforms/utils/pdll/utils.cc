/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/utils/pdll/utils.h"

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/transforms/utils/utils.h"

namespace mlir {
namespace tfg {
namespace {
#include "tensorflow/core/transforms/utils/pdll/PDLLUtils.h.inc"
}  // namespace

static LogicalResult OpHasCpuDeviceImpl(PatternRewriter &rewriter,
                                        Operation *op) {
  return success(util::OpHasDevice(op, tensorflow::DEVICE_CPU));
}

void RegisterPDLLUtils(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerConstraintFunction("OpHasCpuDevice",
                                                       OpHasCpuDeviceImpl);
}

}  // namespace tfg
}  // namespace mlir
