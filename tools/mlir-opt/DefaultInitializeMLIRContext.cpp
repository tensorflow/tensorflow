//===- DefaultInitializeMLIRContext.cpp - Initialize an MLIR context ------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file is linked into tools that want to use the standard TensorFlow ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/TensorFlow/ControlFlowOps.h"
#include "mlir/TensorFlow/TensorFlowOps.h"

using namespace mlir;

// Register the TFControlFlow ops and TF ops with the MLIRContext.
void initializeMLIRContext(MLIRContext *ctx) {
  TFControlFlow::registerOperations(*ctx);
  TF::registerOperations(*ctx);
}
