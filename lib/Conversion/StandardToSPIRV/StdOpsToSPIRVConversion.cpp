//===- StdOpsToSPIRVLowering.cpp - Std Ops to SPIR-V dialect conversion ---===//
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
// This file implements a pass to convert MLIR standard ops into the SPIR-V
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToSPIRV/StdOpsToSPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/StandardOps/Ops.h"

using namespace mlir;

namespace {
/// A pass converting MLIR Standard operations into the SPIR-V dialect.
class StdOpsToSPIRVConversionPass
    : public FunctionPass<StdOpsToSPIRVConversionPass> {
  void runOnFunction() override;
};

#include "StdOpsToSPIRVConversion.cpp.inc"
} // namespace

namespace mlir {
void populateStdOpsToSPIRVPatterns(MLIRContext *context,
                                   OwningRewritePatternList &patterns) {
  populateWithGenerated(context, &patterns);
}
} // namespace mlir

void StdOpsToSPIRVConversionPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();

  populateStdOpsToSPIRVPatterns(func.getContext(), patterns);
  applyPatternsGreedily(func, std::move(patterns));
}

FunctionPassBase *mlir::spirv::createStdOpsToSPIRVConversionPass() {
  return new StdOpsToSPIRVConversionPass();
}

static PassRegistration<StdOpsToSPIRVConversionPass>
    pass("std-to-spirv", "Convert Standard Ops to SPIR-V dialect");
