//===- ConvertStandardToSPIRVPass.cpp - Convert Std Ops to SPIR-V Ops -----===//
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
// ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// A pass converting MLIR Standard operations into the SPIR-V dialect.
class ConvertStandardToSPIRVPass
    : public ModulePass<ConvertStandardToSPIRVPass> {
  void runOnModule() override;
};
} // namespace

void ConvertStandardToSPIRVPass::runOnModule() {
  OwningRewritePatternList patterns;
  auto module = getModule();

  SPIRVTypeConverter typeConverter;
  populateStandardToSPIRVPatterns(module.getContext(), typeConverter, patterns);
  ConversionTarget target(*(module.getContext()));
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addLegalOp<FuncOp>();

  if (failed(applyPartialConversion(module, target, patterns))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertStandardToSPIRVPass() {
  return std::make_unique<ConvertStandardToSPIRVPass>();
}

static PassRegistration<ConvertStandardToSPIRVPass>
    pass("convert-std-to-spirv", "Convert Standard Ops to SPIR-V dialect");
