//===- ConvertStandardToSPIRVPass.h - StdOps to SPIR-V pass -----*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a pass to lower from StandardOps to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRVPASS_H
#define MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRVPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Pass to convert StandardOps to SPIR-V ops.
std::unique_ptr<OpPassBase<ModuleOp>> createConvertStandardToSPIRVPass();

/// Pass to legalize ops that are not directly lowered to SPIR-V.
std::unique_ptr<Pass> createLegalizeStdOpsForSPIRVLoweringPass();

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRVPASS_H
