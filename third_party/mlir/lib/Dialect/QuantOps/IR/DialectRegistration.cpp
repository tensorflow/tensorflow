//===- DialectRegistration.cpp - Register Quantization dialect ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/QuantOps.h"

using namespace mlir;
using namespace mlir::quant;

// Static initialization for Quantization dialect registration.
static mlir::DialectRegistration<QuantizationDialect> QuantizationOps;
