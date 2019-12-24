//===- DialectRegistration.cpp - MLIR SPIR-V dialect registration ---------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVDialect.h"

// Static initialization for SPIR-V dialect registration.
static mlir::DialectRegistration<mlir::spirv::SPIRVDialect> spirvDialect;
