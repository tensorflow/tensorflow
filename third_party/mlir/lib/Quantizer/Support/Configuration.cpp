//===- Configuration.cpp - Configuration object base classes --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Support/Configuration.h"

#include <limits>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::quantizer;

TargetConfiguration::TargetConfiguration(SolverContext &context) {}

void TargetConfiguration::addOpHandlerByName(StringRef name, OpHandlerFn fn) {
  opHandlers[name] = fn;
}

void TargetConfiguration::addRequireStatsOpByName(StringRef opName) {
  requireStatsOpNames.insert(opName);
}

bool TargetConfiguration::isRequireStatsOp(Operation *op) const {
  return requireStatsOpNames.find(op->getName().getStringRef()) !=
         requireStatsOpNames.end();
}

void TargetConfiguration::handleOp(Operation *op, CAGSlice &cag) const {
  auto found_it = opHandlers.find(op->getName().getStringRef());
  if (found_it != opHandlers.end())
    found_it->second(op, cag);
}
