//===- Configuration.cpp - Configuration object base classes --------------===//
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
