//===- TestDialect.cpp - MLIR Dialect for Testing -------------------------===//
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

#include "TestDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

TestDialect::TestDialect(MLIRContext *context)
    : Dialect(getDialectName(), context) {
  addOperations<
#define GET_OP_LIST
#include "TestOps.cpp.inc"
      >();
  allowUnknownOperations();
}

// Static initialization for Test dialect registration.
static mlir::DialectRegistration<mlir::TestDialect> testDialect;

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"
