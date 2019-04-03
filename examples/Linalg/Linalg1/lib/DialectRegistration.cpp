//===- DialectRegistration.cpp - Registration of the Linalg dialect -------===//
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
// This file registers the Linalg dialect and should live in a standalone
// library. Linking with this library will create a static global object that
// performs dialect registration.
//
//===----------------------------------------------------------------------===//

#include "linalg1/Dialect.h"
#include "linalg1/Ops.h"
#include "linalg1/Types.h"

using namespace mlir;
using namespace linalg;

LinalgDialect::LinalgDialect(MLIRContext *context)
    : Dialect("linalg", context) {
  addTypes<RangeType, ViewType>();
  addOperations<RangeOp, SliceOp, ViewOp>();
}

// Dialect registration triggers the creation of a `LinalgDialect` object which
// adds the proper types and operations to the dialect.
static mlir::DialectRegistration<LinalgDialect> LinalgOps;
