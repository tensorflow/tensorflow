//===- DialectConstruction.cpp - Construction of the Linalg dialect -------===//
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
// This file implements the constructor for the Linalg Dialect. This is
// explicitly separated from the core library to allow incremental buildup of
// the codebase for the tutorial.
//
//===----------------------------------------------------------------------===//

#include "linalg1/Dialect.h"
#include "linalg1/Types.h"
#include "linalg3/Ops.h"

using namespace linalg;

LinalgDialect::LinalgDialect(mlir::MLIRContext *context)
    : Dialect("linalg", context) {
  addTypes<RangeType, ViewType>();
  addOperations<DotOp, LoadOp, MatvecOp, MatmulOp, RangeOp, SliceOp, StoreOp,
                ViewOp>();
}
