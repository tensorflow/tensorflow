//===- Intrinsics.h - MLIR EDSC Intrinsics for Linalg -----------*- C++ -*-===//
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
#ifndef MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using linalg_fill = OperationBuilder<linalg::FillOp>;
using linalg_yield = OperationBuilder<linalg::YieldOp>;

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_
