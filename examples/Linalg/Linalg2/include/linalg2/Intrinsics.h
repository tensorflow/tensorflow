//===- Intrinsics.h - Linalg intrinsics definitions -----------------------===//
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

#ifndef LINALG2_INTRINSICS_H_
#define LINALG2_INTRINSICS_H_

#include "linalg1/Intrinsics.h"
#include "linalg2/Ops.h"

namespace linalg {
namespace intrinsics {
using dot = mlir::edsc::intrinsics::OperationBuilder<DotOp>;
using matmul = mlir::edsc::intrinsics::OperationBuilder<MatmulOp>;
using matvec = mlir::edsc::intrinsics::OperationBuilder<MatvecOp>;
} // namespace intrinsics
} // namespace linalg

#endif // LINALG2_INTRINSICS_H_
