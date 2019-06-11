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

#ifndef MLIR_LINALG_INTRINSICS_H_
#define MLIR_LINALG_INTRINSICS_H_

#include "mlir/EDSC/Intrinsics.h"

namespace mlir {
namespace linalg {
class DimOp;
class RangeOp;
class SliceOp;
class ViewOp;
namespace intrinsics {
using dim = mlir::edsc::intrinsics::ValueBuilder<linalg::DimOp>;
using range = mlir::edsc::intrinsics::ValueBuilder<RangeOp>;
using slice = mlir::edsc::intrinsics::ValueBuilder<SliceOp>;
using view = mlir::edsc::intrinsics::ValueBuilder<ViewOp>;
} // namespace intrinsics
} // namespace linalg
} // namespace mlir

#endif // MLIR_LINALG_INTRINSICS_H_
