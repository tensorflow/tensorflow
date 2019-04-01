//===- RangeType.h - Linalg RangeType definition --------------------------===//
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

#ifndef LINALG_RANGETYPE_H_
#define LINALG_RANGETYPE_H_

#include "linalg/Types.h"

namespace mlir {
class MLIRContext;
}

namespace linalg {
/// A RangeType is the simplest possible form of a type in MLIR. It represents
/// a minimal range abstraction (min, max, step). Since RangeType is constructed
/// without any additional argument, this example illustrates the minimal
/// amount of information required to implement a new custom MLIR type.
class RangeType : public mlir::Type::TypeBase<RangeType, mlir::Type> {
public:
  // Used to implement llvm-style cast.
  using Base::Base;
  /// Construction hook.
  static RangeType get(mlir::MLIRContext *context) {
    /// Custom, uniqu'ed construction in the mlir::MLIRContext.
    return Base::get(context, LinalgTypes::Range);
  }
  /// Used to implement llvm-style cast.
  static bool kindof(unsigned kind) { return kind == LinalgTypes::Range; }
};

} // namespace linalg

#endif // LINALG_RANGETYPE_H_
