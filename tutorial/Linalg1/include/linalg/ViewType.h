//===- ViewType.h - Linalg ViewType definition --------------------------===//
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

#ifndef LINALG_VIEWTYPE_H_
#define LINALG_VIEWTYPE_H_

#include "linalg/Types.h"

namespace linalg {

/// A ViewType represents a range abstraction on top of an underlying storage
/// type. It is parameterizable by the underlying element type and the rank of
/// the view.
class ViewType
    : public mlir::Type::TypeBase<ViewType, mlir::Type, ViewTypeStorage> {
public:
  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this type.
  //////////////////////////////////////////////////////////////////////////////
  // Used to implement llvm-style cast.
  using Base::Base;
  // Used to implement llvm-style cast.
  static bool kindof(unsigned kind) { return kind == LinalgTypes::View; }
  /// Construction hook.
  static ViewType get(mlir::MLIRContext *context, mlir::Type elementType,
                      unsigned rank);

  //////////////////////////////////////////////////////////////////////////////
  // Type-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  /// Return the underlying elemental type.
  mlir::Type getElementType();
  /// Return the rank of the view.
  /// This is the number of indexings needed to reach an underlying element.
  unsigned getRank();
};

} // namespace linalg

#endif // LINALG_VIEWTYPE_H_
