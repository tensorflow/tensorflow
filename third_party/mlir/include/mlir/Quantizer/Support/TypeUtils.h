//===- TypeUtils.h - Helper function for manipulating types -----*- C++ -*-===//
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
// This file defines various helper functions for manipulating types. The
// process of quantizing typically involves a number of type manipulations
// that are not very common elsewhere, and it is best to name them and define
// them here versus inline in the rest of the tool.
//
//===----------------------------------------------------------------------===//

#ifndef THIRD_PARTY_MLIR_EDGE_FXPSOLVER_SUPPORT_TYPEUTILS_H_
#define THIRD_PARTY_MLIR_EDGE_FXPSOLVER_SUPPORT_TYPEUTILS_H_

#include "mlir/IR/Types.h"

namespace mlir {
namespace quantizer {

/// Given an arbitrary container or primitive type, returns the element type,
/// where the element type is just the type for non-containers.
Type getElementOrPrimitiveType(Type t);

} // namespace quantizer
} // namespace mlir

#endif // THIRD_PARTY_MLIR_EDGE_FXPSOLVER_SUPPORT_TYPEUTILS_H_
