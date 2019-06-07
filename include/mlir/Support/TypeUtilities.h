//===- TypeUtilities.h - Helper function for type queries -------*- C++ -*-===//
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
// This file defines generic type utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TYPEUTILITIES_H
#define MLIR_SUPPORT_TYPEUTILITIES_H

namespace mlir {

class Attribute;
class Type;
class Value;

/// Return the element type or return the type itself.
Type getElementTypeOrSelf(Type type);

/// Return the element type or return the type itself.
Type getElementTypeOrSelf(Attribute attr);
Type getElementTypeOrSelf(Value *val);
Type getElementTypeOrSelf(Value &val);

} // end namespace mlir

#endif // MLIR_SUPPORT_TYPEUTILITIES_H
