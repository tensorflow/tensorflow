//===- TypeUtils.cpp - Helper function for manipulating types -------------===//
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

#include "mlir/Quantizer/Support/TypeUtils.h"

#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::quantizer;

Type mlir::quantizer::getElementOrPrimitiveType(Type t) {
  if (auto sType = t.dyn_cast<ShapedType>()) {
    return sType.getElementType();
  } else {
    return t;
  }
}
