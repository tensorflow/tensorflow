//===- Attributes.cpp - MLIR Attribute Implementation ---------------------===//
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

#include "mlir/IR/Attributes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/STLExtras.h"
using namespace mlir;

void Attribute::print(raw_ostream &os) const {
  switch (getKind()) {
  case Kind::Bool:
    os << (cast<BoolAttr>(this)->getValue() ? "true" : "false");
    break;
  case Kind::Integer:
    os << cast<IntegerAttr>(this)->getValue();
    break;
  case Kind::Float:
    // FIXME: this isn't precise, we should print with a hex format.
    os << cast<FloatAttr>(this)->getValue();
    break;
  case Kind::String:
    // FIXME: should escape the string.
    os << '"' << cast<StringAttr>(this)->getValue() << '"';
    break;
  case Kind::Array: {
    auto elts = cast<ArrayAttr>(this)->getValue();
    os << '[';
    interleave(elts,
                [&](Attribute *attr) { attr->print(os); },
                [&]() { os << ", "; });
    os << ']';
    break;
  }
  }
}

void Attribute::dump() const {
  print(llvm::errs());
}
