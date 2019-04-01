//===- Dialect.cpp - Implementation of the linalg dialect -----------------===//
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
// This file implements a simple Linalg dialect to which we gradually add
// complexity.
//
//===----------------------------------------------------------------------===//

#include "linalg/Dialect.h"
#include "linalg/RangeOp.h"
#include "linalg/RangeType.h"
#include "linalg/SliceOp.h"
#include "linalg/ViewOp.h"
#include "linalg/ViewType.h"
#include "mlir/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

using llvm::raw_ostream;
using llvm::StringRef;
using mlir::Location;
using mlir::Type;

using namespace linalg;

Type LinalgDialect::parseType(StringRef spec, Location loc) const {
  llvm_unreachable("Unhandled linalg dialect parsing");
  return Type();
}

/// RangeType prints as just "range".
static void print(RangeType rt, raw_ostream &os) { os << "range"; }

/// ViewType prints as:
///
/// ```{.mlir}
///   view<i8xf32xi1>
/// ```
///
/// or
///
/// ```{.mlir}
///   view<0xf32>
/// ```
///
/// for 0-D views (a.k.a pointer to a scalar value).
static void print(linalg::ViewType rt, raw_ostream &os) {
  os << "view<";
  if (rt.getRank() > 0) {
    for (unsigned i = 0, e = rt.getRank(); i < e; ++i) {
      os << rt.getElementType() << ((i == e - 1) ? "" : "x");
    }
  } else {
    os << "0x" << rt.getElementType();
  }
  os << ">";
}

void LinalgDialect::printType(Type type, raw_ostream &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled linalg type");
  case LinalgTypes::Range:
    print(type.cast<RangeType>(), os);
    break;
  case linalg::LinalgTypes::View:
    print(type.cast<linalg::ViewType>(), os);
    break;
  }
}
