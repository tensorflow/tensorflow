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

#include "linalg1/Dialect.h"
#include "linalg1/Ops.h"
#include "linalg1/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

using llvm::raw_ostream;
using llvm::StringRef;
using namespace mlir;

using namespace linalg;

Type LinalgDialect::parseType(StringRef spec, Location loc) const {
  MLIRContext *context = getContext();
  if (spec == "range")
    return RangeType::get(getContext());

  StringRef str = spec;
  if (str.consume_front("view<")) {
    // Just count the number of ? to get the rank, the type must be f32 for now.
    unsigned rank = 0;
    while (str.consume_front("?x"))
      ++rank;
    if (str.consume_front("bf16>"))
      return ViewType::get(context, FloatType::getBF16(context), rank);
    if (str.consume_front("f16>"))
      return ViewType::get(context, FloatType::getF16(context), rank);
    if (str.consume_front("f32>"))
      return ViewType::get(context, FloatType::getF32(context), rank);
    if (str.consume_front("f64>"))
      return ViewType::get(context, FloatType::getF64(context), rank);
  }
  return (emitError(loc, "unknown Linalg type: " + spec), nullptr);
}

/// RangeType prints as just "range".
static void print(RangeType rt, raw_ostream &os) { os << "range"; }

/// ViewType prints as:
///
/// ```{.mlir}
///   view<?x?xf32>
/// ```
///
/// or
///
/// ```{.mlir}
///   view<?xf32>
/// ```
///
/// for 0-D views (a.k.a pointer to a scalar value).
static void print(linalg::ViewType rt, raw_ostream &os) {
  os << "view<";
  if (rt.getRank() > 0) {
    for (unsigned i = 0, e = rt.getRank(); i < e; ++i) {
      os << "?x";
    }
  }
  os << rt.getElementType();
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
