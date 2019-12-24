//===- TGRegion.h - TableGen region definitions -----------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_REGION_H_
#define MLIR_TABLEGEN_REGION_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Constraint.h"

namespace mlir {
namespace tblgen {

// Wrapper class providing helper methods for accessing Region defined in
// TableGen.
class Region : public Constraint {
public:
  using Constraint::Constraint;

  static bool classof(const Constraint *c) { return c->getKind() == CK_Region; }
};

// A struct bundling a region's constraint and its name.
struct NamedRegion {
  StringRef name;
  Region constraint;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_REGION_H_
