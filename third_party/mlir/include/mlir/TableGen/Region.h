//===- TGRegion.h - TableGen region definitions -----------------*- C++ -*-===//
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
