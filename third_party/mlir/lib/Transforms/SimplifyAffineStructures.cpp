//===- SimplifyAffineStructures.cpp ---------------------------------------===//
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
// This file implements a pass to simplify affine structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#define DEBUG_TYPE "simplify-affine-structure"

using namespace mlir;

namespace {

/// Simplifies affine maps and sets appearing in the operations of the Function.
/// This part is mainly to test the simplifyAffineExpr method. In addition,
/// all memrefs with non-trivial layout maps are converted to ones with trivial
/// identity layout ones.
struct SimplifyAffineStructures
    : public FunctionPass<SimplifyAffineStructures> {
  void runOnFunction() override;

  /// Utility to simplify an affine attribute and update its entry in the parent
  /// operation if necessary.
  template <typename AttributeT>
  void simplifyAndUpdateAttribute(Operation *op, Identifier name,
                                  AttributeT attr) {
    auto &simplified = simplifiedAttributes[attr];
    if (simplified == attr)
      return;

    // This is a newly encountered attribute.
    if (!simplified) {
      // Try to simplify the value of the attribute.
      auto value = attr.getValue();
      auto simplifiedValue = simplify(value);
      if (simplifiedValue == value) {
        simplified = attr;
        return;
      }
      simplified = AttributeT::get(simplifiedValue);
    }

    // Simplification was successful, so update the attribute.
    op->setAttr(name, simplified);
  }

  /// Performs basic integer set simplifications. Checks if it's empty, and
  /// replaces it with the canonical empty set if it is.
  IntegerSet simplify(IntegerSet set) {
    FlatAffineConstraints fac(set);
    if (fac.isEmpty())
      return IntegerSet::getEmptySet(set.getNumDims(), set.getNumSymbols(),
                                     &getContext());
    return set;
  }

  /// Performs basic affine map simplifications.
  AffineMap simplify(AffineMap map) {
    MutableAffineMap mMap(map);
    mMap.simplify();
    return mMap.getAffineMap();
  }

  DenseMap<Attribute, Attribute> simplifiedAttributes;
};

} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createSimplifyAffineStructuresPass() {
  return std::make_unique<SimplifyAffineStructures>();
}

void SimplifyAffineStructures::runOnFunction() {
  auto func = getFunction();
  simplifiedAttributes.clear();
  func.walk([&](Operation *opInst) {
    for (auto attr : opInst->getAttrs()) {
      if (auto mapAttr = attr.second.dyn_cast<AffineMapAttr>())
        simplifyAndUpdateAttribute(opInst, attr.first, mapAttr);
      else if (auto setAttr = attr.second.dyn_cast<IntegerSetAttr>())
        simplifyAndUpdateAttribute(opInst, attr.first, setAttr);
    }
  });

  // Turn memrefs' non-identity layouts maps into ones with identity. Collect
  // alloc ops first and then process since normalizeMemRef replaces/erases ops
  // during memref rewriting.
  SmallVector<AllocOp, 4> allocOps;
  func.walk([&](AllocOp op) { allocOps.push_back(op); });
  for (auto allocOp : allocOps) {
    normalizeMemRef(allocOp);
  }
}

static PassRegistration<SimplifyAffineStructures>
    pass("simplify-affine-structures",
         "Simplify affine expressions in maps/sets and normalize memrefs");
