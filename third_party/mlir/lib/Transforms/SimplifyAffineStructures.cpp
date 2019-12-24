//===- SimplifyAffineStructures.cpp ---------------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
