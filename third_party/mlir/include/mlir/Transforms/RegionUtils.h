//===- RegionUtils.h - Region-related transformation utilities --*- C++ -*-===//
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

#ifndef MLIR_TRANSFORMS_REGIONUTILS_H_
#define MLIR_TRANSFORMS_REGIONUTILS_H_

#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {

/// Check if all values in the provided range are defined above the `limit`
/// region.  That is, if they are defined in a region that is a proper ancestor
/// of `limit`.
template <typename Range>
bool areValuesDefinedAbove(Range values, Region &limit) {
  for (ValuePtr v : values)
    if (!v->getParentRegion()->isProperAncestor(&limit))
      return false;
  return true;
}

/// Replace all uses of `orig` within the given region with `replacement`.
void replaceAllUsesInRegionWith(ValuePtr orig, ValuePtr replacement,
                                Region &region);

/// Calls `callback` for each use of a value within `region` or its descendants
/// that was defined at the ancestors of the `limit`.
void visitUsedValuesDefinedAbove(Region &region, Region &limit,
                                 function_ref<void(OpOperand *)> callback);

/// Calls `callback` for each use of a value within any of the regions provided
/// that was defined in one of the ancestors.
void visitUsedValuesDefinedAbove(MutableArrayRef<Region> regions,
                                 function_ref<void(OpOperand *)> callback);

/// Fill `values` with a list of values defined at the ancestors of the `limit`
/// region and used within `region` or its descendants.
void getUsedValuesDefinedAbove(Region &region, Region &limit,
                               llvm::SetVector<ValuePtr> &values);

/// Fill `values` with a list of values used within any of the regions provided
/// but defined in one of the ancestors.
void getUsedValuesDefinedAbove(MutableArrayRef<Region> regions,
                               llvm::SetVector<ValuePtr> &values);

/// Run a set of structural simplifications over the given regions. This
/// includes transformations like unreachable block elimination, dead argument
/// elimination, as well as some other DCE. This function returns success if any
/// of the regions were simplified, failure otherwise.
LogicalResult simplifyRegions(MutableArrayRef<Region> regions);

} // namespace mlir

#endif // MLIR_TRANSFORMS_REGIONUTILS_H_
