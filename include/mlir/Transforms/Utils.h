//===- Utils.h - General transformation utilities ---------------*- C++ -*-===//
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
// This header file defines prototypes for various transformation utilities for
// memref's and non-loop IR structures. These are not passes by themselves but
// are used either by passes, optimization sequences, or in turn by other
// transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_UTILS_H
#define MLIR_TRANSFORMS_UTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {

class ForInst;
class FuncBuilder;
class Location;
class Module;
class OperationInst;

class Function;

/// Replaces all uses of oldMemRef with newMemRef while optionally remapping the
/// old memref's indices using the supplied affine map, 'indexRemap'. The new
/// memref could be of a different shape or rank. 'extraIndices' provides
/// additional access indices to be added to the start. 'indexRemap' remaps
/// indices of the old memref access to a new set of indices that are used to
/// index the memref. Additional input operands to indexRemap can be optionally
/// provided, and they are added at the start of its input list. 'indexRemap' is
/// expected to have only dimensional inputs, and the number of its inputs equal
/// to extraOperands.size() plus rank of the memref.  'extraOperands' is an
/// optional argument that corresponds to additional operands (inputs) for
/// indexRemap at the beginning of its input list. An additional optional
/// argument 'domInstFilter' restricts the replacement to only those operations
/// that are dominated by the former. Returns true on success and false if the
/// replacement is not possible (whenever a memref is used as an operand in a
/// non-deferencing scenario). See comments at function definition for an
/// example.
//  Ex: to replace load %A[%i, %j] with load %Abuf[%t mod 2, %ii - %i, %j]:
//  The SSA value corresponding to '%t mod 2' should be in 'extraIndices', and
//  index remap will perform (%i, %j) -> (%ii - %i, %j), i.e., indexRemap = (d0,
//  d1, d2) -> (d0 - d1, d2), and %ii will be the extra operand. Without any
//  extra operands, note that 'indexRemap' would just be applied to existing
//  indices (%i, %j).
//  TODO(bondhugula): allow extraIndices to be added at any position.
bool replaceAllMemRefUsesWith(const Value *oldMemRef, Value *newMemRef,
                              ArrayRef<Value *> extraIndices = {},
                              AffineMap indexRemap = AffineMap::Null(),
                              ArrayRef<Value *> extraOperands = {},
                              const Instruction *domInstFilter = nullptr);

/// Creates and inserts into 'builder' a new AffineApplyOp, with the number of
/// its results equal to the number of operands, as a composition
/// of all other AffineApplyOps reachable from input parameter 'operands'. If
/// different operands were drawing results from multiple affine apply ops,
/// these will also be collected into a single (multi-result) affine apply op.
/// The final results of the composed AffineApplyOp are returned in output
/// parameter 'results'. Returns the affine apply op created.
OperationInst *
createComposedAffineApplyOp(FuncBuilder *builder, Location loc,
                            ArrayRef<Value *> operands,
                            ArrayRef<OperationInst *> affineApplyOps,
                            SmallVectorImpl<Value *> *results);

/// Given an operation instruction, inserts a new single affine apply operation,
/// that is exclusively used by this operation instruction, and that provides
/// all operands that are results of an affine_apply as a function of loop
/// iterators and program parameters and whose results are.
///
/// Before
///
/// for %i = 0 to #map(%N)
///   %idx = affine_apply (d0) -> (d0 mod 2) (%i)
///   send %A[%idx], ...
///   %v = "compute"(%idx, ...)
///
/// After
///
/// for %i = 0 to #map(%N)
///   %idx = affine_apply (d0) -> (d0 mod 2) (%i)
///   send %A[%idx], ...
///   %idx_ = affine_apply (d0) -> (d0 mod 2) (%i)
///   %v = "compute"(%idx_, ...)

/// This allows the application of  different transformations on send and
/// compute (for eg.  / different shifts/delays)
///
/// Returns nullptr if none of the operands were the result of an affine_apply
/// and thus there was no affine computation slice to create. Returns the newly
/// affine_apply operation instruction otherwise.
OperationInst *createAffineComputationSlice(OperationInst *opInst);

/// Folds the lower and upper bounds of a 'for' inst to constants if possible.
/// Returns false if the folding happens for at least one bound, true otherwise.
bool constantFoldBounds(ForInst *forInst);

/// Replaces (potentially nested) function attributes in the operation "op"
/// with those specified in "remappingTable".
void remapFunctionAttrs(
    OperationInst &op, const DenseMap<Attribute, FunctionAttr> &remappingTable);

/// Replaces (potentially nested) function attributes all operations of the
/// Function "fn" with those specified in "remappingTable".
void remapFunctionAttrs(
    Function &fn, const DenseMap<Attribute, FunctionAttr> &remappingTable);

/// Replaces (potentially nested) function attributes in the entire module
/// with those specified in "remappingTable".  Ignores external functions.
void remapFunctionAttrs(
    Module &module, const DenseMap<Attribute, FunctionAttr> &remappingTable);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_UTILS_H
