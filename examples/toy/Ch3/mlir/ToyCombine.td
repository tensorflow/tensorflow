//===- ToyCombine.td - Pattern Match Optimizations for Toy -*- tablegen -*-===//
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
// Defines language-specific pattern match optimizations for Toy using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_COMBINE
#define TOY_COMBINE

#ifndef OP_BASE
include "toy/Ops.td"
#endif // OP_BASE

/* Pattern-Match and Rewrite using DRR:
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
*/

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// Reshape(Reshape(x)) = x
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite using Native Code Call
//===----------------------------------------------------------------------===//

// Native Code Calls may be used for more complex transformations using inline
// C++ and C++ helper functions.

// Reshape(Constant(x)) = x'
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(($1->getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------------------------------===//

// DRR allows for constraint checking when the transformation is conditional
// on operand properties.

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0->getType() == $1->getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;

#endif // TOY_COMBINE
