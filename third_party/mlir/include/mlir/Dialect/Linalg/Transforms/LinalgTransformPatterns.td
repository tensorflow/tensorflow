//===- LinalgPatterns.td - Linalg transformation patterns --*- tablegen -*-===//
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
// This is the pattern definition file for declarative Linalg transformation.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_TRANSFORMS
#define LINALG_TRANSFORMS

include "mlir/Dialect/Linalg/IR/LinalgOps.td"
include "mlir/Dialect/Linalg/IR/LinalgLibraryOps.td"

def HasNoLinalgTransformMarker : CPred<[{
  !$0.getAttrOfType<StringAttr>(kLinalgTransformMarker)
}]>;

class HasLinalgTransformMarker<string value> : CPred<[{
  $0.getAttrOfType<StringAttr>(kLinalgTransformMarker).getValue() == "}] #
  value # [{"}]>;

//===----------------------------------------------------------------------===//
// Linalg transformation patterns.
//===----------------------------------------------------------------------===//
class TileLinalgOp<list<int> sizes, string value> : NativeCodeCall<
  "auto res = tileLinalgOperation($_builder, $0, ArrayRef<int64_t>{" #
    StrJoinInt<sizes>.result # "});" # [{
    if (!res)
      return matchFailure();
    res->op.setAttr(kLinalgTransformMarker, StringAttr::get("}] # value #
    [{", $0.getContext()));}]>;

def : Pat<(MatmulOp:$op $A, $B, $C),
          (TileLinalgOp<[2000, 3000, 4000], "L3"> $op),
          [(Constraint<Or<[HasNoLinalgTransformMarker,
                           HasLinalgTransformMarker<"MEM">]>> $op)]>;
def : Pat<(MatmulOp:$op $A, $B, $C),
          (TileLinalgOp<[200, 300, 400], "L2"> $op),
          [(Constraint<HasLinalgTransformMarker<"L3">> $op)]>;
def : Pat<(MatmulOp:$op $A, $B, $C),
          (TileLinalgOp<[20, 30, 40], "L1"> $op),
          [(Constraint<HasLinalgTransformMarker<"L2">> $op)]>;
def : Pat<(MatmulOp:$op $A, $B, $C),
          (TileLinalgOp<[2, 3, 4], "REG"> $op),
          [(Constraint<HasLinalgTransformMarker<"L1">> $op)]>;

def : Pattern<(MatvecOp:$op $A, $b, $c),
              [(TileLinalgOp<[5, 6], "L1"> $op)],
              [(Constraint<HasNoLinalgTransformMarker> $op)]>;

def : Pattern<(DotOp:$op $a, $b, $c),
              [(TileLinalgOp<[8000], "L1"> $op)],
              [(Constraint<Or<[HasNoLinalgTransformMarker,
                               HasLinalgTransformMarker<"MEM">,
                               HasLinalgTransformMarker<"L3">,
                               HasLinalgTransformMarker<"L2">]>> $op)]>;
def : Pattern<(DotOp:$op $a, $b, $c),
              [(TileLinalgOp<[8], "REG"> $op)],
              [(Constraint<HasLinalgTransformMarker<"L1">> $op)]>;

#endif // LINALG_TRANSFORMS
