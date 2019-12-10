//===- TestLinalgTransformPatterns.td - Test patterns --*- tablegen ----*-===//
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
// This is the pattern definition file for declarative Linalg transformations
// tests.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LINALG_TRANSFORMS_PATTERNS
#define TEST_LINALG_TRANSFORMS_PATTERNS

include "mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.td"

//===----------------------------------------------------------------------===//
// Test Linalg fusion patterns.
//===----------------------------------------------------------------------===//
def : Pat<(MatmulOp:$consumer $A, $B, $C),
          (TileAndFuseLinalgOp<[100, 150], [0], "L1"> $consumer),
          [
            (Constraint<HasNoLinalgTransformMarker> $consumer),
            (Constraint<IsProducedByOpOfType<"MatmulOp">> $consumer, $A),
          ],
          // In the buffer world there is no use-def chains or dags so benefits
          // cannot be computed automatically from the length of the matched
          // pattern. Instead we specify the benefit ourselves for now.
          // This is not expected to be a big challenge long-term because
          // pattern benefits are akin to feature engineering: features should
          // be learned.
          (addBenefit 1)>;

//===----------------------------------------------------------------------===//
// Linalg tiling patterns.
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Linalg to loops patterns.
//===----------------------------------------------------------------------===//
def : Pattern<(DotOp:$op $a, $b, $c),
              [(LinalgOpToLoops<"DotOp"> $op)],
              [(Constraint<HasLinalgTransformMarker<"REG">> $op)]>;

//===----------------------------------------------------------------------===//
// Linalg to vector contraction patterns.
//===----------------------------------------------------------------------===//
def : Pattern<(GenericOp:$op $_1, $_2, $_3, $_4, $_5, $_6, $_7),
              [(LinalgOpToVectorContraction<"GenericOp"> $op)],
              [(Constraint<HasLinalgTransformMarker<"_marked_matmul_">> $op)]>;

#endif // TEST_LINALG_TRANSFORMS_PATTERNS
