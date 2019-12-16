//===- TestVectorTransformPatterns.td - Test patterns ---*- tablegen ----*-===//
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
// This is the pattern definition file for declarative Vector transformations
// tests.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_VECTOR_TRANSFORMS_PATTERNS
#define TEST_VECTOR_TRANSFORMS_PATTERNS

include "mlir/Dialect/StandardOps/Ops.td"
include "mlir/Dialect/VectorOps/VectorOps.td"
include "mlir/Dialect/VectorOps/VectorTransformPatterns.td"

def : Pat<(AddFOp:$op_results $a, $b),
          (UnrollVectorOp<[2, 2]> $op_results, $a, $b),
          [(Constraint<HasShape<[4, 2]>> $a)]>;

def : Pat<(AddFOp:$op_results $a, $b),
          (UnrollVectorOp<[2, 2]> $op_results, $a, $b),
          [(Constraint<HasShape<[4, 4]>> $a)]>;

// TODO(andydavis) Add Constraints on lhs/rhs shapes.
def : Pat<(Vector_ContractionOp:$op_results $a, $b, $c, $masks, $attr0, $attr1),
          (UnrollVectorOp<[2, 2, 2]> $op_results, $a, $b, $c),
          [(Constraint<HasShape<[4, 4]>> $c)]>;

#endif // TEST_VECTOR_TRANSFORMS_PATTERNS
