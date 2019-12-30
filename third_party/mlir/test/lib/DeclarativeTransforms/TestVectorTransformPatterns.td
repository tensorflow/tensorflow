//===- TestVectorTransformPatterns.td - Test patterns ---*- tablegen ----*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
