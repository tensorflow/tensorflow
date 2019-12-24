//===- VectorTransformPatterns.td - Vector-Vector patterns -*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the pattern definition file for declarative Vector transformations.
//
//===----------------------------------------------------------------------===//

#ifndef VECTOR_TRANSFORM_PATTERNS
#define VECTOR_TRANSFORM_PATTERNS

include "mlir/IR/OpBase.td"

class HasShape<list<int> shape> :
  CPred<"$0->getType().cast<ShapedType>().hasStaticShape({" #
    StrJoinInt<shape>.result # "})">;

class UnrollVectorOp<list<int> factors> : NativeCodeCall<
  "unrollSingleResultOpMatchingType($_builder, $0->getDefiningOp(), " #
    "{" # StrJoinInt<factors>.result # "})">;

#endif // VECTOR_TRANSFORM_PATTERNS
