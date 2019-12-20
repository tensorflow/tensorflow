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
include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.td"
include "mlir/Dialect/AffineOps/AffineOps.td"

def HasNoLinalgTransformMarker : CPred<[{
  !$0.getAttrOfType<StringAttr>(LinalgTransforms::kLinalgTransformMarker)
}]>;

class HasLinalgTransformMarker<string str> : CPred<[{
  $0.getAttrOfType<StringAttr>(
    LinalgTransforms::kLinalgTransformMarker) &&
  $0.getAttrOfType<StringAttr>(
    LinalgTransforms::kLinalgTransformMarker).getValue() == "}] # str # [{"}]>;

class IsProducedByOpOfType<string str> :
  CPred<"isProducedByOpOfType<" # str # ">($0, $1)">;

class AffineMapDomainHasDim<int n> : CPred<[{
  $0.getAttrOfType<ArrayAttr>(getIndexingMapsAttrName()).getValue()[0].
  cast<AffineMapAttr>().getValue().getNumDims() ==}] # n # [{}]>;

class HasOperandsOfType<string type>: CPred<[{
    llvm::any_of($0.getOperands(),
        [](Value* v) {
          return dyn_cast_or_null<}] # type # [{>(v->getDefiningOp());
        })
}]>;

//===----------------------------------------------------------------------===//
// Linalg fusion patterns.
//===----------------------------------------------------------------------===//
//
// In the future, tile sizes should be derived from op properties + machine
// description but we do not need to wait on this to start having useful
// patterns.
class TileAndFuseLinalgOp<
    list<int> sizes, list<int> operandIndices, string value> : NativeCodeCall<
  "if (failed(tileAndFuseLinalgOpAndSetMarker($_builder, $0, {" #
  StrJoinInt<sizes>.result # "}, {" # StrJoinInt<operandIndices>.result # "}," #
      " \"" # value # "\")))" #
  "  return matchFailure();">;

//===----------------------------------------------------------------------===//
// Linalg tiling patterns.
//===----------------------------------------------------------------------===//
//
// In the future, tile sizes should be derived from op properties + machine
// description but we do not need to wait on this to start having useful
// patterns.
// `permutation` is an optional parameter to specify the ordering of the
// tiled loops. If provided, it must be a list of integers with the same number
// of elements as `sizes`.
class TileLinalgOp<list<int> sizes, string value, list<int> permutation=[]> :
  NativeCodeCall<
    "if (failed(tileLinalgOpAndSetMarker($_builder, $0, {" #
    StrJoinInt<sizes>.result # "}, \"" # value # "\", {" #
    StrJoinInt<permutation>.result # "})))" #
    "  return matchFailure();">;

//===----------------------------------------------------------------------===//
// Linalg to loop patterns.
//===----------------------------------------------------------------------===//
class LinalgOpToLoops<string OpType> : NativeCodeCall<
  "if (failed(linalgOpToLoops<" # OpType # ">($_builder, $0))) " #
  "  return matchFailure();">;

class LinalgOpToAffineLoops<string OpType> : NativeCodeCall<
  "if (failed(linalgOpToAffineLoops<" # OpType # ">($_builder, $0))) " #
  "  return matchFailure();">;

//===----------------------------------------------------------------------===//
// Linalg to vector contraction patterns.
//===----------------------------------------------------------------------===//
class LinalgOpToVectorContraction<string OpType> : NativeCodeCall<
  "if (failed(vectorizeGenericOp($_builder, $0))) " #
  "  return matchFailure();">;

//===----------------------------------------------------------------------===//
// Linalg generic permutation patterns.
//===----------------------------------------------------------------------===//
class PermuteGenericLinalgOp<list<int> permutation, string value> :
  NativeCodeCall<
    "if (failed(permuteGenericLinalgOp($_builder, $0, {" #
    StrJoinInt<permutation>.result # "}, \"" # value # "\"))) " #
    "  return matchFailure();">;

//===----------------------------------------------------------------------===//
// Linalg promote subview operands.
//===----------------------------------------------------------------------===//
class LinalgOpPromoteSubviews<string OpType> : NativeCodeCall<
  "if (failed(linalgOpPromoteSubviews($_builder, $0))) " #
  "  return matchFailure();">;
#endif // LINALG_TRANSFORMS
