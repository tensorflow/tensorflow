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

#ifndef LINALG_OPS
include "mlir/Dialect/Linalg/IR/LinalgOps.td"
#endif // LINALG_OPS
#ifndef LINALG_LIBRARY_OPS
include "mlir/Dialect/Linalg/IR/LinalgLibraryOps.td"
#endif // LINALG_LIBRARY_OPS

def HasNoLinalgTransformMarker : CPred<[{
  !$0.getAttrOfType<StringAttr>(LinalgTransforms::kLinalgTransformMarker)
}]>;

class HasLinalgTransformMarker<string str> : CPred<[{
  $0.getAttrOfType<StringAttr>(
    LinalgTransforms::kLinalgTransformMarker).getValue() == "}] # str # [{"}]>;

class IsProducedByOpOfType<string str> :
  CPred<"isProducedByOpOfType<" # str # ">($0, $1)">;

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
class TileLinalgOp<list<int> sizes, string value> : NativeCodeCall<
  "if (failed(tileLinalgOpAndSetMarker($_builder, $0, {" #
  StrJoinInt<sizes>.result # "}, \"" # value # "\")))" #
  "  return matchFailure();">;

#endif // LINALG_TRANSFORMS
