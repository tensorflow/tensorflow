//===- LoopLikeInterface.td - LoopLike interface -----------*- tablegen -*-===//
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
// Defines the interface for loop-like operations as used by LICM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LOOPLIKEINTERFACE
#define MLIR_LOOPLIKEINTERFACE

include "mlir/IR/OpBase.td"

def LoopLikeOpInterface : OpInterface<"LoopLikeOpInterface"> {
  let description = [{
    Encodes properties of a loop. Operations that implement this interface will
    be considered by loop-invariant code motion.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns true if the given value is defined outside of the loop.
        A sensible implementation could be to check whether the value's defining
        operation lies outside of the loops body region. If the loop uses
        explicit capture of dependencies, an implementation could check whether
        the value corresponds to a captured dependency.
      }],
      "bool", "isDefinedOutsideOfLoop", (ins "Value *":$value)
    >,
    InterfaceMethod<[{
        Returns the region that makes up the body of the loop and should be
        inspected for loop-invariant operations.
      }],
      "Region &", "getLoopBody"
    >,
    InterfaceMethod<[{
        Moves the given vector of operations out of the loop. The vector is
        sorted topologically.
      }],
      "LogicalResult", "moveOutOfLoop", (ins "ArrayRef<Operation *>":$ops)
    >,
  ];
}

#endif // MLIR_LOOPLIKEINTERFACE
