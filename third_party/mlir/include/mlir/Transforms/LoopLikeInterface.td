//===- LoopLikeInterface.td - LoopLike interface -----------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
      "bool", "isDefinedOutsideOfLoop", (ins "Value ":$value)
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
