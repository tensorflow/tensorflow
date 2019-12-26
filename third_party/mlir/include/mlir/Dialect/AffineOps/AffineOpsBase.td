//===- AffineOpsBase.td - Affine operation definitions -----*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines base support for MLIR affine operations.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINE_OPS_BASE
#define AFFINE_OPS_BASE

include "mlir/IR/OpBase.td"

// Attributes containing affine maps.
def AffineMapAttr : Attr<
    CPred<"$_self.isa<AffineMapAttr>()">, "AffineMap attribute"> {
  let storageType = [{ AffineMapAttr }];
  let returnType = [{ AffineMap }];
  let constBuilderCall = "AffineMapAttr::get($0)";
}

def AffineMapArrayAttr : TypedArrayAttrBase<AffineMapAttr,
                                      "AffineMap array attribute"> {
  let constBuilderCall = "$_builder.getAffineMapArrayAttr($0)";
}

#endif // AFFINE_OPS_BASE
