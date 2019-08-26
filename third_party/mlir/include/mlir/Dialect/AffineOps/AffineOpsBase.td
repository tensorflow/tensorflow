//===- AffineOpsBase.td - Affine operation definitions -----*- tablegen -*-===//
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
// Defines base support for MLIR affine operations.
//
//===----------------------------------------------------------------------===//

#ifdef AFFINE_OPS_BASE
#else
#define AFFINE_OPS_BASE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

// Attributes containing affine maps.
def AffineMapAttr : Attr<
    CPred<"$_self.isa<AffineMapAttr>()">, "AffineMap attribute"> {
  let storageType = [{ AffineMapAttr }];
  let returnType = [{ AffineMap }];
  let constBuilderCall = "$_builder.getAffineMapAttr($0)";
}

def AffineMapArrayAttr : TypedArrayAttrBase<AffineMapAttr,
                                      "AffineMap array attribute"> {
  let constBuilderCall = "$_builder.getAffineMapArrayAttr($0)";
}

#endif // AFFINE_OPS_BASE
