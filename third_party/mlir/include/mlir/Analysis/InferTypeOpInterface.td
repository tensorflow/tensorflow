//===- InferTypeOpInterface.td - Infer Type interfaces -----*- tablegen -*-===//
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
// This file contains a set of interfaces that can be used to define information
// related to type inference.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INFERTYPEOPINTERFACE
#define MLIR_INFERTYPEOPINTERFACE

include "mlir/IR/OpBase.td"

// OpInterface to compute the return type of an operation. The arguments match
// those in Operation::create with the exception that the location is optional
// (if no location is provided, then the method will not emit an error on
// mismatch).
def InferTypeOpInterface : OpInterface<"InferTypeOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that could be used during op construction, verification or
    type inference.
  }];

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{Returns the return types that an op would generate.

      The method takes an optional location which, if set, will be used to
      report errors on. The operands and attributes correspond to those with
      which an Operation would be created (e.g., as used in Operation::create)
      and the regions of the op.
      }],
      /*retTy=*/"SmallVector<Type, 2>",
      /*methodName=*/"inferReturnTypes",
      /*args=*/(ins "llvm::Optional<Location>":$location,
                    "ArrayRef<Value*>":$operands,
                    "ArrayRef<NamedAttribute>":$attributes,
                    "ArrayRef<Region>":$regions)
    >,
    StaticInterfaceMethod<
      /*desc=*/"Returns whether two array of types are compatible result types"
               " for an op.",
      /*retTy=*/"bool",
      /*methodName=*/"isCompatibleReturnTypes",
      /*args=*/(ins "ArrayRef<Type>":$lhs, "ArrayRef<Type>":$rhs),
      [{
        return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
      }]
    >,
  ];
}

// Default implementations for some of the interface methods above:
// - compatibleReturnTypes returns whether strictly true.
def InferTypeOpInterfaceDefault : NativeOpTrait<"TypeOpInterfaceDefault">;

#endif // MLIR_INFERTYPEOPINTERFACE
