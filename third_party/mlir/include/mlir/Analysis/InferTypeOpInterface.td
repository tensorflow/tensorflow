//===- InferTypeOpInterface.td - Infer Type interfaces -----*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
      /*desc=*/[{Infer the return types that an op would generate.

      The method takes an optional location which, if set, will be used to
      report errors on. The operands and attributes correspond to those with
      which an Operation would be created (e.g., as used in Operation::create)
      and the regions of the op.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"inferReturnTypes",
      /*args=*/(ins "Optional<Location>":$location,
                    "ValueRange":$operands,
                    "ArrayRef<NamedAttribute>":$attributes,
                    "RegionRange":$regions,
                    "SmallVectorImpl<Type>&":$inferedReturnTypes)
    >,
    StaticInterfaceMethod<
      /*desc=*/"Returns whether two array of types are compatible result types"
               " for an op.",
      /*retTy=*/"bool",
      /*methodName=*/"isCompatibleReturnTypes",
      /*args=*/(ins "ArrayRef<Type>":$lhs, "ArrayRef<Type>":$rhs),
      /*methodBody=*/[{
        return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
      }],
      /*defaultImplementation=*/[{
        /// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
      }]
    >,
  ];
}

#endif // MLIR_INFERTYPEOPINTERFACE
