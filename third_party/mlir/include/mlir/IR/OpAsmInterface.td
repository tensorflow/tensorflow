//===- OpAsmInterface.td - Asm Interfaces for opse ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Interfaces for interacting with the AsmParser and
// AsmPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_OPASMINTERFACE
#define MLIR_OPASMINTERFACE

include "mlir/IR/OpBase.td"

/// Interface for hooking into the OpAsmPrinter and OpAsmParser.
def OpAsmOpInterface : OpInterface<"OpAsmOpInterface"> {
  let description = [{
    This interface provides hooks to interact with the AsmPrinter and AsmParser
    classes.
  }];

  let methods = [
    InterfaceMethod<[{
        Get a special name to use when printing the results of this operation.
        The given callback is invoked with a specific result value that starts a
        result "pack", and the name to give this result pack. To signal that a
        result pack should use the default naming scheme, a None can be passed
        in instead of the name.

        For example, if you have an operation that has four results and you want
        to split these into three distinct groups you could do the following:

        ```c++
          setNameFn(getResult(0), "first_result");
          setNameFn(getResult(1), "middle_results");
          setNameFn(getResult(3), ""); // use the default numbering.
        ```

        This would print the operation as follows:

        ```mlir
          %first_result, %middle_results:2, %0 = "my.op" ...
        ```
      }],
      "void", "getAsmResultNames", (ins "OpAsmSetValueNameFn":$setNameFn)
    >,
  ];
}

#endif // MLIR_OPASMINTERFACE
