//===- OpAsmInterface.td - Asm Interfaces for opse ---------*- tablegen -*-===//
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
// This file contains Interfaces for interacting with the AsmParser and
// AsmPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_OPASMINTERFACE
#define MLIR_OPASMINTERFACE

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

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
