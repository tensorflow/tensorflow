/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_RUNTIME_PASSES
#define XLA_RUNTIME_PASSES

include "mlir/Pass/PassBase.td"

//===-----------------------------------------------------------------------===/
// Transformations targeting `rt` dialect.
//===-----------------------------------------------------------------------===/

def ExportFunctions : Pass<"xla-rt-export-functions", "mlir::ModuleOp"> {
  let summary = "Exports function(s) to Xla runtime";

  let description = [{
    Converts functions marked for export with a `rt.export` operation to Xla
    runtime ABI, and makes them invocable by the executable, i.e.:
     - first argument is an `!rt.execution_context`
     - all results returned via the `rt.set_result` operation
     - failed asserts set the results error via the `rt.set_error` operation
     - function calls marked with `rt.custom_call` attribute (on the callee)
       converted to the `rt.custom_call` operations.
     - function declarations annotated with `rt.dynamic` attribute converted to
       dynamic custom calls that are resolved at run time by name

    See the `ir/runtime/rt_ops.td` to find how Xla executable returns results
    and errors using the runtime APIs.

    When converting function call to the custom call operation, custom call
    attributes will be a union of custom call function declaration attributes,
    and the call operation attributes. Function call attributes will override
    any attributes defined by the custom call function declaration.

    Example:

      ```mlir
      func @custom_call() -> memref<?xf32>
        attributes { rt.custom_call = "custom_call", attr = <value> }

      rt.export @compute

      func @compute(...) -> memref<?xf32> {
        %0 = ... : i1
        assert %0, "Oops"
        %1 = call @custom_call() { attr = <new_value> }: () -> memref<?xf32>
        return %1
      }
      ```

    converted to:

      ```mlir
      func @compute(%ctx: !rt.execution_context, ...)
          attributes { rt.exported } {
        %0 = ... : i1
        cond_br %0, ^ok0, ^err0
      ^ok0:
        %status, %1 = rt.custom_call %ctx, "custom_call"()
                      { attr = <new value> } : () -> memref<?xf32>
        %success = rt.is_ok %status : !rt.status
        cond_br %success, ^ok1, ^err1
      ^ok1:
        rt.set_output %ctx, 0, %1 : memref<xf32>
        return
      ^err0:
        rt.set_error %ctx, "Oops"
        return;
      ^err1:
        rt.set_error %ctx, "Custom call failed"
        return;
      }
      ```
  }];

  let constructor = "xla::runtime::CreateExportRuntimeFunctionsPass()";
  let dependentDialects = ["xla::runtime::RuntimeDialect"];
}

//===-----------------------------------------------------------------------===/
// Conversions targeting `rt` dialect.
//===-----------------------------------------------------------------------===/

def ConvertRuntimeToLLVMPass : Pass<"xla-rt-to-llvm", "ModuleOp"> {
  let summary = "Convert RT dialect to LLVM dialect";
  let constructor = "::xla::runtime::CreateConvertRuntimeToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif  // XLA_RUNTIME_PASSES
