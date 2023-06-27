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

def OrdinalAssignment : Pass<"xla-rt-ordinal-assignment", "mlir::ModuleOp"> {
  let summary = "Assigns ordinals to all exported functions";

  let description = [{
    Assigns unique ordinals to all exported functions. After assignment it's
    guaranteed that all `rt.export` operations have been assigned a valid
    ordinal.

    Example:

      ```mlir
      rt.export @compute.0
      rt.export @compute.1

      func @compute.0(...) -> memref<?xf32> { ... }
      func @compute.1(...) -> memref<?xf32> { ... }
      ```

    converted to:

      ```mlir
      rt.export @compute.0 ordinal 0
      rt.export @compute.1 ordinal 1

      func @compute.0(...) -> memref<?xf32> { ... }
      func @compute.1(...) -> memref<?xf32> { ... }
      ```
  }];

  let constructor = "xla::runtime::CreateOrdinalAssignmentPass()";
  let dependentDialects = ["xla::runtime::RuntimeDialect"];
}

def ExportFunctions : Pass<"xla-rt-export-functions", "mlir::ModuleOp"> {
  let summary = "Exports function(s) to Xla runtime";

  let description = [{
    Converts functions marked for export with an `rt.export` operation with
    specified ordinal attribute to Xla runtime ABI, and makes them invocable by
    the executable, i.e.:
     - first argument is an `!rt.execution_context`
     - all results returned via the `rt.set_result` operation

    Exported function can also return errors to the runtime using `rt.set_error`
    operation, and all asserts can be automatically converted to safe run-time
    errors with the `ConvertAsserts` pass defined below.

    See the `ir/runtime/rt_ops.td` to find how Xla executable returns results
    and errors using the runtime APIs.

    Example:

      ```mlir
      rt.export @compute ordinal 0

      func @compute(...) -> memref<?xf32> {
        %0 = "allocate_memref" : () -> memref<?xf32>
        return %0 : memref<?xf32>
      }
      ```

    converted to:

      ```mlir
      func @compute(%ctx: !rt.execution_context, ...)
        attributes { rt.exported = 0 : i32 } {
        %0 = "allocate_memref" : () -> memref<?xf32>
        rt.set_output %ctx, 0, %1 : memref<xf32>
        return
      }
      ```
  }];

  let constructor = "xla::runtime::CreateExportRuntimeFunctionsPass()";
  let dependentDialects = ["xla::runtime::RuntimeDialect"];
}

def ConvertCustomCalls : Pass<"xla-rt-convert-custom-calls", "ModuleOp"> {
  let summary = "Converts custom calls to explicit `rt.custom_call` operations";

  let description = [{
    Converts function calls to custom call declarations (function declarations
    annotated with `rt.custom_call` attribute) to explicit `rt.custom_call`
    operations. Function declarations annotated with `rt.dynamic` attribute
    are converted to dynamic custom calls that are resolved at run time by name.

    When converting function call to the custom call operation, custom call
    attributes will be a union of custom call function declaration attributes,
    and the call operation attributes. Function call attributes will override
    any attributes defined by the custom call function declaration.

    Example:

      ```mlir
      func @custom_call() -> memref<?xf32>
        attributes { rt.custom_call = "custom_call", attr = <value> }

      func @compute(%ctx: !rt.execution_context, %arg0: i1) -> memref<?xf32>
          attributes { rt.exported = 0 : i32 } {
        %0 = call @custom_call() { attr = <new_value> }: () -> memref<?xf32>
        return %0 : memref<?xf32>
      }
      ```

    converted to:

      ```mlir
      func @compute(%ctx: !rt.execution_context, %arg0: i1) -> memref<?xf32>
          attributes { rt.exported = 0 : i32 } {

        %status, %0 = rt.custom_call %ctx["custom_call"] () {
                        attr = <new value>
                      } : () -> memref<?xf32>

        %success = rt.is_ok %status : !rt.status
        assert %success, "Custom call to `custom_call` failed"

        return %0 : memref<?xf32>
      }
      ```
  }];

  let constructor = "xla::runtime::CreateConvertCustomCallsPass()";
  let dependentDialects = ["xla::runtime::RuntimeDialect"];
}

def ConvertAsserts : Pass<"xla-rt-convert-asserts", "ModuleOp"> {
  let summary = "Converts asserts in exported functions to run-time errors";

  let description = [{
    Converts failed assertions to calls to the `rt.set_error` operation to
    safely return errors to the caller. The default lowering of failed
    assertions calls `std::abort`, which causes abnormal program termination.

    This conversion only happens to exported runtime functions. These functions
    have access to an `!rt.execution_context`, through which all results and
    errors must be returned.

    Example:

      ```mlir
      func @compute(%ctx: !rt.execution_context, %arg0: i1)
          attributes { rt.exported = 0} {
        cf.assert %arg0, "Oops"
        return
      }
      ```

    converted to:

      ```mlir
      func @compute(%ctx: !rt.execution_context, %arg0: i1)
          attributes { rt.exported = 0 : i32 } {
        %0 = ... : i1
        cond_br %0, ^ok, ^err
      ^err:
        rt.set_error %ctx, "Oops"
        return
      ^ok:
        return
      }
      ```
  }];

  let constructor = "xla::runtime::CreateConvertAssertsPass()";
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
