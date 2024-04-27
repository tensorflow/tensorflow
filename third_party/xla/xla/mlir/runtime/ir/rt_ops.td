/* Copyright 2022 The OpenXLA Authors.

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

//===- rt_ops.td ----------------------------------------------------------===//
//
// Operation definitions for RT dialect.
//
//===----------------------------------------------------------------------===//

#ifdef RT_OPS
#else
#define RT_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

include "xla/mlir/runtime/ir/rt_dialect.td"
include "xla/mlir/runtime/ir/rt_interfaces.td"

//===----------------------------------------------------------------------===//
// Op definitions.
//===----------------------------------------------------------------------===//

class RT_Op<string mnemonic, list<Trait> traits = []> :
      Op<RuntimeDialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// ExportOp
//===----------------------------------------------------------------------===//

def RT_ExportOp : RT_Op<"export", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    HasParent<"mlir::ModuleOp">]> {
  let summary = "exports a function from the module";

  let description = [{
    Specifies an exported function that can be called externally by the XLA
    runtime and becomes visible as an XLA executable function.

    Optional ordinal attribute specifies exported function ordinal in the
    executable. If ordinal is not defined, it will be assigned automatically by
    the ordinal assignment pass.

    Each ordinal must be unique and ordinals are contiguous starting from zero.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$function_ref,
    OptionalAttr<RT_Ordinal>:$ordinal
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::FunctionOpInterface":$function_ref)>,
    OpBuilder<(ins "mlir::FunctionOpInterface":$function_ref, 
                   "unsigned":$ordinal)>,
  ];

  let extraClassDeclaration = [{
    std::optional<unsigned> ordinal();
    mlir::FunctionOpInterface exported(mlir::SymbolTable& sym_table);
  }];

  let assemblyFormat = "$function_ref (`ordinal` $ordinal^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// SetOutputOp
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Rename to SetResult for consistent use of argument/result.

def RT_SetOutputOp : RT_Op<"set_output"> {
  let summary = "set the result to a given value";

  let description = [{
    This operation sets the executable result at the given index to the given
    value. In XLA executables we do not return the results using the
    conventional return statement, but use the runtime context API to pass
    values back to the runtime.

    We want to support early returns from the function in case of an error. In
    C++ we would use `StatusOr<Result>`, however we do not want to define the
    ABI for this type, and instead we rely on the runtime APIs (see `set_error`
    operation defined below) to return either a result or an error.

    The result becomes available to the caller only when the executable returns
    the control flow, and not when the `set_result` is called. Executable can
    set the result for the same `index` multiple times, and the last one will be
    returned to the caller.

    Example:

      ```mlir
      func @compute(%ctx: !rt.execution_context) {
        %out0 = ... : memref<?xf32>
        %out1 = ... : memref<?x?xf32>
        rt.set_output %ctx, 0, %out0 : memref<?xf32>
        rt.set_output %ctx, 1, %out1 : memref<?x?xf32>
      }
      ```

    is an equivalent of a regular function:

      ```mlir
      func @compute() -> (memref<?xf32>, memref<?x?xf32) {
        %out0 = ... : memref<?xf32>
        %out1 = ... : memref<?x?xf32>
        return %out0, %out1 : memref<?xf32>, memref<?x?xf32>
      }
      ```
  }];

  let arguments = (ins
    ExecutionContextType:$ctx,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$index,
    AnyType:$value
  );

  let assemblyFormat = [{
    $ctx `,` $index `,` $value `:` type($value) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// SetErrorOp
//===----------------------------------------------------------------------===//

def RT_SetErrorOp : RT_Op<"set_error"> {
  let summary = "set all executable results to the error state";

  let description = [{
    This operation sets all XLA executable results to the error state. An XLA
    executable can call set_error only once, and must not set any of the results
    in this case (before or after calling `set_error`). The provided error
    message may be used by a runtime to propagate the error to the user.

    Example:

      ```mlir
      func @compute(%ctx: !rt.execution_context) {
        %precondition = arith.cmpi ...
        cond_br %precondition, ^ok, ^err

      ^ok:
        %result = "compute_result"(): () -> memref<?xf32>
        rt.set_output %ctx, 0, %result : memref<?xf32>
        return

      ^err:
        rt.set_error %ctx, "Failed precondition"
        return
      }
      ```
  }];

  let arguments = (ins
    ExecutionContextType:$ctx,
    StrAttr:$error);

  let assemblyFormat = "$ctx `,` $error attr-dict";
}

//===----------------------------------------------------------------------===//
// IsOkOp
//===----------------------------------------------------------------------===//

def RT_IsOkOp : RT_Op<"is_ok"> {
  let summary = "returns true if status is ok";
  let description = "Checks if the runtime status is ok.";

  let arguments = (ins StatusType:$status);
  let results = (outs I1:$ok);

  let assemblyFormat = "$status attr-dict";
}

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

def RT_CallOp : RT_Op<"call"> {
  let summary = "calls a custom function registered with the runtime";

  let description = [{
    This operation calls a custom function registered with the runtime. This
    mechanism allows to call any C++ function from the compiled XLA program. For
    example, the XLA Gpu runtime is implemented as a set of function calls into
    the Gpu abstraction layer.

    Returns `!rt.status` value which can be checked to see if the custom call
    was successful.

    Example:

      ```mlir
      func @compute(%ctx: !rt.execution_context,
                    %arg0: memref<?xf32>, %arg1: memref<?xf32>) {
        %status = rt.call %ctx["xla.gpu.kernel_launch"] (%arg0, %arg1)
          { kernel = "fusion.0" } : (memref<?xf32>, memref<?xf32>) -> ()
        %0 = rt.is_ok %status
        cf.assert %0, "failed to call xla.gpu.kernel_launch custom call"
        return
      }
      ```

    To avoid collisions users should group custom calls into libraries and put
    them into namespaces (similar to MLIR dialects). In this example there is
    an assumption that all XLA Gpu related custom calls will be registered with
    an `xla.gpu` prefix.

    Dynamic custom calls are resolved at runtime using the custom calls
    registry, which incurs additional overheads because the custom call handler
    has to be looked up by name (expensive string map lookup).

    Direct custom calls are linked to the custom call handler when compiling the
    XLA executable, and the user must pass a runtime symbol map (see executable
    compilation options) that binds custom call callees to the function pointers
    implementing the custom call API:

      ```
      bool CustomCallImpl(xla::runtime::ExecutionContext* ctx,
                          void** args, void** attrs, void** rets);
      ```
  }];

  let arguments = (ins
    ExecutionContextType:$ctx,
    StrAttr:$callee,
    UnitAttr:$dynamic,
    Variadic<AnyType>:$arguments
  );

  let results = (outs
    StatusType:$status,
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    (`dynamic` $dynamic^)? $ctx `[` $callee `]`  `(` $arguments `)`
    attr-dict `:` functional-type($arguments, $results)
  }];
}

//===----------------------------------------------------------------------===//
// TraceOp
//===----------------------------------------------------------------------===//

def RT_TraceOp : RT_Op<"trace",
    [AutomaticAllocationScope, OpAsmOpInterface,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Trace operation";

  let description = [{
    Traces execution of the attached region using provided annotation.

    Example: trace gemm with a corresponding HLO operation annotation

    ```mlir
    %0 = rt.trace %ctx, #rt.hlo<op=gemm, module=foo> -> !tensor<?x?xf32> {
      %1 = call @local_xla.gemm(...) : (...) -> tensor<?x?xf32>
      yield %1 : tensor<?x?xf32>
    }
    ```
  }];

  let arguments = (ins
    ExecutionContextType:$ctx,
    RT_TraceAnnotationAttrInterface:$annotation
  );

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    qualified($annotation) `,` $ctx (`->` type($results)^)? $body attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // OpAsmOpInterface: Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "rt"; }
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins
      "mlir::TypeRange":$results,
      "mlir::Value":$exec_ctx,
      "TraceAnnotationAttrInterface":$annotation,
      CArg<"llvm::function_ref<void(mlir::OpBuilder&, mlir::Location)>",
           "nullptr">:$bodyBuilder)>,
  ];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def RT_YieldOp : RT_Op<"yield",
    [HasParent<"TraceOp">, Pure, Terminator,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "terminator for rt.trace operation";

  let description = [{
    The `rt.yield` is a special terminator operation for the block inside
    `rt.trace` operation that returns values defined inside the body block to
    the users of `rt.trace` operation results.
  }];

  let arguments = (ins Variadic<AnyType>:$arguments);

  let assemblyFormat = "($arguments^ `:` type($arguments))? attr-dict";

  // Default builder needed for ensureTerminator
  let builders = [OpBuilder<(ins), "build($_builder, $_state, {});">];
}

//===----------------------------------------------------------------------===//
// UnsignedCast
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): This is not a reasonable long term solution for the problem.
// We should either get rid of unsigned types before we lower to gpu dialect, or
// at least put this operation to some other dialect that facilitates lowering
// to LLVM, and do not keep it in `rt`.

def RT_UnsignedCastOp : RT_Op<"unsigned_cast"> {
  let summary = [{
    Converts any integer value to an unsigned integer value.
  }];

  let description = [{
    This is a work around the fact that `arith` and `LLVM` dialects do not
    support unsigned integer constants, and we need to generate a valid IR for
    `gpu.memset` operation with unsigned memref argument.

    This operation is erased by the `rt-to-llvm` pass.
  }];

  let arguments = (ins SignlessIntegerLike:$value );
  let results = (outs AnyType:$result );
}

#endif // RT_OPS
