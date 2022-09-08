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

//===- rt_ops.td ----------------------------------------------------------===//
//
// Operation definitions for RT dialect.
//
//===----------------------------------------------------------------------===//

#ifdef RT_OPS
#else
#define RT_OPS

include "tensorflow/compiler/xla/mlir/ir/runtime/rt_base.td"

//===----------------------------------------------------------------------===//
// Op definitions.
//===----------------------------------------------------------------------===//

class RT_Op<string mnemonic, list<Trait> traits = []> :
      Op<RuntimeDialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// SetOutputOp
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Rename to SetResult for consistent use of argument/result.

def SetOutputOp : RT_Op<"set_output"> {
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

def SetErrorOp : RT_Op<"set_error"> {
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

def IsOkOp : RT_Op<"is_ok"> {
  let summary = "returns true if status is ok";
  let description = "Checks if the runtime status is ok.";

  let arguments = (ins StatusType:$status);
  let results = (outs I1:$ok);

  let assemblyFormat = "$status attr-dict";
}

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

def CustomCallOp : RT_Op<"custom_call"> {
  let summary = "calls a custom function registered with the runtime";

  let description = [{
    This operation calls a custom function registered with the runtime. This
    mechanism allows to call any C++ function from the compiled XLA program, for
    example this can be used as an extension mechanism to register vendor
    specific operation implementations (e.g. call oneDNN convolution).

    Returns `!rt.status` value which can be checked to see if the custom call
    was successful.

    Example:

      ```mlir
      func @compute(%ctx: !rt.execution_context, %arg0: memref<?xf32>,
                                              %arg1: memref<?xf32>) {
        %status = rt.custom_call %ctx["one_dnn.some_operation"] (%arg0, %arg1)
          : (memref<?xf32>, memref<?xf32>) -> ()
        %0 = rt.is_ok %status
        cf.assert %0, "failed to call one_dnn custom call"
        return
      }
      ```

    To avoid collisions users should group custom calls into libraries and put
    them into namespaces (similar to MLIR dialects). In this example there is
    an assumption that all OneDnn related custom calls will be registered with
    a `one_dnn` prefix.

    Indirect custom calls are resolved at runtime using the custom calls registry,
    which incurs additional overheads because the custom call handler has to
    be looked up by name (expensive string map lookup).

    Direct custom calls are linked to the custom call handler when compiling the
    XLA executable, and the user must pass a runtime symbol map (see executable
    compilation options) that binds custom call callees to the function pointers
    implementing the custom call API:

      ```
      bool OneDnnSomeOpImpl(xla::runtime::ExecutionContext* ctx,
                            void** args, void** attrs);
      ```
  }];

  let arguments = (ins
    ExecutionContextType:$ctx,
    StrAttr:$callee,
    UnitAttr:$direct,
    Variadic<AnyType>:$operands
  );

  let results = (outs
    StatusType:$status,
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    (`direct` $direct^)? $ctx `[` $callee `]`  `(` $operands `)`
    attr-dict `:` functional-type($operands, $results)
  }];
}

#endif // RT_OPS
