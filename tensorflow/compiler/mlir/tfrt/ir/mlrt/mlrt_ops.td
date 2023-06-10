/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifdef MLRT_OPS
#else
#define MLRT_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.td"

class Mlrt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mlrt_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

def CondOp: Mlrt_Op<"cond", []> {
  let summary = "mlrt.cond op";

  let description = [{
    Execute $a_true_fn with $args if $cond is true; otherwise, %b_false_fn is
    executed.
  }];

  let arguments = (ins
    I1:$cond,
    Variadic<AnyType>:$args,
    SymbolRefAttr:$a_true_fn,
    SymbolRefAttr:$b_false_fn
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    $cond $a_true_fn $b_false_fn `(` $args `)` attr-dict `:` `(` type($args) `)` `->` `(` type($results) `)`
  }];
}

def AsyncOp: Mlrt_Op<"async", []> {
  let summary = "Launches a function asynchronously.";

  let description = [{
    Launch a function asynchronously.

    $args: a list of arguments to be passed.
    $callee: The function to be launched. Its return op must not have operands.

    $handle: This op returns a handle object that manages the context of the async execution.
  }];

  let arguments = (ins
    Variadic<AnyType>:$args,
    SymbolRefAttr:$callee
  );

  let results = (outs
    MlrtAsyncHandleType:$handle
  );

  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $handle)";
}

def AwaitHandleOp: Mlrt_Op<"await_handle", []> {
  let summary = "Awaits an async execution ";

  let description = [{
    Awaits an async execution.

    $handle: The handle returned by mlrt.async op.
  }];

  let arguments = (ins
    MlrtAsyncHandleType:$handle
  );

  let assemblyFormat = "operands attr-dict";
}

def AwaitAllHandleOp: Mlrt_Op<"await_all_handle", []> {
  let summary = "Awaits multiple async executions";

  let description = [{
    Awaits multiple async execution.

    $handles: A list of handles returned by mlrt.async ops.
  }];

  let arguments = (ins
    Variadic<MlrtAsyncHandleType>:$handles
  );

  let assemblyFormat = "operands attr-dict `:` type($handles)";
}

def AwaitControlOp: Mlrt_Op<"await_control", []> {
  let summary = "Await a signal from a future";

  let description = [{
    Await a signal, instead of a value, from a future.

    $future: A value of !mlrt.future type.
  }];

  let arguments = (ins
    MlrtFutureType:$future
  );

  let assemblyFormat = "operands attr-dict";
}

def AwaitAllControlOp: Mlrt_Op<"await_all_control", []> {
  let summary = "Awaits multiple signals";

  let description = [{
    Awaits multiple signals

    $futures: A list of !mlrt.futures
  }];

  let arguments = (ins
    Variadic<MlrtFutureType>:$futures
  );

  let assemblyFormat = "operands attr-dict `:` type($futures)";
}

def PromiseControlOp: Mlrt_Op<"promise_control", []> {
  let summary = "Set a control promise";

  let description = [{
    Set a control promise.

    $promise: A value of !mlrt.promise type.
  }];

  let arguments = (ins
    MlrtPromiseType:$promise
  );

  let assemblyFormat = "operands attr-dict";
}

def CaseOp : Mlrt_Op<"case"> {
  let summary = "An n-way switch statement which calls a single branch function.";
  let description = [{
    An n-way switch statement, implementing the following:
        ```
        switch (branch_index) {
          case 0:
            outputs = branches[0](inputs);
            break;
          case 1:
            outputs = branches[1](inputs);
            break;
          ...
          case [[nbranches-1]]:
          default:
            outputs = branches[nbranches-1](inputs);
            break;
        }
        ```
    Example: %res = mlrt.case %branch_idx [@branch0, @branch1] (%arg0, %arg1) : (i32, i32) -> (i32)
  }];

  let arguments = (ins I32:$branch_index,
                       ConfinedAttr<SymbolRefArrayAttr, [ArrayMinCount<1>]>:$branches,
                       Variadic<AnyType>:$branch_operands);

  let results = (outs Variadic<AnyType>:$branch_outputs);
  let assemblyFormat = [{
    $branch_index $branches `(` $branch_operands `)` attr-dict `:` `(` type($branch_operands) `)` `->` `(` type($branch_outputs) `)`
  }];
}

def AllocateControlFuturesOp: Mlrt_Op<"allocate_control_futures", [AttrSizedResultSegments]> {
  let summary = "Allocate futures and corresponding promises";

  let description = [{
    Allocate futures and corresponding promises.

    $num: The number of futures to be allocated.

    $promises: There are $num promises, and promises[i] shares the state with futures[i].
    $futures: There are $num futures, and futures[i] shares the state with promises[i].
  }];

  let arguments = (ins
    I32Attr:$num
  );

  let results = (outs
    Variadic<MlrtPromiseType>:$promises,
    Variadic<MlrtFutureType>:$futures
  );
}

def WhileOp : Mlrt_Op<"while", []> {
  let summary = "while operation";
  let description = [{
    cond: The boolean to control whether the first iteration should be
      executed.
    operands: The arguments to the first iteration.
    results: The results of the last iteration. The number and types of results
      excluding the last one are the same as the number and types of operands. The
      last element of results is an I1 value that is false.
    body_fn: The body function that takes the arguments and returns the results
      that includes an I1 value to indicate whether next iteration should be executed.

    The pseudo code:

    while(cond) {
      results = body_fn(operands)
      cond = results#1
    }
    return results

  }];

  let arguments = (ins I1:$cond,
                       Variadic<AnyType>:$arguments,
                       FlatSymbolRefAttr:$body_fn);

  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $cond $body_fn `(` $arguments `)` attr-dict `:` `(` type($arguments) `)` `->` `(` type(results) `)`
  }];
}
#endif
