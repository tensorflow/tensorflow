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

// This is the definition file for the TensorFlow Executor Dialect.
//

#ifndef TF_EXECUTOR_DIALECT
#define TF_EXECUTOR_DIALECT

include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

//===----------------------------------------------------------------------===//
// TensorFlow dialect definitions
//===----------------------------------------------------------------------===//

def TfExecutor_Dialect : Dialect {
  let name = "tf_executor";

  let description = [{
    The TensorFlow Executor dialect.

    This dialect models the TensorFlow executor semantics and can represent
    arbitrary TensorFlow graphs. As such it follows the existing execution model
    that includes deadness propagation, concurrent semantics, and control
    dependencies.

    Operations in this dialect return a value of type `!tf_executor.control` as
    last returned value (exceptions are `tf_executor.NextIteration.graph`,
    `tf_executor.NextIteration.sink` and `tf_executor.fetch` which don’t return any
    value).
}];

  let cppNamespace = "::mlir::tf_executor";
}

// Control type.
def TfeControlType : Type<CPred<"llvm::isa<ControlType>($_self)">, "control">,
                     BuildableType<"$_builder.getType<ControlType>()">;

// Token type.
def TfeTokenType : Type<CPred<"llvm::isa<TokenType>($_self)">, "token">,
                   BuildableType<"$_builder.getType<TokenType>()">;

// TODO(hinsu): Define and use TensorType instead of AnyType for data operands
// and results. For example, MergeOp output type.

//===----------------------------------------------------------------------===//
// TensorFlow Executor Type Constraint
//===----------------------------------------------------------------------===//

// Predicate to verify all control inputs appear after any non-control inputs.
def ControlOperandsAfterAllData :
    PredOpTrait<"all control inputs must appear after any non-control input",
                CPred<"succeeded(VerifyControlOperandsAfterAllData(&$_op))">>;

//===----------------------------------------------------------------------===//
// TensorFlow op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect, it'll forward the verifier,
// printer, and parser to free functions.
class TfExecutor_Op<string mnemonic, list<Trait> traits = []> :
    Op<TfExecutor_Dialect, mnemonic, traits> {

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def TfExecutor_GraphOp : TfExecutor_Op<"graph",
    [SingleBlockImplicitTerminator<"FetchOp">]> {
  let summary = [{The `tf_executor.graph` operation contains a region with a
    single block that lists the operations in a TensorFlow graph.}];


  let description = [{
    The operations are topologically sorted in-order (no cycles are allowed in
    the values). The execution model for operations in this block follows the
    TensorFlow executor semantics:
      1. Operations that don’t have any transitive dependencies through the
         def/use chains may be executed in parallel
         (`tf_executor.NextIteration.Source` is the exception).
      2. SSA values in this block can be implicitly dead. This means that every
         SSA value defined in a `tf_executor.graph` can be considered implicitly
         wrapped in a conceptual `dead_or<T>` structure, and includes a runtime
         flag indicating if the value is dead or present.
      3. Operations may have special case handling of dead values.

    The `tf_executor.graph` op only allows specific `tf_executor` dialect
    operations in its body: the `tf_executor.graph` verifier will reject any
    unknown operation. In order to execute standard `tf` dialect operations
    (like `tf.Add`) they must be wrapped in the `tf_executor.island` operation.

    The `tf_executor.graph` operation does not accept any operands, inputs are
    implicitly captured by the region, representing the feeds to the graph.

    The region attached to `tf_executor.graph` is terminated by a
    `tf_executor.fetch` operation. The operands of the terminator correspond to
    the result values (or fetches) of the `tf_executor.graph` operation. The
    behavior is undefined if any of the operands of the `tf_executor.fetch` is
    dead.
  }];

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    FetchOp GetFetch();
  }];

  let hasCanonicalizer = 1;
}

def TfExecutor_FetchOp : TfExecutor_Op<"fetch",
    [Terminator, ControlOperandsAfterAllData, HasParent<"GraphOp">]> {
  let summary = [{
    The `tf_executor.fetch` operation terminates the graph and returns values;
  }];

  let description = [{
    The non-control operands of the fetch operation are returned outside of the
    graph and must match the return type of the graph.
  }];

  let arguments = (ins
    Variadic<AnyType>:$fetches
  );

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, {});
    }]>
   ];

  let assemblyFormat = "($fetches^ `:` type($fetches))? attr-dict";

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 0;
}

def TfExecutor_IslandOp : TfExecutor_Op<"island",
    [HasParent<"GraphOp">, SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = [{
    The `tf_executor.island` operation is a wrapper for operations in other
    dialects to be nested in a `tf_executor.graph`.
  }];

  let description = [{
    The `tf_executor.graph` operation does not allow `tf` dialect operations to
    be immediately nested underneath it. The `tf_executor.island` is introduced
    as a wrapper for `tf` dialect operations: this results in a more consistent
    representation which makes analysis and transformation simpler.
    The `tf_executor.island` operation has a single region with a single block
    attached (only functional control flow is allowed). The block is terminated
    by a `tf_executor.yield` operation. The operands of the terminator
    correspond to the result values of the `tf_executor.island` operation. An
    extra result of type `!tf_executor.control` is always produced by every
    `tf_executor.island`.
    Within an island, execution semantics follow standard sequential behavior as
    expected by TF2 and by compiler analyses and transformations, and values
    can’t be dead. Other nested `tf_executor.graph` operations can be present in
    the region to re-enable the TensorFlow executor for a subsection of the
    code.
     - Initially the functional control flow operations are calling functions
       involving graphs, if `tf_executor.graph` weren’t allowed in an island,
       these operations would need to have an equivalent in the `tf_executor`
       dialect to be modelled in a graph.
     - Nesting also allows forming islands without involving inter-procedural
       analyses: any function call may involve a callee with a graph.
    The `tf_executor.island` region allows implicit capture. If any value
    captured by a `tf_executor.island` is dead, the whole region does not
    execute and every produced value is marked as dead as well.
    An arbitrary number of `tf_executor.control` operands are accepted by a
    `tf_executor.island` operation.
    If any operand or implicitly captured value are dead, the region is not
    executed and dead values are immediately returned for every result.
  }];

  let arguments = (ins
    Variadic<TfeControlType>:$controlInputs
  );

  let results = (outs
    Variadic<AnyType>:$outputs,
    TfeControlType:$control
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    YieldOp GetYield();
    bool WrapsSingleOp();
  }];

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

def TfExecutor_YieldOp : TfExecutor_Op<"yield",
    [Terminator, ControlOperandsAfterAllData, HasParent<"IslandOp">]> {
  let summary = [{
    The `tf_executor.yield` operation terminates and returns values for the
    `tf_executor.island` operation.
  }];

  let arguments = (ins
    Variadic<AnyType>:$fetches
  );

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, {});
    }]>
   ];

  let assemblyFormat = "($fetches^ `:` type($fetches))? attr-dict";

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 0;
}

def TfExecutor_SwitchOp : TfExecutor_Op<"Switch",
    [ControlOperandsAfterAllData, HasParent<"GraphOp">,
     PredOpTrait<"data operand must be broadcastable to true result",
                 TF_OpIsBroadcastableToRes<0, 0>>,
     PredOpTrait<"data operand must be broadcastable to false result",
                 TF_OpIsBroadcastableToRes<0, 1>>]>{
  let summary = [{
    The "tf_executor.Switch" operation takes a data operand and a boolean
    predicate condition, and returns two values matching the type of the data
    predicate.
  }];

  let description = [{
    More details can be found in Tensorflow Control Flow white paper:
    https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf

    This is defined in TensorFlow as:

    REGISTER_OP("Switch")
       .Input("data: T")
       .Input("pred: bool")
       .Output("output_false: T")
       .Output("output_true: T")

    For example:
      %2 = tf_executor.Switch %0, %1 : tensor<*xf32>

    Note: Additional result corresponds to the control output.
  }];

  let arguments = (ins
    AnyType:$data,
    TensorOf<[I1]>:$predicate,
    // Optional extra control inputs.
    Variadic<TfeControlType>:$controlInputs
  );

  let results = (outs
    AnyType: $falseOutput,
    AnyType: $trueOutput,
    TfeControlType: $control
  );

   let hasVerifier = 0;
}

def TfExecutor_SwitchNOp : TfExecutor_Op<"_SwitchN",
    [ControlOperandsAfterAllData, HasParent<"GraphOp">]> {
  let summary = [{
    The "tf_executor._SwitchN" operation takes two inputs, `data` and `index`
    and an integer attribute `num_outs` indicating the number of outputs. The
    `data` input is copied to output indicated by the `index` input. The other
    outputs are marked as dead. If one of the inputs or a control token is
    dead, then all of the outputs are marked as dead as well.
  }];

  let description = [{
    This is defined in TensorFlow as:

    REGISTER_OP("_SwitchN")
        .Input("data: T")
        .Input("output_index: int32")
        .Output("outputs: num_outs * T")
        .Attr("num_outs: int >= 1")
        .Attr("T: type")
        .SetShapeFn(SwitchNShape);

    For example:
      %2:6 = tf_executor.SwitchN %0, %1 of 5 : tensor<??xf32>

    Note: One additional result corresponds to the control output.
  }];

  let arguments = (ins
    AnyType:$data,
    TensorOf<[I32]>:$index,
    // Optional extra control inputs.
    Variadic<TfeControlType>:$controlInputs,
    I64Attr:$num_outs
  );

  let results = (outs
    Variadic<AnyType>:$outputs,
    TfeControlType: $control
  );
}

def TfExecutor_MergeOp : TfExecutor_Op<"Merge",
    [ControlOperandsAfterAllData, HasParent<"GraphOp">]> {
  let summary = [{
    The "tf_executor.Merge" operation takes a list of input operands and returns
    a value of the operand type along with the index of the first match encountered.
  }];

  let description = [{
    More details can be found in Tensorflow Control Flow white paper:
    https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf

    This is defined in TensorFlow as:

    REGISTER_OP("Merge")
       .Input("inputs: N * T")
       .Output("output: T")
       .Output("value_index: int32")

    For example:
      %2 = tf_executor.Merge %0, %1, %2, %3 : tensor<*xf32>

    Note: Additional result corresponds to the control output.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs_and_control
  );

  let results = (outs
    AnyTensor:$output,
    TensorOf<[I32]>:$value_index,
    TfeControlType:$control
  );
}

def TfExecutor_EnterOp : TfExecutor_Op<"Enter",
    [ControlOperandsAfterAllData, HasParent<"GraphOp">,
     PredOpTrait<"data operand must be broadcastable to result",
                 TF_OpIsBroadcastableToRes<0, 0>>]>{
  let summary = [{
    The "tf_executor.Enter" operation forwards its input to Tensorflow while
    loop.
  }];

  let description = [{
    More details can be found in Tensorflow Control Flow white paper:
    https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf

    Each tensor needs its own tf_executor.Enter to be made available inside a
    while loop.

    This is defined in Tensorflow as:

    REGISTER_OP("Enter")
       .Input("data: T")
       .Output("output: T")
       .Attr("T: type")
       .Attr("frame_name: string")
       .Attr("is_constant: bool = false")
       .Attr("parallel_iterations: int = 10")

    For example:
       %res:2 = tf_executor.Enter %arg0 frame "some/frame" parallel_iterations 42 constant : tensor<*xf32>

    Note: Additional result corresponds to the control output.
  }];

  let arguments = (ins
    AnyType:$data,
    StrAttr:$frame_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_constant,
    DefaultValuedOptionalAttr<I64Attr, "10">:$parallel_iterations,
    // Optional extra control inputs.
    Variadic<TfeControlType>:$controlInputs
  );


  let results = (outs
    AnyType:$output,
    TfeControlType:$control
  );

  let hasVerifier = 0;
}

def TfExecutor_NextIterationSourceOp : TfExecutor_Op<"NextIteration.Source",
    [HasParent<"GraphOp">]> {
  let summary = [{
    The "tf_executor.NextIteration.Source" is paired with a
    "tf_executor.NextIteration.sink" to represent NextIteration op in
    Tensorflow.
  }];

  let description = [{
    Tensorflow NextIteration operation forwards its input to the next iteration
    of a while loop. Each loop variable needs its own NextIteration op.

    More details can be found in Tensorflow Control Flow white paper:
    https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf

    In the TF executor dialect, the NextIteration op is broken into
    tf_executor.NextIteration.sink and tf_executor.NextIteration.source because
    NextIteration is a back-edge in Tensorflow graph, which would form a data
    flow cycle if expressed naively in a basic block.
    tf_executor.NextIteration.source takes no input but returns results while
    tf_executor.NextIteration.sink takes input but doesn't return anything. When
    optimizing these ops, they are paired by token and considered as a single
    op.

    This is defined in Tensorflow as:

    REGISTER_OP("NextIteration")
       .Input("data: T")
       .Output("output: T")
       .Attr("T: type")

    For example:
      %value, %token, %ctl = tf_executor.NextIteration.Source : tensor<*xi32>
      tf_executor.NextIteration.sink [%token] (%value) : tensor<*xi32>

    Note: Additional result corresponds to the control output.
  }];

  let results = (outs
    AnyType:$output,
    // The NextIteration.Source operation returns an extra token consumed by the sink.
    TfeTokenType:$token,
    TfeControlType:$control
  );

  let builders = [
    OpBuilder<(ins "Type":$result_type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
    [{
      Type token_type = TokenType::get($_builder.getContext());
      Type control_type = ControlType::get($_builder.getContext());
      $_state.types = { result_type, token_type, control_type };
      $_state.attributes.append(attributes.begin(), attributes.end());
    }]>
   ];

  let extraClassDeclaration = [{
    NextIterationSinkOp GetSink() {
      return cast<NextIterationSinkOp>(*getToken().user_begin());
    }
  }];

  let assemblyFormat = "`:` type($output) attr-dict";
  let hasCustomAssemblyFormat = 0;
}


def TfExecutor_NextIterationSinkOp : TfExecutor_Op<"NextIteration.Sink",
    [HasParent<"GraphOp">]> {
  let summary = [{
    The "tf_executor.NextIteration.Sink" is paired with a
    "tf_executor.NextIteration.source" to represent NextIteration op in
    Tensorflow.
  }];

  let description = [{
    Tensorflow NextIteration operation forwards its input to the next iteration
    of a while loop. Each loop variable needs its own NextIteration op.

    More details can be found in Tensorflow Control Flow white paper:
    https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf

    In the TF executor dialect, the NextIteration op is broken into
    tf_executor.NextIteration.sink and tf_executor.NextIteration.source because
    NextIteration is a back-edge in Tensorflow graph, which would form a data
    flow cycle if expressed naively in a basic block.
    tf_executor.NextIteration.source takes no input but returns results while
    tf_executor.NextIteration.sink takes input but doesn't return anything. When
    optimizing these ops, they are paired by token and considered as a single
    op.

    This is defined in Tensorflow as:

    REGISTER_OP("NextIteration")
       .Input("data: T")
       .Output("output: T")
       .Attr("T: type")

    For example:
      %value, %token, %ctl = tf_executor.NextIteration.Source : tensor<*xi32>
      tf_executor.NextIteration.sink [%token] (%value) : tensor<*xi32>

    Note: Additional result corresponds to the control output.
  }];

  let arguments = (ins
    // The NextIteration.Sink operation consumes the token produced by the source.
    TfeTokenType:$token,
    AnyType:$input,
    // Optional extra control inputs.
    Variadic<TfeControlType>:$controlInputs
  );

  let builders = [
    OpBuilder<(ins "Value":$token, "ArrayRef<Value>":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
    [{
      assert(operands.size() >= 1 && "tf_executor.NextIteration.Sink builder "
             "expects at least one operand");
      $_state.operands.push_back(token);
      $_state.operands.insert($_state.operands.end(), operands.begin(),
                              operands.end());
      $_state.attributes.append(attributes.begin(), attributes.end());
    }]>
   ];

  let extraClassDeclaration = [{
    NextIterationSourceOp GetSource();
  }];

  let assemblyFormat = " `[` $token `]` $input (`,` $controlInputs^)? `:` type($input) attr-dict";

  let hasCustomAssemblyFormat = 0;
}

def TfExecutor_ExitOp : TfExecutor_Op<"Exit",
    [HasParent<"GraphOp">,
     PredOpTrait<"data operand must be broadcastable to result",
                 TF_OpIsBroadcastableToRes<0, 0>>]>{

  let summary = [{
    The "tf_executor.Exit" operation forwards a value from an while loop to its
    consumer outside of loop. Each returned tensor needs its own
    tf_executor.Exit.
  }];

  let description = [{
    More details can be found in Tensorflow Control Flow white paper:
    https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf

    This is defined in Tensorflow as:

    REGISTER_OP("Exit")
       .Input("data: T")
       .Output("output: T")
       .Attr("T: type")

    For example:
     %1:2 = tf_executor.Exit %0#0 : tensor<*xi32> {T: "tfdtype$DT_INT32"}

    Note: Additional result corresponds to the control output.
  }];

  let arguments = (ins
    AnyType:$data,
    Variadic<TfeControlType>:$controlInputs
  );

  let results = (outs
    AnyType:$output,
    TfeControlType:$control
  );

  let hasVerifier = 0;
}

def TfExecutor_ControlTriggerOp : TfExecutor_Op<"ControlTrigger",
    [HasParent<"GraphOp">]> {
  let summary = [{
    The `tf_executor.ControlTrigger` operation is similar to a no-op except that
    it always produces a valid output even when inputs are dead.
  }];

  let description = [{
    Its primary use so far is in the scheduling of recvs, where we add
    ControlTrigger nodes and use them to trigger recvs. We allow ControlTrigger
    nodes to be enabled by dead nodes.
  }];

  let arguments = (ins
    Variadic<TfeControlType>:$controlInputs
  );
  let results = (outs
    TfeControlType:$control
  );

  let hasVerifier = 0;

  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "ArrayRef<Value>":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
    [{
      assert(operands.size() >= 1 && "tf_executor.ControlTrigger builder "
             "expects at least one operand");
      $_state.operands.insert($_state.operands.end(), operands.begin(),
                              operands.end());
      Type control_type = ControlType::get($_builder.getContext());
      $_state.types = {control_type};
      $_state.attributes.append(attributes.begin(), attributes.end());
    }]>
   ];

  let assemblyFormat = "$controlInputs attr-dict";
  let hasCustomAssemblyFormat = 0;
}

def TfExecutor_LoopCondOp : TfExecutor_Op<"LoopCond",
    [HasParent<"GraphOp">]> {
  let summary = [{
    The "tf_executor.LoopCond" operation forwards a boolean value as loop
    condition of Tensorflow while loops.
  }];

  let description = [{
    More details can be found in Tensorflow Control Flow white paper:
    https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf

    This is defined in Tensorflow as:

    REGISTER_OP("LoopCond")
       .Input("input: bool")
       .Output("output: bool")

    For example:
      %5:2 = tf_executor.LoopCond %4#0 {name: "while/LoopCond"}

    Note: Additional result corresponds to the control output.
  }];

  let arguments = (ins
    TensorOf<[I1]>:$input,
    Variadic<TfeControlType>:$controlInputs
  );
  let results = (outs
    TensorOf<[I1]>:$output,
    TfeControlType:$control
  );

  let hasVerifier = 0;
}

#endif // TF_EXECUTOR_DIALECT
