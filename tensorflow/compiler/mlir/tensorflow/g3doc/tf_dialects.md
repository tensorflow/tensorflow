# TensorFlow MLIR Dialects

## Objective

[MLIR](https://medium.com/tensorflow/mlir-a-new-intermediate-representation-and-compiler-framework-beba999ed18d)
is the intermediate representation and compiler framework we are investing in to
build the compiler infrastructure for TensorFlow. The representation for
TensorFlow exposed in this document will be what future high-level
transformations will operate on.

We make use of two different dialects to model TensorFlow graphs in MLIR: first
the `tf_executor` dialect that represents the execution model of the TensorFlow
executor (e.g. control dependencies, deadness propagation) and the `tf` dialect
which represent the regular operations in a TensorFlow graph (the ones that
don’t have special contract with the executor).

One intent of this design is that TensorFlow 2.x features can choose to target
just the `tf` dialect, allowing us to phase out the `tf_executor` dialect in
subsequent TensorFlow releases. The combination of the two dialects allows to
represent arbitrary existing TensorFlow graphs.

The representation in this document does not address the specific needs of
accelerators or "custom backends" for TensorFlow. We plan to provide a generic
infrastructure for replacing the TF/XLA bridge with a more flexible and reusable
system across targets. A later design proposal will address these aspects. Also
this representation does not address shape inference, an independent design
exploration is being conducted separately at the moment.

## TensorFlow Dialect

The TensorFlow dialect in MLIR is an open dialect (it allows operations that
MLIR doesn't know about) that can contain any TensorFlow operation that does not
have a specific handling by the executor. These operations don’t operate on dead
values, don’t have control dependencies, and execute conceptually in program
order. The form used in this dialect aligns with the direction taken by
TensorFlow 2.0 with tf.function and autograph, as well as with the needs of
other frontends. This should ease the development of analyses and
transformations: optimizations operate on a simpler semantics and local graph
transformations can be validated in a local scope. Simple patterns like folding
`x-x` into a constant 0 do not need to update any control dependencies. It
should also be easily lowerable towards multiple accelerators and heterogeneous
systems in general.

Operations in this dialect usually operate on tensor and scalar types defined in
the standard dialect. The extra defined types are specific to TensorFlow: `QINT`
types like !tf.qint8 (etc), `QUINT` types like !tf.quint8, all of the `REF`
types like !tf.uint8ref, as well as !tf.string, !tf.resource, and !tf.variant
which correspond to the tensorflow types of the same name.

### Example:

Below is an example of a function operating on the TensorFlow dialect:

```mlir {.mlir}
/// This is a regular function, taking inputs by value and returning a new value.
/// The body is a regular CFG.
func some_function(%input : tensor<*xf32>) -> tensor<*xf32> {
  // TensorFlow operations are not variadic: this `tf.add` operation always
  // takes two inputs and returns a single output. This simplifies
  // pattern-matching, verification and rewriting.
  %added = tf.Add %input, %input : tensor<*xf32>
  // Operations have sequential execution semantics in a basic block, there are
  // no control dependencies.  The compiler can reorder operations according to
  // the as-if rule ( https://en.wikipedia.org/wiki/As-if_rule ).
  %three = constant splat<tensor<f32>, 3.0>
  %mul = tf.Mul %input, %three : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>

  // Only control flow v2 is supported in TF dialect.
  // The tf.If operation takes three functions that accept the same
  // arguments: the condition returns a bool and the two branches must return
  // the same type, which is also the return of the tf.If.
  %value = "tf.If”(%added, %mul)
             {cond: @cond_func, true_branch: @func_foo, false_branch: @func_bar}
                 : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  return %value : tensor<*xf32>
}
```

## TensorFlow Executor Dialect

The `tf_executor` dialect is intended to model the current TensorFlow executor
semantics and (when combined with the `tf` dialect) can represent arbitrary
TensorFlow 1.x and 2.x graphs. As such it follows the executor model, including
deadness propagation, concurrent semantics, and control dependencies. The
`tf_executor` dialect defines two dialect-specific types:

*   `!tf_executor.control` to represent control dependencies.
*   `!tf_executor.token` to represent the pair of operations modeling
    NextIteration operation.

The `tf_executor` dialect is closed (operations are all known to MLIR) as there
are only 8 TensorFlow ops with specific graph executor behavior and 4 additional
operations to represent islands of predictability.

This dialect models the TensorFlow executor semantics; as such, a large part of
the defined operations are mirroring the
[TensorFlow Control Flow Ops](https://www.tensorflow.org/api_docs/cc/group/control-flow-ops)
and
[implement Control Flow In TensorFlow](http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf).
Also, almost all the operations accept a variadic number of control tokens and
return an extra control token as output. Except for `tf_executor.Merge` and
`tf_executor.ControlTrigger`, operations are propagating deadness: if any of the
input (control and non-control) is dead, all the outputs (control and
non-control) are dead as well. For `tf_executor.Merge`, the output is dead only
when either an input control token is dead or all of the regular inputs are
dead. For `tf_executor.ControlTrigger`, a live control output is always produced
even when some control inputs are dead.

### `tf_executor.graph` Operation

The `tf_executor.graph` operation contains a region with a single block that
lists the operations in a TensorFlow graph. The operations are topologically
sorted in-order (no cycles are allowed in the SSA values). The execution model
for operations in this block follows the TensorFlow executor semantics:

1.  Operations that don’t have any transitive dependencies through the SSA
    def/use chains may be executed in parallel
    (`tf_executor.NextIteration.Source` is the exception).
2.  SSA values in this block can be implicitly dead. This means that every SSA
    value defined in a `tf_executor.graph` can be considered implicitly wrapped
    in a conceptual `dead_or<T>` structure, and includes a runtime flag
    indicating if the value is dead or present. Operations may have special case
    handling of dead values.
3.  Operations in this dialect return a value of type `!tf_executor.control` as
    last returned value (exceptions are `tf_executor.NextIteration.sink` and
    `tf_executor.fetch` which don’t return any value).

The `tf_executor.graph` op only allows specific `tf_executor` dialect operations
in its body: the `tf_executor.graph` verifier will reject any unknown operation.
In order to execute standard `tf` dialect operations (like `tf.Add`) they must
be wrapped in the `tf_executor.island` operation.

The `tf_executor.graph` operation does not accept any operands, inputs are
implicitly captured by the region, representing the feeds to the graph.

The region attached to `tf_executor.graph` is terminated by a
`tf_executor.fetch` operation. The non-control operands of the terminator
correspond to the result values (or fetches) of the `tf_executor.graph`
operation. The behavior is undefined if any of the operands of the
`tf_executor.fetch` is dead.

```mlir {.mlir}
%fetches = tf_executor.graph : tensor<*xf32> {
  // Operations in the current block execute when their inputs are ready,
  // possibly concurrently.
  // Only operations in the tf_executor dialect are expected here.
  // Ops can return multiple outputs and a control token for control
  // dependencies.
  // We don’t mention the control token in the return type here, it is implicit.
  %0, %ctl0 = tf_executor.opA %feed#0, %feed#1 : tensor<*xf32>
  %1, %ctl1 = tf_executor.opB : tensor<*xf32>
  %2, %ctl2 = tf_executor.opC %1, %ctl0 : tensor<*xf32>
  %3, %ctl3 = tf_executor.opD %2 : tensor<*xf32>
  tf_executor.fetch %3 : tensor<*xf32>
} // end of the “tf_executor.graph" operation/region
```

### ‘tf_executor.island’ Operation

The `tf_executor.graph` operation does not allow `tf` dialect operations to be
immediately nested underneath it. The `tf_executor.island` is introduced as a
wrapper for general computation (for example, all the `tf` dialect operations):
this results in a more consistent representation which makes analysis and
transformation simpler.

The `tf_executor.island` operation has a single region with a single block
attached (only functional control flow is allowed). The block is terminated by a
`tf_executor.yield` operation. The operands of the terminator correspond to the
result values of the `tf_executor.graph` operation. An extra result of type
`!_tf_executor.control` is always produced by every `tf_executor.island`.

Within an island, execution semantics follow standard sequential behavior
consistent with the direction of TensorFlow 2.0 and autograph, and desirable for
compiler analyses and transformations. Values in an island can’t be dead. Other
nested `tf_executor.graph` operations can be present in the region (or called
functions) to re-enable the TensorFlow executor behavior for a subsection of the
code. This is important for the following reasons:

*   Initially the functional control flow operations are calling functions
    involving nested graphs, if `tf_executor.graph` weren’t allowed in an
    island, these operations would need to have an equivalent in the
    `tf_executor` dialect.
*   Nesting also allows to form islands without involving inter-procedural
    analyzes: any function call may involve a callee with a graph.

The `tf_executor.island` region allows implicit capture. If any value captured
by a `tf_executor.island` is dead, the whole region does not execute and every
produced value is marked as dead as well.

An arbitrary number of `tf_executor.control` operands are accepted by a
`tf_executor.island` operation. If any operand is dead, the region is not
executed and dead values are immediately returned for every result.

```mlir {.mlir}
// The island is capturing implicitly %0 and %1. It is also taking a control
// dependency %ctl0 as input. It produces a tensor<*xf32> value matching the
// argument of the yield terminator, as well as an extra control token.
%2, %ctl2 = tf_executor.island (%ctl0)
                  : (tensor<*xf32>, !tf_executor<"control">) -> tensor<*xf32> {
  %added = tf.Add %1, %0 : tensor<*xf32>
  %mul = tf.Mul %added, %1 :tensor<*xf32>

  // The yield terminator operands are the result values of the island.
  tf_executor.yield %mul : tensor<*xf32>
}
```

The case where a single operation is wrapped inside an island can even be
compressed by inferring the terminator to be the returned value of the
operation. The example above if it only contained the addition with implicit
capture would be displayed as:

```mlir {.mlir}
%2, %ctl2 = tf_executor.island(%ctl0) wraps tf.Add %1, %0 : tensor<*xf32>
```

### `tf_executor.Switch` Operation

[`tf_executor.Switch`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/switch):
takes two inputs,`predicate`and`data`and returns two regular
outputs,`true_output`,`false_output`. The`data`input is copied
to`true_output`if`predicate`evaluates to true otherwise it is copied
to`false_output`. The other output is marked as dead. If one of the inputs or a
control token is dead, then all of the outputs are marked as dead as well.

### `tf_executor.SwitchN` Operation

[`tf_executor.SwitchN`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/control_flow_ops.cc#L49-L53):
takes two inputs,`data`and`index`and an integer attribute`num_outs`indicating
the number of outputs. The`data`input is copied to output indicated by
the`index` input. The other outputs are marked as dead. If one of the inputs or
a control token is dead, then all of the outputs are marked as dead as well.

### `tf_executor.Merge` Operation

[`tf_executor.Merge`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/merge):
takes a variadic number of inputs, and returns a single output. The output is
defined as a non-dead input (selected in a non-defined way if multiple inputs
are non-dead). If all inputs are dead, the output is also dead.

### NextIteration: `tf_executor.NextIteration.Source` and `tf_executor.NextIteration.Sink` Operation

The TensorFlow
[`NextIteration`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/next-iteration)
op is modeled using these two paired operations. Since _NextIteration_ is
intended for modeling the loop back-edges, breaking it in two different
operations allows to keep a structural
DAG.`tf_executor.NextIteration.Source`does not take any operand and produces two
results: one regular value corresponding to the TensorFlow graph, and a second
value of type`tf_executor.loop_token`. This token is consumed by the
paired`tf_executor.NextIteration.Sink`Operation alongside the value that is
passed through the back-edge. No value is returned
by`tf_executor.NextIteration.Sink`. The type of the result of the source must
match the type of the value operand of the sink.

`tf_executor.NextIteration.Source` is an exception in the executor model in the
sense that it executes after the paired `tf_executor.NextIteration.Sink` even
though there is no data dependency between them.

### `tf_executor.LoopCond` Operation

[`tf_executor.LoopCond`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/loop-cond):
forwards its boolean input to its output,
[it acts as`pivot` for marking the loop termination condition](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/control_flow_ops.h#L115-L118).

### `tf_executor.Enter` Operation

[`tf_executor.Enter`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/control_flow_ops.h##77-L79):
takes a single input and a`name` string attribute that identifies the execution
frame. It forwards its input to its output in the new execution frame.

### `tf_executor.Exit` Operation

[`tf_executor.Exit`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/control_flow_ops.h#L90-L92):
forwards its single input to its output, exiting the current execution frame.

### `tf_executor.ControlTrigger` Operation

[`tf_executor.ControlTrigger`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/control-trigger):
it is similar to
[a no-op](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/control_flow_ops.h#L23-L26)
that acts as a placeholder for control dependencies. It always produces a live
control output even when some control inputs are dead.

### `tf_executor.Send` Operation

[`tf_executor.Send`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/sendrecv_ops.h#L24):
matches TensorFlow semantics.

### `tf_executor.Recv` Operation

[`tf_executor.Recv`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/sendrecv_ops.h#L37):
matches TensorFlow semantics.

## Example

Below is an example of a loop decrementing an initial `%_count.init` integer
until it reaches 0 and returns the last value in the loop.

```mlir {.mlir}
// Loop `%count.init` times and return the last counter (always zero)
%fetches = tf_executor.graph {

  %loop.init, %ctl0 = tf_executor.Enter %count.init : i32

  %next_count, %tok = tf_executor.NextIteration.Source : i32

  %loop.body.init, %ctlMerge = tf_executor.Merge %loop.init, %next_count : i32

  %dec_count, %ctlAdd = tf_executor.island
    wraps tf.Add %loop.body.init, -1 : (i32, i32) -> i32

  %loop_cond, %ctlNE = tf_executor.island
    wraps tf.NotEqual %dec_count, 0 : (i32, i32) -> i1

  %true, %false, %ctlSwitch = tf_executor.Switch %loop_cond, %dec_count  : i32

  tf_executor.NextIteration.Sink[%tok] %false : i32

  %exit_count, %ctlExit = tf_executor.Exit %true : i32

  tf_executor.fetch %exit_count : i32
} // end of the "tf_executor.graph" operation/region
```
