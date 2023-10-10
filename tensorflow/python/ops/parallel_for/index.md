# Vectorized map

*See also https://en.wikipedia.org/wiki/Automatic_vectorization*

TensorFlow provides in-graph looping constructs like `tf.while_loop` which are
similar to loops in other languages: they repeatedly run the loop body, not
keeping memory contiguous or taking advantage of hardware SIMD. When loop
iterations are independent, it is much more efficient to batch tensors together:
a matrix-matrix multiply instead of a loop over vector-matrix multiplies.

`tf.vectorized_map` provides a `tf.map_fn`-like API with the efficiency of
manual batching: while the API matches APIs which are implemented with a
`tf.while_loop` like `tf.map_fn`, `tf.vectorized_map` is implemented by using
batch dimensions of ops. This `tf.map_fn` style is often a more convenient way
to author models, as opposed to juggling a batch dimension explicitly:

```python
def f(args):
  embeddings, index = args
  # embeddings [vocab_size, embedding_dim]
  # index []
  # desired result: [embedding_dim]
  return tf.gather(params=embeddings, indices=index)

@tf.function
def f_auto_vectorized(embeddings, indices):
  # embeddings [num_heads, vocab_size, embedding_dim]
  # indices [num_heads]
  # desired result: [num_heads, embedding_dim]
  return tf.vectorized_map(f, [embeddings, indices])

concrete_vectorized = f_auto_vectorized.get_concrete_function(
  tf.TensorSpec(shape=[None, 100, 16], dtype=tf.float32),
  tf.TensorSpec(shape=[None], dtype=tf.int32))
print(concrete_vectorized.graph.as_graph_def())
```

The vectorized graph contains many ops, but no loops. Instead,
`tf.vectorized_map` looks at the GatherV2 op and its attributes, and generates
the equivalent of `tf.gather(..., batch_dims=1)` without requiring the user to
know how to tell `tf.gather` and every other op they use which dimensions are
batch dimensions.

```python
gdef = concrete_vectorized.graph.as_graph_def()
print([n for n in gdef.node if n.op == "GatherV2"])
```

This prints a bunch of gathers related to pfor infrastructure, but at the time
of writing does include one with a `batch_dims` attribute of `1`.

## Vectorization as a post-trace graph transformation

`tf.vectorized_map` currently works as a graph-to-graph transformation
implemented in Python. This is mostly a historical artifact: it was conceived
before TensorFlow did op-by-op execution. It is similar to `tf.gradients`,
walking the connections in an existing graph and applying op-specific rules
(defined by `RegisterGradient` for `tf.gradients`, `RegisterPFor` for
`tf.vectorized_map`) in order to produce a new graph. While `tf.gradients` adds
a backward pass which references tensors in the forward pass (both execute),
`tf.vectorized_map` creates a transformed graph which executes in place of the
original graph.

For gradients, `tf.GradientTape` was introduced to provide an op-by-op version
of gradients, re-using the per-op `RegisterGradient` definitions. There is no
equivalent for vectorization. Instead, `tf.vectorized_map` wraps the function it
takes as an argument in `tf.function` in order to create a trace to
vectorize. This means that the user's function never executes eagerly, and if
`tf.vectorized_map` is called executing eagerly that the user's function is
re-traced and re-vectorized every call to `tf.vectorized_map`.

While `tf.vectorized_map` is the public-facing API, the implementation is
written in terms of [an integer-indexed for
loop](https://github.com/tensorflow/tensorflow/blob/8b000ce0d5395d399e08791ae9589b41358f651d/tensorflow/python/ops/parallel_for/control_flow_ops.py#L134). The
loop does not execute as a regular for loop, but this is a good mental model and
the implementation makes frequent references to `loop_len`, i.e. the number of
iterations for the hypothetical loop. The user-visible outputs should ideally
match the outputs of an equivalent real for loop, and this is how most of the
unit tests are written.

The virtual for loop setup includes a loop-variant integer loop index
tensor. "Loop-variant" just means a tensor with a different value on each
iteration; loop-variant tensors are represented with a leading extra dimension
corresponding to the loop iteration. `tf.vectorized_map`'s implementation
`tf.gather`s a slice of each input using the loop index and then runs the user's
function on those (loop-variant) values.

Anything with the loop index in its transitive input is loop-variant and must be
transformed. Ops like `tf.constant`, however, create loop-invariant values
(i.e. their values are the same on each loop iteration). Loop-invariant values
returned from vectorization may simply be tiled, but more frequently they feed
into ops with a mix of variant/invariant inputs to produce loop-variant
values. Converters for ops will sometimes have simpler special cases for
loop-invariant inputs, e.g. `tf.roll`'s converter is much simpler if the shift
amount is loop-invariant (a common case).

## Defining vectorizations

As with gradients, most ops have relatively straightforward definitions and
function call / control flow operations have complicated special cases. This
section covers the common cases.

As with `RegisterGradient`, converters are defined for op types (with a
corresponding `REGISTER_OP` macro in C++, named WithUppercase), not Python
endpoints. So if the user writes `tf.roll`, the corresponding
[`RegisterPFor("Roll")`
converter](https://github.com/tensorflow/tensorflow/blob/349172cf0ac29ba1346d244a40dc4761b4600f2e/tensorflow/python/ops/parallel_for/pfor.py#L2653)
is triggered since `tf.roll` is implemented with the "Roll" op.

Like gradients, the set of all TensorFlow ops would ideally be closed under
vectorization (i.e. vectorization would always produce ops which are themselves
vectorizable). In practice not all ops have pfor converters defined, and those
that do sometimes assume inputs are loop-invariant. A "fallback converter" runs
in these cases, adding a `tf.while_loop` in place of a real vectorization for
the op. This is safe but generally slower. `tf.while_loop` can run iterations in
parallel (non-strict execution), but other benefits of vectorization like
contiguous memory layouts and SIMD are not available.

For stateless ops which compute a deterministic value from their inputs, the
common case, pfor converters take loop-variant inputs with an extra dimension
("stacked") and emit an op or subgraph which treats this extra stacked dimension
(the tensor's zeroth dimension) as a batch dimension but otherwise computes the
same value as the original op. This may involve examining the original op's
attributes and forwarding them to newly emitted ops.

The general case for a converter has every input stacked and loop-variant, but
there are often more efficient special cases when some inputs are loop-invariant
and so may be handled "unstacked" (with no special zeroth dimension). Some
converters omit the general case entirely, simply requesting the unstacked input
and relying on the fallback converter triggering if that fails because the input
is loop-variant. Others have branches for various combinations of
stacked/unstacked inputs.

There are many examples of existing converters, all in
tensorflow/python/ops/parallel_for/pfor.py (the user-facing APIs are defined in
control_flow_ops.py in the same directory). Converters can be quite subtle, so
it is important to use the unit test macros to compare to a ground-truth for
loop and to ensure that all relevant combinations of loop variant/invariant
inputs are covered by these tests.

## Stateful ops

Stateful ops and ops with non-deterministic outputs are difficult to deal
with. One option is to use the fallback `tf.while_loop` converter for these ops,
so e.g. `tf.print` would print `loop_len` times with the different loop-variant
values. This makes sense from the "for loop as ground truth" mindset, but it's
less clear that this satisfies user expectations for `tf.vectorized_map` (which
doesn't explicitly mention a loop).

There isn't a great universal answer for this class of ops. Currently `tf.print`
[prints the full vectorized
tensors](https://github.com/tensorflow/tensorflow/blob/349172cf0ac29ba1346d244a40dc4761b4600f2e/tensorflow/python/ops/parallel_for/pfor.py#L3505-L3522) ([example](https://github.com/tensorflow/tensorflow/blob/8b202f08d52e8206af2bdb2112a62fafbc546ec7/tensorflow/python/ops/parallel_for/control_flow_ops_test.py#L956-L970))
rather than printing `loop_len` times. Stateful random ops are [vectorized by
adding an extra dimension to their output](https://github.com/tensorflow/tensorflow/blob/349172cf0ac29ba1346d244a40dc4761b4600f2e/tensorflow/python/ops/parallel_for/pfor.py#L3276-L3294) shape attributes, even though this
gives a different result (but follows the same distribution / independence
structure). Stateless random ops [use the `tf.while_loop` fallback converter](https://github.com/tensorflow/tensorflow/blob/349172cf0ac29ba1346d244a40dc4761b4600f2e/tensorflow/python/ops/parallel_for/pfor.py#L3360-L3377)
since users might care more about the exact values; this may want revisiting if
stateless random ops are used to implement popular APIs.

## Vectorization of control flow (while_loop, cond) and variants

Ops whose execution is defined by a serialized program (generally a FunctionDef
referenced by name in an attribute) need special handling, since the vectorized
op will reference a transformed serialized program.

For function call operations this is relatively straightforward: the converter
converts the function body and generates a new call operation referencing the
vectorized function body. (See `RegisterPfor("PartitionedCall")`; the code is
pretty readable.)

Cond (["If"/"StatelessIf" ops](https://github.com/tensorflow/tensorflow/blob/349172cf0ac29ba1346d244a40dc4761b4600f2e/tensorflow/python/ops/parallel_for/pfor.py#L4499); [example](https://github.com/tensorflow/tensorflow/blob/8b202f08d52e8206af2bdb2112a62fafbc546ec7/tensorflow/python/ops/parallel_for/control_flow_ops_test.py#L2041-L2053)) can be a bit more complicated if the Boolean
condition is loop variant, in which case inputs/outputs must be partitioned
between the branches and both run (although the ops in one branch could have
zero-sized inputs if the loop variant condition happened to not trigger that
branch for any iteration of the virtual for loop). If the condition Boolean is
loop invariant then cond is very similar to a function call operation, just with
two function bodies to transform.

[While loop vectorization](https://github.com/tensorflow/tensorflow/blob/349172cf0ac29ba1346d244a40dc4761b4600f2e/tensorflow/python/ops/parallel_for/pfor.py#L5001) is fairly complicated. This is unrelated to the
fallback converter for ops; it triggers when users define a graph with a
`tf.while_loop` and then request vectorization for it (although the fallback
converter can trigger this case when `tf.vectorized_map` is nested). At a high
level, while loop conversion is an iterative version of the
loop-variant-condition cond conversion. Only one while loop runs in the
vectorized graph, but it keeps track of which iterations of the virtual pfor
loop are done and only runs the while loop body for corresponding inputs. Once
all of the iterations of the virtual pfor loop would have finished their
`tf.while_loop`s the single vectorized loop terminates.

While loops accumulate values across iterations in TensorLists (aka
TensorArrays). These are variant-dtype tensors with a C++ vector of pointers to
other tensors. A straightforward conversion would simply stack variant tensors,
so rather than scalar variant-dtype tensors they would have shape
`[loop_len]`. However, this would make memory non-contiguous: the tensors across
each iteration of the virtual pfor loop would be separate Tensor objects in C++,
and concatenation / splitting would be necessary to push and pop
tensors. Instead, TensorLists are special-cased to use "internal vectorization":
the variant representing a vectorized/stacked TensorList remains a scalar, but
the shape of the tensors it contains has a special zeroth dimension. This makes
many common operations on vectorized TensorLists more efficient, but leads to
some complicated special cases when accessing the vectorization dimension.
