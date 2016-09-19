A subgraph view on an existing tf.Graph.

An instance of this class is a subgraph view on an existing tf.Graph.
"subgraph" means that it can represent part of the whole tf.Graph.
"view" means that it only provides a passive observation and do not to act
on the tf.Graph. Note that in this documentation, the term "subgraph" is often
used as substitute to "subgraph view".

A subgraph contains:
* a list of input tensors, accessible via the "inputs" property.
* a list of output tensors, accessible via the "outputs" property.
* and the operations in between, accessible via the "ops" property.

An subgraph can be seen as a function F(i0, i1, ...) -> o0, o1, ... It is a
function which takes as input some input tensors and returns as output some
output tensors. The computation that the function performs is encoded in the
operations of the subgraph.

The tensors (input or output) can be of two kinds:
- connected: a connected tensor connects to at least one operation contained
in the subgraph. One example is a subgraph representing a single operation
and its inputs and outputs: all the input and output tensors of the op
are "connected".
- passthrough: a passthrough tensor does not connect to any operation
contained in the subgraph. One example is a subgraph representing a
single tensor: this tensor is passthrough. By default a passthrough tensor is
present both in the input and output tensors of the subgraph. It can however
be remapped to only appear as an input (or output) only.

The input and output tensors can be remapped. For instance, some input tensor
can be ommited. For instance, a subgraph representing an operation with two
inputs can be remapped to only take one input. Note that this does not change
at all the underlying tf.Graph (remember, it is a view). It means that
the other input is being ignored, or is being treated as "given".
The analogy with functions can be extended like this: F(x,y) is the original
function. Remapping the inputs from [x, y] to just [x] means that the subgraph
now represent the function F_y(x) (y is "given").

The output tensors can also be remapped. For instance, some output tensor can
be ommited. Other output tensor can be duplicated as well. As mentioned
before, this does not change at all the underlying tf.Graph.
The analogy with functions can be extended like this: F(...)->x,y is the
original function. Remapping the outputs from [x, y] to just [y,y] means that
the subgraph now represent the function M(F(...)) where M is the function
M(a,b)->b,b.

It is useful to describe three other kind of tensors:
* internal: an internal tensor is a tensor connecting operations contained
  in the subgraph. One example in the subgraph representing the two
  operations A and B connected sequentially: -> A -> B ->. The middle arrow
  is an internal tensor.
* actual input: an input tensor of the subgraph, regardless of whether it is
  listed in "inputs" or not (masked-out).
* actual output: an output tensor of the subgraph, regardless of whether it is
  listed in "outputs" or not (masked-out).
* hidden input: an actual input which has been masked-out using an
  input remapping. In other word, a hidden input is a non-internal tensor
  not listed as a input tensor and one of whose consumers belongs to
  the subgraph.
* hidden output: a actual output which has been masked-out using an output
  remapping. In other word, a hidden output is a non-internal tensor
  not listed as an output and one of whose generating operations belongs to
  the subgraph.

Here are some usefull guarantees about an instance of a SubGraphView:
* the input (or output) tensors are not internal.
* the input (or output) tensors are either "connected" or "passthrough".
* the passthrough tensors are not connected to any of the operation of
the subgraph.

Note that there is no guarantee that an operation in a subgraph contributes
at all to its inputs or outputs. For instance, remapping both the inputs and
outputs to empty lists will produce a subgraph which still contains all the
original operations. However, the remove_unused_ops function can be used to
make a new subgraph view whose operations are connected to at least one of
the input or output tensors.

An instance of this class is meant to be a lightweight object which is not
modified in-place by the user. Rather, the user can create new modified
instances of a given subgraph. In that sense, the class SubGraphView is meant
to be used like an immutable python object.

A common problem when using views is that they can get out-of-sync with the
data they observe (in this case, a tf.Graph). This is up to the user to insure
that this doesn't happen. To keep on the safe sife, it is recommended that
the life time of subgraph views are kept very short. One way to achieve this
is to use subgraphs within a "with make_sgv(...) as sgv:" Python context.

To alleviate the out-of-sync problem, some functions are granted the right to
modified subgraph in place. This is typically the case of graph manipulation
functions which, given some subgraphs as arguments, can modify the underlying
tf.Graph. Since this modification is likely to render the subgraph view
invalid, those functions can modify the argument in place to reflect the
change. For instance, calling the function swap_inputs(svg0, svg1) will modify
svg0 and svg1 in place to reflect the fact that their inputs have now being
swapped.
- - -

#### `tf.contrib.graph_editor.SubGraphView.__bool__()` {#SubGraphView.__bool__}

Allows for implicit boolean conversion.


- - -

#### `tf.contrib.graph_editor.SubGraphView.__copy__()` {#SubGraphView.__copy__}

Create a copy of this subgraph.

Note that this class is a "view", copying it only create another view and
does not copy the underlying part of the tf.Graph.

##### Returns:

  A new identical instance of the original subgraph view.


- - -

#### `tf.contrib.graph_editor.SubGraphView.__enter__()` {#SubGraphView.__enter__}

Allow Python context to minize the life time of a subgraph view.

A subgraph view is meant to be a lightweight and transient object. A short
lifetime will alleviate the "out-of-sync" issue mentioned earlier. For that
reason, a SubGraphView instance can be used within a Python context. For
example:

from tensorflow.contrib import graph_editor as ge
with ge.make_sgv(...) as sgv:
  print(sgv)

##### Returns:

  Itself.


- - -

#### `tf.contrib.graph_editor.SubGraphView.__exit__(exc_type, exc_value, traceback)` {#SubGraphView.__exit__}




- - -

#### `tf.contrib.graph_editor.SubGraphView.__init__(inside_ops=(), passthrough_ts=())` {#SubGraphView.__init__}

Create a subgraph containing the given ops and the "passthrough" tensors.

##### Args:


*  <b>`inside_ops`</b>: an object convertible to a list of tf.Operation. This list
    defines all the operations in the subgraph.
*  <b>`passthrough_ts`</b>: an object convertible to a list of tf.Tensor. This list
    define all the "passthrough" tensors. A passthrough tensor is a tensor
    which goes directly from the input of the subgraph to it output, without
    any intermediate operations. All the non passthrough tensors are
    silently ignored.

##### Raises:


*  <b>`TypeError`</b>: if inside_ops cannot be converted to a list of tf.Operation or
    if passthrough_ts cannot be converted to a list of tf.Tensor.


- - -

#### `tf.contrib.graph_editor.SubGraphView.__nonzero__()` {#SubGraphView.__nonzero__}

Allows for implicit boolean conversion.


- - -

#### `tf.contrib.graph_editor.SubGraphView.__str__()` {#SubGraphView.__str__}




- - -

#### `tf.contrib.graph_editor.SubGraphView.connected_inputs` {#SubGraphView.connected_inputs}

The connected input tensors of this subgraph view.


- - -

#### `tf.contrib.graph_editor.SubGraphView.connected_outputs` {#SubGraphView.connected_outputs}

The connected output tensors of this subgraph view.


- - -

#### `tf.contrib.graph_editor.SubGraphView.consumers()` {#SubGraphView.consumers}

Return a Python set of all the consumers of this subgraph view.


- - -

#### `tf.contrib.graph_editor.SubGraphView.copy()` {#SubGraphView.copy}

Return a copy of itself.

Note that this class is a "view", copying it only create another view and
does not copy the underlying part of the tf.Graph.

##### Returns:

  A new instance identical to the original one.


- - -

#### `tf.contrib.graph_editor.SubGraphView.find_op_by_name(op_name)` {#SubGraphView.find_op_by_name}

Return the op named op_name.

##### Args:


*  <b>`op_name`</b>: the name to search for

##### Returns:

  The op named op_name.

##### Raises:


*  <b>`ValueError`</b>: if the op_name could not be found.
*  <b>`AssertionError`</b>: if the name was found multiple time.


- - -

#### `tf.contrib.graph_editor.SubGraphView.graph` {#SubGraphView.graph}

The underlying tf.Graph.


- - -

#### `tf.contrib.graph_editor.SubGraphView.input_index(t)` {#SubGraphView.input_index}

Find the input index corresponding to the given input tensor t.

##### Args:


*  <b>`t`</b>: the input tensor of this subgraph view.

##### Returns:

  The index in the self.inputs list.

##### Raises:


*  <b>`Error`</b>: if t in not an input tensor.


- - -

#### `tf.contrib.graph_editor.SubGraphView.inputs` {#SubGraphView.inputs}

The input tensors of this subgraph view.


- - -

#### `tf.contrib.graph_editor.SubGraphView.is_passthrough(t)` {#SubGraphView.is_passthrough}

Check whether a tensor is passthrough.


- - -

#### `tf.contrib.graph_editor.SubGraphView.op(op_id)` {#SubGraphView.op}

Get an op by its index.


- - -

#### `tf.contrib.graph_editor.SubGraphView.ops` {#SubGraphView.ops}

The operations in this subgraph view.


- - -

#### `tf.contrib.graph_editor.SubGraphView.output_index(t)` {#SubGraphView.output_index}

Find the output index corresponding to given output tensor t.

##### Args:


*  <b>`t`</b>: the output tensor of this subgraph view.

##### Returns:

  The index in the self.outputs list.

##### Raises:


*  <b>`Error`</b>: if t in not an output tensor.


- - -

#### `tf.contrib.graph_editor.SubGraphView.outputs` {#SubGraphView.outputs}

The output tensors of this subgraph view.


- - -

#### `tf.contrib.graph_editor.SubGraphView.passthroughs` {#SubGraphView.passthroughs}

The passthrough tensors, going straight from input to output.


- - -

#### `tf.contrib.graph_editor.SubGraphView.remap(new_input_indices=None, new_output_indices=None)` {#SubGraphView.remap}

Remap the inputs and outputs of the subgraph.

Note that this is only modifying the view: the underlying tf.Graph is not
affected.

##### Args:


*  <b>`new_input_indices`</b>: an iterable of integers representing a mapping between
    the old inputs and the new ones. This mapping can be under-complete and
    must be without repetitions.
*  <b>`new_output_indices`</b>: an iterable of integers representing a mapping between
    the old outputs and the new ones. This mapping can be under-complete and
    can have repetitions.

##### Returns:

  A new modified instance of the original subgraph view with remapped
    inputs and outputs.


- - -

#### `tf.contrib.graph_editor.SubGraphView.remap_default(remove_input_map=True, remove_output_map=True)` {#SubGraphView.remap_default}

Remap the inputs and/or outputs to the default mapping.

##### Args:


*  <b>`remove_input_map`</b>: if True the input map is reset to the default one.
*  <b>`remove_output_map`</b>: if True the output map is reset to the default one.

##### Returns:

  A new modified instance of the original subgraph view with its
    input and/or output mapping reset to the default one.


- - -

#### `tf.contrib.graph_editor.SubGraphView.remap_inputs(new_input_indices)` {#SubGraphView.remap_inputs}

Remap the inputs of the subgraph.

If the inputs of the original subgraph are [t0, t1, t2], remapping to [2,0]
will create a new instance whose inputs is [t2, t0].

Note that this is only modifying the view: the underlying tf.Graph is not
affected.

##### Args:


*  <b>`new_input_indices`</b>: an iterable of integers representing a mapping between
    the old inputs and the new ones. This mapping can be under-complete and
    must be without repetitions.

##### Returns:

  A new modified instance of the original subgraph view with remapped
    inputs.


- - -

#### `tf.contrib.graph_editor.SubGraphView.remap_outputs(new_output_indices)` {#SubGraphView.remap_outputs}

Remap the output of the subgraph.

If the output of the original subgraph are [t0, t1, t2], remapping to
[1,1,0] will create a new instance whose outputs is [t1, t1, t0].

Note that this is only modifying the view: the underlying tf.Graph is not
affected.

##### Args:


*  <b>`new_output_indices`</b>: an iterable of integers representing a mapping between
    the old outputs and the new ones. This mapping can be under-complete and
    can have repetitions.

##### Returns:

  A new modified instance of the original subgraph view with remapped
    outputs.


- - -

#### `tf.contrib.graph_editor.SubGraphView.remap_outputs_make_unique()` {#SubGraphView.remap_outputs_make_unique}

Remap the outputs so that all the tensors appears only once.


- - -

#### `tf.contrib.graph_editor.SubGraphView.remap_outputs_to_consumers()` {#SubGraphView.remap_outputs_to_consumers}

Remap the outputs to match the number of consumers.


- - -

#### `tf.contrib.graph_editor.SubGraphView.remove_unused_ops(control_inputs=True)` {#SubGraphView.remove_unused_ops}

Remove unused ops.

##### Args:


*  <b>`control_inputs`</b>: if True, control inputs are used to detect used ops.

##### Returns:

  A new subgraph view which only contains used operations.


