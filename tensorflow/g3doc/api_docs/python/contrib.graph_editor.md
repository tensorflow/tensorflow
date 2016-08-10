<!-- This file is machine generated: DO NOT EDIT! -->

# Graph Editor (contrib)
[TOC]

Graph editor module allows to modify an existing graph in place.

## Other Functions and Classes
- - -

### `class tf.contrib.graph_editor.SubGraphView` {#SubGraphView}

A subgraph view on an existing tf.Graph.

An instance of this class is a subgraph view on an existing tf.Graph.
"subgraph" means that it can represent part of the whole tf.Graph.
"view" means that it only provides a passive observation and do not to act
on the tf.Graph. Note that in this documentation, the term "subgraph" is often
used as substitute to "subgraph view".

A subgraph contains:
- a list of input tensors, accessible via the "inputs" property.
- a list of output tensors, accessible via the "outputs" property.
- and the operations in between, accessible via the "ops" property.

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
- internal: an internal tensor is a tensor connecting operations contained
in the subgraph. One example in the subgraph representing the two operations
A and B connected sequentially: -> A -> B ->. The middle arrow is an internal
tensor.
- actual input: an input tensor of the subgraph, regardless of whether it is
  listed in "inputs" or not (masked-out).
- actual output: an output tensor of the subgraph, regardless of whether it is
  listed in "outputs" or not (masked-out).
- hidden input: an actual input which has been masked-out using an
  input remapping. In other word, a hidden input is a non-internal tensor
  not listed as a input tensor and one of whose consumers belongs to
  the subgraph.
- hidden output: a actual output which has been masked-out using an output
  remapping. In other word, a hidden output is a non-internal tensor
  not listed as an output and one of whose generating operations belongs to
  the subgraph.

Here are some usefull guarantees about an instance of a SubGraphView:
- the input (or output) tensors are not internal.
- the input (or output) tensors are either "connected" or "passthrough".
- the passthrough tensors are not connected to any of the operation of
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

  a new instance identical to the original one.


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

  the index in the self.inputs list.

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

  the index in the self.outputs list.

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



- - -

### `class tf.contrib.graph_editor.Transformer` {#Transformer}

Transform a subgraph into another one.

By default, the constructor create a transform which copy a subgraph and
replaces inputs with placeholders. This behavior can be modified by changing
the handlers.
- - -

#### `tf.contrib.graph_editor.Transformer.__init__()` {#Transformer.__init__}

Transformer constructor.

The following members can be modified:
transform_op_handler: handle the transformation of a tf.Operation.
  This handler defaults to a simple copy.
assign_collections_handler: handle the assignment of collections.
  This handler defaults to assigning new collections created under the
  given name-scope.
transform_input_handler: handle the transform of the inputs to the given
  subgraph. This handler defaults to creating placeholders instead of the
  ops just before the input tensors of the subgraph.
transform_hidden_input_handler: handle the transform of the hidden inputs of
  the subgraph, that is, the inputs which are not listed in sgv.inputs.
  This handler defaults to a transform which keep the same input if the
  source and destination graphs are the same, otherwise use placeholders.
transform_original_op_hanlder: handle the transform of original_op. This
  handler defaults to transforming original_op only if they are in the
  subgraph, otherwise they are ignored.


- - -

#### `tf.contrib.graph_editor.Transformer.new_name(name)` {#Transformer.new_name}

Compute a destination name from a source name.

##### Args:


*  <b>`name`</b>: the name to be "transformed".

##### Returns:

  the transformed name.

##### Raises:


*  <b>`ValueError`</b>: if the source scope is used (that is, not an empty string)
    and the source name does not belong to the source scope.



- - -

### `tf.contrib.graph_editor.bypass(sgv)` {#bypass}

Bypass the given subgraph by connecting its inputs to its outputs.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be bypassed. This argument is converted to a
    subgraph using the same rules than the function subgraph.make_view.

##### Returns:

  A new subgraph view of the bypassed subgraph.
    Note that sgv is also modified in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.connect(sgv0, sgv1, disconnect_first=False)` {#connect}

Connect the outputs of sgv0 to the inputs of sgv1.

##### Args:


*  <b>`sgv0`</b>: the first subgraph to have its outputs swapped. This argument is
    converted to a subgraph using the same rules as the function
    subgraph.make_view.
*  <b>`sgv1`</b>: the second subgraph to have its outputs swapped. This argument is
    converted to a subgraph using the same rules as the function
    subgraph.make_view.
*  <b>`disconnect_first`</b>: if True the current outputs of sgv0 are disconnected.

##### Returns:

  Two new subgraph views (now connected). sgv0 and svg1 are also modified
    in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv0 or sgv1 cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.detach(sgv, control_inputs=False, control_outputs=None, control_ios=None)` {#detach}

Detach both the inputs and the outputs of a subgraph view.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of util.ControlOutputs or None. If not None,
    control outputs are enabled.
*  <b>`control_ios`</b>: An instance of util.ControlOutputs or None. If not None, both
    control inputs and control outputs are enabled. This is equivalent to set
    control_inputs to True and control_outputs to the util.ControlOutputs
    instance.

##### Returns:

  A new subgraph view of the detached subgraph.
    Note that sgv is also modified in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.detach_inputs(sgv, control_inputs=False)` {#detach_inputs}

Detach the inputs of a subgraph view.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
*  <b>`control_inputs`</b>: if True control_inputs are also detached.

##### Returns:

  A new subgraph view of the detached subgraph.
    Note that sgv is also modified in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.detach_outputs(sgv, control_outputs=None)` {#detach_outputs}

Detach the outputa of a subgraph view.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
*  <b>`control_outputs`</b>: a util.ControlOutputs instance or None. If not None the
    control outputs are also detached.

##### Returns:

  A new subgraph view of the detached subgraph.
    Note that sgv is also modified in place.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `class tf.contrib.graph_editor.matcher` {#matcher}

Graph match class.
- - -

#### `tf.contrib.graph_editor.matcher.__init__(positive_filter)` {#matcher.__init__}

Graph match constructor.


- - -

#### `tf.contrib.graph_editor.matcher.control_input_ops(*args)` {#matcher.control_input_ops}

Add input matches.


- - -

#### `tf.contrib.graph_editor.matcher.input_ops(*args)` {#matcher.input_ops}

Add input matches.


- - -

#### `tf.contrib.graph_editor.matcher.output_ops(*args)` {#matcher.output_ops}

Add output matches.



- - -

### `tf.contrib.graph_editor.ph(dtype, shape=None, scope=None)` {#ph}

Create a tf.placeholder for the Graph Editor.

Note that the correct graph scope must be set by the calling function.
The placeholder is named using the function placeholder_name (with no
tensor argument).

##### Args:


*  <b>`dtype`</b>: the tensor type.
*  <b>`shape`</b>: the tensor shape (optional).
*  <b>`scope`</b>: absolute scope within which to create the placeholder. None
    means that the scope of t is preserved. "" means the root scope.

##### Returns:

  A newly created tf.placeholder.


- - -

### `tf.contrib.graph_editor.reroute_a2b(sgv0, sgv1)` {#reroute_a2b}

Re-route the inputs and outputs of sgv0 to sgv1 (see _reroute).


- - -

### `tf.contrib.graph_editor.reroute_a2b_inputs(sgv0, sgv1)` {#reroute_a2b_inputs}

Re-route all the inputs of sgv0 to sgv1 (see reroute_inputs).


- - -

### `tf.contrib.graph_editor.reroute_a2b_outputs(sgv0, sgv1)` {#reroute_a2b_outputs}

Re-route all the outputs of sgv0 to sgv1 (see _reroute_outputs).


- - -

### `tf.contrib.graph_editor.reroute_b2a(sgv0, sgv1)` {#reroute_b2a}

Re-route the inputs and outputs of sgv1 to sgv0 (see _reroute).


- - -

### `tf.contrib.graph_editor.reroute_b2a_inputs(sgv0, sgv1)` {#reroute_b2a_inputs}

Re-route all the inputs of sgv1 to sgv0 (see reroute_inputs).


- - -

### `tf.contrib.graph_editor.reroute_b2a_outputs(sgv0, sgv1)` {#reroute_b2a_outputs}

Re-route all the outputs of sgv1 to sgv0 (see _reroute_outputs).


- - -

### `tf.contrib.graph_editor.select_ops(*args, **kwargs)` {#select_ops}

Helper to select operations.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    tf.Operation. tf.Tensor instances are silently ignored.
*  <b>`**kwargs`</b>: 'graph': tf.Graph in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if positive_filter(elem) is
      True. This is optional.
    'restrict_ops_regex': a regular expression is ignored if it doesn't start
      with the substring "(?#ops)".

##### Returns:

  list of tf.Operation

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a tf.Graph
    or if an argument in args is not an (array of) tf.Operation
    or an (array of) tf.Tensor (silently ignored) or a string
    or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.


- - -

### `tf.contrib.graph_editor.select_ts(*args, **kwargs)` {#select_ts}

Helper to select tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    tf.Tensor. tf.Operation instances are silently ignored.
*  <b>`**kwargs`</b>: 'graph': tf.Graph in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if positive_filter(elem) is
      True. This is optional.
    'restrict_ts_regex': a regular expression is ignored if it doesn't start
      with the substring "(?#ts)".

##### Returns:

  list of tf.Tensor

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a tf.Graph
    or if an argument in args is not an (array of) tf.Tensor
    or an (array of) tf.Operation (silently ignored) or a string
    or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.


- - -

### `tf.contrib.graph_editor.sgv(*args, **kwargs)` {#sgv}

Create a SubGraphView from selected operations and passthrough tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    tf.Operation 3) (array of) tf.Tensor. Those objects will be converted
    into a list of operations and a list of candidate for passthrough tensors.
*  <b>`**kwargs`</b>: keyword graph is used 1) to check that the ops and ts are from
    the correct graph 2) for regular expression query

##### Returns:

  A subgraph view.

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a tf.Graph
    or if an argument in args is not an (array of) tf.Tensor
    or an (array of) tf.Operation or a string or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected.


- - -

### `tf.contrib.graph_editor.sgv_scope(scope, graph)` {#sgv_scope}

Make a subgraph from a name scope.

##### Args:


*  <b>`scope`</b>: the name of the scope.
*  <b>`graph`</b>: the tf.Graph.

##### Returns:

  A subgraph view representing the given scope.


- - -

### `tf.contrib.graph_editor.swap(sgv0, sgv1)` {#swap}

Swap the inputs and outputs of sgv1 to sgv0 (see _reroute).


- - -

### `tf.contrib.graph_editor.swap_inputs(sgv0, sgv1)` {#swap_inputs}

Swap all the inputs of sgv0 and sgv1 (see reroute_inputs).


- - -

### `tf.contrib.graph_editor.swap_outputs(sgv0, sgv1)` {#swap_outputs}

Swap all the outputs of sgv0 and sgv1 (see _reroute_outputs).


- - -

### `tf.contrib.graph_editor.ts(*args, **kwargs)` {#ts}

Helper to select tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    tf.Tensor. tf.Operation instances are silently ignored.
*  <b>`**kwargs`</b>: 'graph': tf.Graph in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if positive_filter(elem) is
      True. This is optional.
    'restrict_ts_regex': a regular expression is ignored if it doesn't start
      with the substring "(?#ts)".

##### Returns:

  list of tf.Tensor

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a tf.Graph
    or if an argument in args is not an (array of) tf.Tensor
    or an (array of) tf.Operation (silently ignored) or a string
    or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.


