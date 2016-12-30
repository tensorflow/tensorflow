<!-- This file is machine generated: DO NOT EDIT! -->

# Graph Editor (contrib)
[TOC]

TensorFlow Graph Editor.

The TensorFlow Graph Editor library allows for modification of an existing
`tf.Graph` instance in-place.

The author's github username is [purpledog](https://github.com/purpledog).

## Library overview

Appending new nodes is the only graph editing operation allowed by the
TensorFlow core library. The Graph Editor library is an attempt to allow for
other kinds of editing operations, namely, *rerouting* and *transforming*.

* *rerouting* is a local operation consisting in re-plugging existing tensors
  (the edges of the graph). Operations (the nodes) are not modified by this
  operation. For example, rerouting can be used to insert an operation adding
  noise in place of an existing tensor.
* *transforming* is a global operation consisting in transforming a graph into
  another. By default, a transformation is a simple copy but it can be
  customized to achieved other goals. For instance, a graph can be transformed
  into another one in which noise is added after all the operations of a
  specific type.

**Important: modifying a graph in-place with the Graph Editor must be done
`offline`, that is, without any active sessions.**

Of course new operations can be appended online but Graph Editor specific
operations like rerouting and transforming can currently only be done offline.

Here is an example of what you **cannot** do:

* Build a graph.
* Create a session and run the graph.
* Modify the graph with the Graph Editor.
* Re-run the graph with the `same` previously created session.

To edit an already running graph, follow these steps:

* Build a graph.
* Create a session and run the graph.
* Save the graph state and terminate the session
* Modify the graph with the Graph Editor.
* create a new session and restore the graph state
* Re-run the graph with the newly created session.

Note that this procedure is very costly because a new session must be created
after any modifications. Among other things, it takes time because the entire
graph state must be saved and restored again.

## Sub-graph

Most of the functions in the Graph Editor library operate on *sub-graph*.
More precisely, they take as input arguments instances of the SubGraphView class
(or anything which can be converted to it). Doing so allows the same function
to transparently operate on single operations as well as sub-graph of any size.

A subgraph can be created in several ways:

* using a list of ops:

```python
my_sgv = ge.sgv(ops)
```

* from a name scope:

```python
my_sgv = ge.sgv_scope("foo/bar", graph=tf.get_default_graph())
```

* using regular expression:

```python
my_sgv = ge.sgv("foo/.*/.*read$", graph=tf.get_default_graph())
```

Note that the Graph Editor is meant to manipulate several graphs at the same
time, typically during transform or copy operation. For that reason,
to avoid any confusion, the default graph is never used and the graph on
which to operate must always be given explicitly. This is the reason why
*`graph=tf.get_default_graph()`* is used in the code snippets above.

## Modules overview

* util: utility functions.
* select: various selection methods of TensorFlow tensors and operations.
* match: TensorFlow graph matching. Think of this as regular expressions for
  graphs (but not quite yet).
* reroute: various ways of rerouting tensors to different consuming ops like
  *swap* or *reroute_a2b*.
* subgraph: the SubGraphView class, which enables subgraph manipulations in a
  TensorFlow `tf.Graph`.
* edit: various editing functions operating on subgraphs like *detach*,
  *connect* or *bypass*.
* transform: the Transformer class, which enables transforming
  (or simply copying) a subgraph into another one.

## Module: util

- - -

### `tf.contrib.graph_editor.make_list_of_op(ops, check_graph=True, allow_graph=True, ignore_ts=False)` {#make_list_of_op}

Convert ops to a list of `tf.Operation`.

##### Args:


*  <b>`ops`</b>: can be an iterable of `tf.Operation`, a `tf.Graph` or a single
    operation.
*  <b>`check_graph`</b>: if `True` check if all the operations belong to the same graph.
*  <b>`allow_graph`</b>: if `False` a `tf.Graph` cannot be converted.
*  <b>`ignore_ts`</b>: if True, silently ignore `tf.Tensor`.

##### Returns:

  A newly created list of `tf.Operation`.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of `tf.Operation` or,
   if `check_graph` is `True`, if all the ops do not belong to the
   same graph.


- - -

### `tf.contrib.graph_editor.get_tensors(graph)` {#get_tensors}

get all the tensors which are input or output of an op in the graph.

##### Args:


*  <b>`graph`</b>: a `tf.Graph`.

##### Returns:

  A list of `tf.Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if graph is not a `tf.Graph`.


- - -

### `tf.contrib.graph_editor.make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False)` {#make_list_of_t}

Convert ts to a list of `tf.Tensor`.

##### Args:


*  <b>`ts`</b>: can be an iterable of `tf.Tensor`, a `tf.Graph` or a single tensor.
*  <b>`check_graph`</b>: if `True` check if all the tensors belong to the same graph.
*  <b>`allow_graph`</b>: if `False` a `tf.Graph` cannot be converted.
*  <b>`ignore_ops`</b>: if `True`, silently ignore `tf.Operation`.

##### Returns:

  A newly created list of `tf.Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `ts` cannot be converted to a list of `tf.Tensor` or,
   if `check_graph` is `True`, if all the ops do not belong to the same graph.


- - -

### `tf.contrib.graph_editor.get_generating_ops(ts)` {#get_generating_ops}

Return all the generating ops of the tensors in `ts`.

##### Args:


*  <b>`ts`</b>: a list of `tf.Tensor`

##### Returns:

  A list of all the generating `tf.Operation` of the tensors in `ts`.

##### Raises:


*  <b>`TypeError`</b>: if `ts` cannot be converted to a list of `tf.Tensor`.


- - -

### `tf.contrib.graph_editor.get_consuming_ops(ts)` {#get_consuming_ops}

Return all the consuming ops of the tensors in ts.

##### Args:


*  <b>`ts`</b>: a list of `tf.Tensor`

##### Returns:

  A list of all the consuming `tf.Operation` of the tensors in `ts`.

##### Raises:


*  <b>`TypeError`</b>: if ts cannot be converted to a list of `tf.Tensor`.


- - -

### `class tf.contrib.graph_editor.ControlOutputs` {#ControlOutputs}

The control outputs topology.
- - -

#### `tf.contrib.graph_editor.ControlOutputs.__init__(graph)` {#ControlOutputs.__init__}

Create a dictionary of control-output dependencies.

##### Args:


*  <b>`graph`</b>: a `tf.Graph`.

##### Returns:

  A dictionary where a key is a `tf.Operation` instance and the
     corresponding value is a list of all the ops which have the key
     as one of their control-input dependencies.

##### Raises:


*  <b>`TypeError`</b>: graph is not a `tf.Graph`.


- - -

#### `tf.contrib.graph_editor.ControlOutputs.get(op)` {#ControlOutputs.get}

return the control outputs of op.


- - -

#### `tf.contrib.graph_editor.ControlOutputs.get_all()` {#ControlOutputs.get_all}




- - -

#### `tf.contrib.graph_editor.ControlOutputs.graph` {#ControlOutputs.graph}




- - -

#### `tf.contrib.graph_editor.ControlOutputs.update()` {#ControlOutputs.update}

Update the control outputs if the graph has changed.



- - -

### `tf.contrib.graph_editor.placeholder_name(t=None, scope=None)` {#placeholder_name}

Create placeholder name for the graph editor.

##### Args:


*  <b>`t`</b>: optional tensor on which the placeholder operation's name will be based
    on
*  <b>`scope`</b>: absolute scope with which to prefix the placeholder's name. None
    means that the scope of t is preserved. "" means the root scope.

##### Returns:

  A new placeholder name prefixed by "geph". Note that "geph" stands for
    Graph Editor PlaceHolder. This convention allows to quickly identify the
    placeholder generated by the Graph Editor.

##### Raises:


*  <b>`TypeError`</b>: if t is not None or a tf.Tensor.


- - -

### `tf.contrib.graph_editor.make_placeholder_from_tensor(t, scope=None)` {#make_placeholder_from_tensor}

Create a `tf.placeholder` for the Graph Editor.

Note that the correct graph scope must be set by the calling function.

##### Args:


*  <b>`t`</b>: a `tf.Tensor` whose name will be used to create the placeholder
    (see function placeholder_name).
*  <b>`scope`</b>: absolute scope within which to create the placeholder. None
    means that the scope of `t` is preserved. `""` means the root scope.

##### Returns:

  A newly created `tf.placeholder`.

##### Raises:


*  <b>`TypeError`</b>: if `t` is not `None` or a `tf.Tensor`.


- - -

### `tf.contrib.graph_editor.make_placeholder_from_dtype_and_shape(dtype, shape=None, scope=None)` {#make_placeholder_from_dtype_and_shape}

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



## Module: select

- - -

### `tf.contrib.graph_editor.filter_ts(ops, positive_filter)` {#filter_ts}

Get all the tensors which are input or output of an op in ops.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of `tf.Operation`.
*  <b>`positive_filter`</b>: a function deciding whether to keep a tensor or not.
    If `True`, all the tensors are returned.

##### Returns:

  A list of `tf.Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of `tf.Operation`.


- - -

### `tf.contrib.graph_editor.filter_ts_from_regex(ops, regex)` {#filter_ts_from_regex}

Get all the tensors linked to ops that match the given regex.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of tf.Operation.
*  <b>`regex`</b>: a regular expression matching the tensors' name.
    For example, "^foo(/.*)?:\d+$" will match all the tensors in the "foo"
    scope.

##### Returns:

  A list of tf.Tensor.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of tf.Operation.


- - -

### `tf.contrib.graph_editor.filter_ops(ops, positive_filter)` {#filter_ops}

Get the ops passing the given filter.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of tf.Operation.
*  <b>`positive_filter`</b>: a function deciding where to keep an operation or not.
    If True, all the operations are returned.

##### Returns:

  A list of selected tf.Operation.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of tf.Operation.


- - -

### `tf.contrib.graph_editor.filter_ops_from_regex(ops, regex)` {#filter_ops_from_regex}

Get all the operations that match the given regex.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of `tf.Operation`.
*  <b>`regex`</b>: a regular expression matching the operation's name.
    For example, `"^foo(/.*)?$"` will match all the operations in the "foo"
    scope.

##### Returns:

  A list of `tf.Operation`.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of `tf.Operation`.


- - -

### `tf.contrib.graph_editor.get_name_scope_ops(ops, scope)` {#get_name_scope_ops}

Get all the operations under the given scope path.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of tf.Operation.
*  <b>`scope`</b>: a scope path.

##### Returns:

  A list of tf.Operation.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of tf.Operation.


- - -

### `tf.contrib.graph_editor.check_cios(control_inputs=False, control_outputs=None, control_ios=None)` {#check_cios}

Do various check on control_inputs and control_outputs.

##### Args:


*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of util.ControlOutputs or None. If not None,
    control outputs are enabled.
*  <b>`control_ios`</b>: An instance of util.ControlOutputs or None. If not None, both
    control inputs and control outputs are enabled. This is equivalent to set
    control_inputs to True and control_outputs to the util.ControlOutputs
    instance.

##### Returns:

  A tuple `(control_inputs, control_outputs)` where:
    `control_inputs` is a boolean indicating whether to use control inputs.
    `control_outputs` is an instance of util.ControlOutputs or None

##### Raises:


*  <b>`ValueError`</b>: if control_inputs is an instance of util.ControlOutputs but
    control_outputs is not None
*  <b>`TypeError`</b>: if control_outputs is not None and is not a util.ControlOutputs.


- - -

### `tf.contrib.graph_editor.get_ops_ios(ops, control_inputs=False, control_outputs=None, control_ios=None)` {#get_ops_ios}

Return all the `tf.Operation` which are connected to an op in ops.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of `tf.Operation`.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of `util.ControlOutputs` or `None`. If not
    `None`, control outputs are enabled.
*  <b>`control_ios`</b>: An instance of `util.ControlOutputs` or `None`. If not `None`,
    both control inputs and control outputs are enabled. This is equivalent to
    set `control_inputs` to `True` and `control_outputs` to the
    `util.ControlOutputs` instance.

##### Returns:

  All the `tf.Operation` surrounding the given ops.

##### Raises:


*  <b>`TypeError`</b>: if `ops` cannot be converted to a list of `tf.Operation`.


- - -

### `tf.contrib.graph_editor.compute_boundary_ts(ops)` {#compute_boundary_ts}

Compute the tensors at the boundary of a set of ops.

This function looks at all the tensors connected to the given ops (in/out)
and classify them into three categories:
1) input tensors: tensors whose generating operation is not in ops.
2) output tensors: tensors whose consumer operations are not in ops
3) inside tensors: tensors which are neither input nor output tensors.

Note that a tensor can be both an inside tensor and an output tensor if it is
consumed by operations both outside and inside of `ops`.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of tf.Operation.

##### Returns:

  A tuple `(outside_input_ts, outside_output_ts, inside_ts)` where:
    `outside_input_ts` is a Python list of input tensors;
    `outside_output_ts` is a python list of output tensors;
    `inside_ts` is a python list of inside tensors.
  Since a tensor can be both an inside tensor and an output tensor,
  `outside_output_ts` and `inside_ts` might intersect.

##### Raises:


*  <b>`TypeError`</b>: if ops cannot be converted to a list of tf.Operation.


- - -

### `tf.contrib.graph_editor.get_within_boundary_ops(ops, seed_ops, boundary_ops=(), inclusive=True, control_inputs=False, control_outputs=None, control_ios=None)` {#get_within_boundary_ops}

Return all the `tf.Operation` within the given boundary.

##### Args:


*  <b>`ops`</b>: an object convertible to a list of `tf.Operation`. those ops define the
    set in which to perform the operation (if a `tf.Graph` is given, it
    will be converted to the list of all its operations).
*  <b>`seed_ops`</b>: the operations from which to start expanding.
*  <b>`boundary_ops`</b>: the ops forming the boundary.
*  <b>`inclusive`</b>: if `True`, the result will also include the boundary ops.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of `util.ControlOutputs` or `None`. If not
    `None`, control outputs are enabled.
*  <b>`control_ios`</b>: An instance of `util.ControlOutputs` or `None`. If not
    `None`, both control inputs and control outputs are enabled. This is
    equivalent to set control_inputs to True and control_outputs to
    the `util.ControlOutputs` instance.

##### Returns:

  All the `tf.Operation` surrounding the given ops.

##### Raises:


*  <b>`TypeError`</b>: if `ops` or `seed_ops` cannot be converted to a list of
    `tf.Operation`.
*  <b>`ValueError`</b>: if the boundary is intersecting with the seeds.


- - -

### `tf.contrib.graph_editor.get_forward_walk_ops(seed_ops, inclusive=True, within_ops=None, stop_at_ts=(), control_outputs=None)` {#get_forward_walk_ops}

Do a forward graph walk and return all the visited ops.

##### Args:


*  <b>`seed_ops`</b>: an iterable of operations from which the forward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the consumers of those tensors.
*  <b>`inclusive`</b>: if True the given seed_ops are also part of the resulting set.
*  <b>`within_ops`</b>: an iterable of `tf.Operation` within which the search is
    restricted. If `within_ops` is `None`, the search is performed within
    the whole graph.
*  <b>`stop_at_ts`</b>: an iterable of tensors at which the graph walk stops.
*  <b>`control_outputs`</b>: a `util.ControlOutputs` instance or None.
    If not `None`, it will be used while walking the graph forward.

##### Returns:

  A Python set of all the `tf.Operation` ahead of `seed_ops`.

##### Raises:


*  <b>`TypeError`</b>: if `seed_ops` or `within_ops` cannot be converted to a list of
    `tf.Operation`.


- - -

### `tf.contrib.graph_editor.get_backward_walk_ops(seed_ops, inclusive=True, within_ops=None, stop_at_ts=(), control_inputs=False)` {#get_backward_walk_ops}

Do a backward graph walk and return all the visited ops.

##### Args:


*  <b>`seed_ops`</b>: an iterable of operations from which the backward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the generators of those tensors.
*  <b>`inclusive`</b>: if True the given seed_ops are also part of the resulting set.
*  <b>`within_ops`</b>: an iterable of `tf.Operation` within which the search is
    restricted. If `within_ops` is `None`, the search is performed within
    the whole graph.
*  <b>`stop_at_ts`</b>: an iterable of tensors at which the graph walk stops.
*  <b>`control_inputs`</b>: if True, control inputs will be used while moving backward.

##### Returns:

  A Python set of all the `tf.Operation` behind `seed_ops`.

##### Raises:


*  <b>`TypeError`</b>: if `seed_ops` or `within_ops` cannot be converted to a list of
    `tf.Operation`.


- - -

### `tf.contrib.graph_editor.get_walks_intersection_ops(forward_seed_ops, backward_seed_ops, forward_inclusive=True, backward_inclusive=True, within_ops=None, control_inputs=False, control_outputs=None, control_ios=None)` {#get_walks_intersection_ops}

Return the intersection of a forward and a backward walk.

##### Args:


*  <b>`forward_seed_ops`</b>: an iterable of operations from which the forward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the consumers of those tensors.
*  <b>`backward_seed_ops`</b>: an iterable of operations from which the backward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the generators of those tensors.
*  <b>`forward_inclusive`</b>: if True the given forward_seed_ops are also part of the
    resulting set.
*  <b>`backward_inclusive`</b>: if True the given backward_seed_ops are also part of the
    resulting set.
*  <b>`within_ops`</b>: an iterable of tf.Operation within which the search is
    restricted. If within_ops is None, the search is performed within
    the whole graph.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of util.ControlOutputs or None. If not None,
    control outputs are enabled.
*  <b>`control_ios`</b>: An instance of util.ControlOutputs or None. If not None, both
    control inputs and control outputs are enabled. This is equivalent to set
    control_inputs to True and control_outputs to the util.ControlOutputs
    instance.

##### Returns:

  A Python set of all the tf.Operation in the intersection of a forward and a
    backward walk.

##### Raises:


*  <b>`TypeError`</b>: if `forward_seed_ops` or `backward_seed_ops` or `within_ops`
    cannot be converted to a list of `tf.Operation`.


- - -

### `tf.contrib.graph_editor.get_walks_union_ops(forward_seed_ops, backward_seed_ops, forward_inclusive=True, backward_inclusive=True, within_ops=None, control_inputs=False, control_outputs=None, control_ios=None)` {#get_walks_union_ops}

Return the union of a forward and a backward walk.

##### Args:


*  <b>`forward_seed_ops`</b>: an iterable of operations from which the forward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the consumers of those tensors.
*  <b>`backward_seed_ops`</b>: an iterable of operations from which the backward graph
    walk starts. If a list of tensors is given instead, the seed_ops are set
    to be the generators of those tensors.
*  <b>`forward_inclusive`</b>: if True the given forward_seed_ops are also part of the
    resulting set.
*  <b>`backward_inclusive`</b>: if True the given backward_seed_ops are also part of the
    resulting set.
*  <b>`within_ops`</b>: restrict the search within those operations. If within_ops is
    None, the search is done within the whole graph.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of util.ControlOutputs or None. If not None,
    control outputs are enabled.
*  <b>`control_ios`</b>: An instance of util.ControlOutputs or None. If not None, both
    control inputs and control outputs are enabled. This is equivalent to set
    control_inputs to True and control_outputs to the util.ControlOutputs
    instance.

##### Returns:

  A Python set of all the tf.Operation in the union of a forward and a
    backward walk.

##### Raises:


*  <b>`TypeError`</b>: if forward_seed_ops or backward_seed_ops or within_ops cannot be
    converted to a list of tf.Operation.


- - -

### `tf.contrib.graph_editor.select_ops(*args, **kwargs)` {#select_ops}

Helper to select operations.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    `tf.Operation`. `tf.Tensor` instances are silently ignored.
*  <b>`**kwargs`</b>: 'graph': `tf.Graph` in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if `positive_filter(elem)` is
      `True`. This is optional.
    'restrict_ops_regex': a regular expression is ignored if it doesn't start
      with the substring "(?#ops)".

##### Returns:

  A list of `tf.Operation`.

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a `tf.Graph`
    or if an argument in args is not an (array of) `tf.Operation`
    or an (array of) `tf.Tensor` (silently ignored) or a string
    or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.


- - -

### `tf.contrib.graph_editor.select_ts(*args, **kwargs)` {#select_ts}

Helper to select tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    `tf.Tensor`. `tf.Operation` instances are silently ignored.
*  <b>`**kwargs`</b>: 'graph': `tf.Graph` in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if `positive_filter(elem)` is
      `True`. This is optional.
    'restrict_ts_regex': a regular expression is ignored if it doesn't start
      with the substring "(?#ts)".

##### Returns:

  A list of `tf.Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a `tf.Graph`
    or if an argument in args is not an (array of) `tf.Tensor`
    or an (array of) `tf.Operation` (silently ignored) or a string
    or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.


- - -

### `tf.contrib.graph_editor.select_ops_and_ts(*args, **kwargs)` {#select_ops_and_ts}

Helper to select operations and tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    `tf.Operation` 3) (array of) tf.Tensor. Regular expressions matching
    tensors must start with the comment `"(?#ts)"`, for instance:
    `"(?#ts)^foo/.*"`.
*  <b>`**kwargs`</b>: 'graph': `tf.Graph` in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if `positive_filter(elem)` is
      `True`. This is optional.

##### Returns:

  A tuple `(ops, ts)` where:
    `ops` is a list of `tf.Operation`, and
    `ts` is a list of `tf.Tensor`

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a `tf.Graph`
    or if an argument in args is not an (array of) `tf.Tensor`
    or an (array of) `tf.Operation` or a string or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.



## Module: subgraph

- - -

### `class tf.contrib.graph_editor.SubGraphView` {#SubGraphView}

A subgraph view on an existing `tf.Graph`.

An instance of this class is a subgraph view on an existing `tf.Graph`.
"subgraph" means that it can represent part of the whole `tf.Graph`.
"view" means that it only provides a passive observation and do not to act
on the `tf.Graph`. Note that in this documentation, the term "subgraph" is
often used as substitute to "subgraph view".

A subgraph contains:

* a list of input tensors, accessible via the `inputs` property.
* a list of output tensors, accessible via the `outputs` property.
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
can be omitted. For instance, a subgraph representing an operation with two
inputs can be remapped to only take one input. Note that this does not change
at all the underlying `tf.Graph` (remember, it is a view). It means that
the other input is being ignored, or is being treated as "given".
The analogy with functions can be extended like this: F(x,y) is the original
function. Remapping the inputs from [x, y] to just [x] means that the subgraph
now represent the function F_y(x) (y is "given").

The output tensors can also be remapped. For instance, some output tensor can
be omitted. Other output tensor can be duplicated as well. As mentioned
before, this does not change at all the underlying `tf.Graph`.
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

Here are some useful guarantees about an instance of a SubGraphView:

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
data they observe (in this case, a `tf.Graph`). This is up to the user to
ensure that this doesn't happen. To keep on the safe side, it is recommended
that the life time of subgraph views are kept very short. One way to achieve
this is to use subgraphs within a "with make_sgv(...) as sgv:" Python context.

To alleviate the out-of-sync problem, some functions are granted the right to
modified subgraph in place. This is typically the case of graph manipulation
functions which, given some subgraphs as arguments, can modify the underlying
`tf.Graph`. Since this modification is likely to render the subgraph view
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
does not copy the underlying part of the `tf.Graph`.

##### Returns:

  A new identical instance of the original subgraph view.


- - -

#### `tf.contrib.graph_editor.SubGraphView.__enter__()` {#SubGraphView.__enter__}

Allow Python context to minimize the life time of a subgraph view.

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


*  <b>`inside_ops`</b>: an object convertible to a list of `tf.Operation`. This list
    defines all the operations in the subgraph.
*  <b>`passthrough_ts`</b>: an object convertible to a list of `tf.Tensor`. This list
    define all the "passthrough" tensors. A passthrough tensor is a tensor
    which goes directly from the input of the subgraph to it output, without
    any intermediate operations. All the non passthrough tensors are
    silently ignored.

##### Raises:


*  <b>`TypeError`</b>: if inside_ops cannot be converted to a list of `tf.Operation`
    or if `passthrough_ts` cannot be converted to a list of `tf.Tensor`.


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

A consumer of a subgraph view is a tf.Operation which is a consumer
of one of the output tensors and is not in the subgraph.

##### Returns:

  A list of `tf.Operation` which are the consumers of this subgraph view.


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

The underlying `tf.Graph`.


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

Note that this is only modifying the view: the underlying `tf.Graph` is not
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

### `tf.contrib.graph_editor.make_view(*args, **kwargs)` {#make_view}

Create a SubGraphView from selected operations and passthrough tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    `tf.Operation` 3) (array of) `tf.Tensor`. Those objects will be converted
    into a list of operations and a list of candidate for passthrough tensors.
*  <b>`**kwargs`</b>: keyword graph is used 1) to check that the ops and ts are from
    the correct graph 2) for regular expression query

##### Returns:

  A subgraph view.

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a `tf.Graph`
    or if an argument in args is not an (array of) `tf.Tensor`
    or an (array of) `tf.Operation` or a string or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected.


- - -

### `tf.contrib.graph_editor.make_view_from_scope(scope, graph)` {#make_view_from_scope}

Make a subgraph from a name scope.

##### Args:


*  <b>`scope`</b>: the name of the scope.
*  <b>`graph`</b>: the `tf.Graph`.

##### Returns:

  A subgraph view representing the given scope.



## Module: reroute

- - -

### `tf.contrib.graph_editor.swap_ts(ts0, ts1, can_modify=None, cannot_modify=None)` {#swap_ts}

For each tensor's pair, swap the end of (t0,t1).

B0 B1     B0 B1
|  |    =>  X
A0 A1     A0 A1

##### Args:


*  <b>`ts0`</b>: an object convertible to a list of `tf.Tensor`.
*  <b>`ts1`</b>: an object convertible to a list of `tf.Tensor`.
*  <b>`can_modify`</b>: iterable of operations which can be modified. Any operation
    outside within_ops will be left untouched by this function.
*  <b>`cannot_modify`</b>: iterable of operations which cannot be modified.
    Any operation within cannot_modify will be left untouched by this
    function.

##### Returns:

  The number of individual modifications made by the function.

##### Raises:


*  <b>`TypeError`</b>: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
*  <b>`TypeError`</b>: if can_modify or cannot_modify is not None and cannot be
    converted to a list of tf.Operation.


- - -

### `tf.contrib.graph_editor.reroute_a2b_ts(ts0, ts1, can_modify=None, cannot_modify=None)` {#reroute_a2b_ts}

For each tensor's pair, replace the end of t1 by the end of t0.

B0 B1     B0 B1
|  |    => |/
A0 A1     A0 A1

The end of the tensors in ts1 are left dangling.

##### Args:


*  <b>`ts0`</b>: an object convertible to a list of `tf.Tensor`.
*  <b>`ts1`</b>: an object convertible to a list of `tf.Tensor`.
*  <b>`can_modify`</b>: iterable of operations which can be modified. Any operation
    outside within_ops will be left untouched by this function.
*  <b>`cannot_modify`</b>: iterable of operations which cannot be modified. Any
    operation within cannot_modify will be left untouched by this function.

##### Returns:

  The number of individual modifications made by the function.

##### Raises:


*  <b>`TypeError`</b>: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
*  <b>`TypeError`</b>: if can_modify or cannot_modify is not None and cannot be
    converted to a list of tf.Operation.


- - -

### `tf.contrib.graph_editor.reroute_b2a_ts(ts0, ts1, can_modify=None, cannot_modify=None)` {#reroute_b2a_ts}

For each tensor's pair, replace the end of t0 by the end of t1.

B0 B1     B0 B1
|  |    =>  \|
A0 A1     A0 A1

The end of the tensors in ts0 are left dangling.

##### Args:


*  <b>`ts0`</b>: an object convertible to a list of `tf.Tensor`.
*  <b>`ts1`</b>: an object convertible to a list of `tf.Tensor`.
*  <b>`can_modify`</b>: iterable of operations which can be modified. Any operation
    outside within_ops will be left untouched by this function.
*  <b>`cannot_modify`</b>: iterable of operations which cannot be modified.
    Any operation within cannot_modify will be left untouched by this
    function.

##### Returns:

  The number of individual modifications made by the function.

##### Raises:


*  <b>`TypeError`</b>: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
*  <b>`TypeError`</b>: if can_modify or cannot_modify is not None and cannot be
    converted to a list of tf.Operation.


- - -

### `tf.contrib.graph_editor.swap_inputs(sgv0, sgv1)` {#swap_inputs}

Swap all the inputs of sgv0 and sgv1 (see reroute_inputs).


- - -

### `tf.contrib.graph_editor.reroute_a2b_inputs(sgv0, sgv1)` {#reroute_a2b_inputs}

Re-route all the inputs of sgv0 to sgv1 (see reroute_inputs).


- - -

### `tf.contrib.graph_editor.reroute_b2a_inputs(sgv0, sgv1)` {#reroute_b2a_inputs}

Re-route all the inputs of sgv1 to sgv0 (see reroute_inputs).


- - -

### `tf.contrib.graph_editor.swap_outputs(sgv0, sgv1)` {#swap_outputs}

Swap all the outputs of sgv0 and sgv1 (see _reroute_outputs).


- - -

### `tf.contrib.graph_editor.reroute_a2b_outputs(sgv0, sgv1)` {#reroute_a2b_outputs}

Re-route all the outputs of sgv0 to sgv1 (see _reroute_outputs).


- - -

### `tf.contrib.graph_editor.reroute_b2a_outputs(sgv0, sgv1)` {#reroute_b2a_outputs}

Re-route all the outputs of sgv1 to sgv0 (see _reroute_outputs).


- - -

### `tf.contrib.graph_editor.swap(sgv0, sgv1)` {#swap}

Swap the inputs and outputs of sgv1 to sgv0 (see _reroute).


- - -

### `tf.contrib.graph_editor.reroute_a2b(sgv0, sgv1)` {#reroute_a2b}

Re-route the inputs and outputs of sgv0 to sgv1 (see _reroute).


- - -

### `tf.contrib.graph_editor.reroute_b2a(sgv0, sgv1)` {#reroute_b2a}

Re-route the inputs and outputs of sgv1 to sgv0 (see _reroute).


- - -

### `tf.contrib.graph_editor.remove_control_inputs(op, cops)` {#remove_control_inputs}

Remove the control inputs cops from co.

Warning: this function is directly manipulating the internals of the
`tf.Graph`.

##### Args:


*  <b>`op`</b>: a `tf.Operation` from which to remove the control inputs.
*  <b>`cops`</b>: an object convertible to a list of `tf.Operation`.

##### Raises:


*  <b>`TypeError`</b>: if op is not a `tf.Operation`.
*  <b>`ValueError`</b>: if any cop in cops is not a control input of op.


- - -

### `tf.contrib.graph_editor.add_control_inputs(op, cops)` {#add_control_inputs}

Add the control inputs cops to co.

Warning: this function is directly manipulating the internals of the tf.Graph.

##### Args:


*  <b>`op`</b>: a tf.Operation to which the control inputs are added.
*  <b>`cops`</b>: an object convertible to a list of `tf.Operation`.

##### Raises:


*  <b>`TypeError`</b>: if op is not a tf.Operation
*  <b>`ValueError`</b>: if any cop in cops is already a control input of op.



## Module: edit

- - -

### `tf.contrib.graph_editor.detach_control_inputs(sgv)` {#detach_control_inputs}

Detach all the external control inputs of the subgraph sgv.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.detach_control_outputs(sgv, control_outputs)` {#detach_control_outputs}

Detach all the external control outputs of the subgraph sgv.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
*  <b>`control_outputs`</b>: a util.ControlOutputs instance.


- - -

### `tf.contrib.graph_editor.detach_inputs(sgv, control_inputs=False)` {#detach_inputs}

Detach the inputs of a subgraph view.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
    Note that sgv is modified in place.
*  <b>`control_inputs`</b>: if True control_inputs are also detached.

##### Returns:

  A tuple `(sgv, input_placeholders)` where
    `sgv` is a new subgraph view of the detached subgraph;
    `input_placeholders` is a list of the created input placeholders.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.detach_outputs(sgv, control_outputs=None)` {#detach_outputs}

Detach the output of a subgraph view.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
    Note that sgv is modified in place.
*  <b>`control_outputs`</b>: a util.ControlOutputs instance or None. If not None the
    control outputs are also detached.

##### Returns:

  A tuple `(sgv, output_placeholders)` where
    `sgv` is a new subgraph view of the detached subgraph;
    `output_placeholders` is a list of the created output placeholders.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.detach(sgv, control_inputs=False, control_outputs=None, control_ios=None)` {#detach}

Detach both the inputs and the outputs of a subgraph view.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be detached. This argument is converted to a
    subgraph using the same rules as the function subgraph.make_view.
    Note that sgv is modified in place.
*  <b>`control_inputs`</b>: A boolean indicating whether control inputs are enabled.
*  <b>`control_outputs`</b>: An instance of util.ControlOutputs or None. If not None,
    control outputs are enabled.
*  <b>`control_ios`</b>: An instance of util.ControlOutputs or None. If not None, both
    control inputs and control outputs are enabled. This is equivalent to set
    control_inputs to True and control_outputs to the util.ControlOutputs
    instance.

##### Returns:

  A tuple `(sgv, detached_inputs, detached_outputs)` where:
  `sgv` is a new subgraph view of the detached subgraph;
  `detach_inputs` is a list of the created input placeholders;
  `detach_outputs` is a list of the created output placeholders.

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
    Note that sgv0 is modified in place.
*  <b>`sgv1`</b>: the second subgraph to have its outputs swapped. This argument is
    converted to a subgraph using the same rules as the function
    subgraph.make_view.
    Note that sgv1 is modified in place.
*  <b>`disconnect_first`</b>: if True the current outputs of sgv0 are disconnected.

##### Returns:

  A tuple `(sgv0, sgv1)` of the now connected subgraphs.

##### Raises:


*  <b>`StandardError`</b>: if sgv0 or sgv1 cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.bypass(sgv)` {#bypass}

Bypass the given subgraph by connecting its inputs to its outputs.

##### Args:


*  <b>`sgv`</b>: the subgraph view to be bypassed. This argument is converted to a
    subgraph using the same rules than the function subgraph.make_view.
    Note that sgv is modified in place.

##### Returns:

  A tuple `(sgv, detached_inputs)` where:
    `sgv` is a new subgraph view of the bypassed subgraph;
    `detached_inputs` is a list of the created input placeholders.

##### Raises:


*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.



## Module: transform

- - -

### `tf.contrib.graph_editor.replace_t_with_placeholder_handler(info, t)` {#replace_t_with_placeholder_handler}

Transform a tensor into a placeholder tensor.

This handler is typically used to transform a subgraph input tensor into a
placeholder.

##### Args:


*  <b>`info`</b>: Transform._Info instance.
*  <b>`t`</b>: tensor whose input must be transformed into a place holder.

##### Returns:

  The tensor generated by the newly created place holder.


- - -

### `tf.contrib.graph_editor.keep_t_if_possible_handler(info, t)` {#keep_t_if_possible_handler}

Transform a tensor into itself (identity) if possible.

This handler transform a tensor into itself if the source and destination
graph are the same. Otherwise it will create a placeholder.
This handler is typically used to transform a hidden input tensors.

##### Args:


*  <b>`info`</b>: Transform._Info instance.
*  <b>`t`</b>: tensor whose input must be transformed into a place holder.

##### Returns:

  The tensor generated by the newly created place holder.


- - -

### `tf.contrib.graph_editor.assign_renamed_collections_handler(info, elem, elem_)` {#assign_renamed_collections_handler}

Add the transformed elem to the (renamed) collections of elem.

##### Args:


*  <b>`info`</b>: Transform._Info instance.
*  <b>`elem`</b>: the original element (`tf.Tensor` or `tf.Operation`)
*  <b>`elem_`</b>: the transformed element


- - -

### `tf.contrib.graph_editor.transform_op_if_inside_handler(info, op, keep_if_possible=True)` {#transform_op_if_inside_handler}

Transform an optional op only if it is inside the subgraph.

This handler is typically use to handle original op: it is fine to keep them
if they are inside the subgraph, otherwise they are just ignored.

##### Args:


*  <b>`info`</b>: Transform._Info instance.
*  <b>`op`</b>: the optional op to transform (or ignore).
*  <b>`keep_if_possible`</b>: re-attach to the original op if possible, that is,
    if the source graph and the destination graph are the same.

##### Returns:

  The transformed op or None.


- - -

### `tf.contrib.graph_editor.copy_op_handler(info, op, copy_shape=True)` {#copy_op_handler}

Copy a `tf.Operation`.

##### Args:


*  <b>`info`</b>: Transform._Info instance.
*  <b>`op`</b>: the `tf.Operation` to be copied.
*  <b>`copy_shape`</b>: also copy the shape of the tensor

##### Returns:

  A copy of op.


- - -

### `tf.contrib.graph_editor.transform_op_in_place(info, op, detach_outputs=False)` {#transform_op_in_place}

Transform a op in-place - experimental!

Transform an operation in place. It reconnects the inputs if they have been
modified. if detach_outputs is True, the outputs of op are also detached.

##### Args:


*  <b>`info`</b>: Transform._Info instance.
*  <b>`op`</b>: the op to transform in place.
*  <b>`detach_outputs`</b>: if True, the outputs of op are detached, ready for the user
    to add more operation.

##### Returns:

  The transformed op.


- - -

### `class tf.contrib.graph_editor.Transformer` {#Transformer}

Transform a subgraph into another one.

By default, the constructor create a transform which copy a subgraph and
replaces inputs with placeholders. This behavior can be modified by changing
the handlers.
- - -

#### `tf.contrib.graph_editor.Transformer.__call__(sgv, dst_graph, dst_scope, src_scope='', reuse_dst_scope=False)` {#Transformer.__call__}

Execute the transformation.

##### Args:


*  <b>`sgv`</b>: the source subgraph-view.
*  <b>`dst_graph`</b>: the destination graph.
*  <b>`dst_scope`</b>: the destination scope.
*  <b>`src_scope`</b>: the source scope, which specify the path from which the
    relative path of the transformed nodes are computed. For instance, if
    src_scope is a/ and dst_scoped is b/, then the node a/x/y will have a
    relative path of x/y and will be transformed into b/x/y.
*  <b>`reuse_dst_scope`</b>: if True the dst_scope is re-used if it already exists.
    Otherwise, the scope is given a unique name based on the one given
    by appending an underscore followed by a digit (default).

##### Returns:

  A tuple `(sgv, info)` where:
    `sgv` is the transformed subgraph view;
    `info` is an instance of Transformer.ResultInfo containing
    information about the transform, including mapping between
    original and transformed tensors and operations.

##### Raises:


*  <b>`ValueError`</b>: if the arguments are invalid.


- - -

#### `tf.contrib.graph_editor.Transformer.__init__()` {#Transformer.__init__}

Transformer constructor.

The following members can be modified:
transform_op_handler: handle the transformation of a `tf.Operation`.
  This handler defaults to a simple copy.
assign_collections_handler: handle the assignment of collections.
  This handler defaults to assigning new collections created under the
  given name-scope.
transform_external_input_handler: handle the transform of the inputs to
  the given subgraph. This handler defaults to creating placeholders
  instead of the ops just before the input tensors of the subgraph.
transform_external_hidden_input_handler: handle the transform of the
  hidden inputs of the subgraph, that is, the inputs which are not listed
  in sgv.inputs. This handler defaults to a transform which keep the same
  input if the source and destination graphs are the same, otherwise
  use placeholders.
transform_original_op_handler: handle the transform of original_op. This
  handler defaults to transforming original_op only if they are in the
  subgraph, otherwise they are ignored.


- - -

#### `tf.contrib.graph_editor.Transformer.new_name(name)` {#Transformer.new_name}

Compute a destination name from a source name.

##### Args:


*  <b>`name`</b>: the name to be "transformed".

##### Returns:

  The transformed name.

##### Raises:


*  <b>`ValueError`</b>: if the source scope is used (that is, not an empty string)
    and the source name does not belong to the source scope.



- - -

### `tf.contrib.graph_editor.copy(sgv, dst_graph=None, dst_scope='', src_scope='', reuse_dst_scope=False)` {#copy}

Copy a subgraph.

##### Args:


*  <b>`sgv`</b>: the source subgraph-view. This argument is converted to a subgraph
    using the same rules than the function subgraph.make_view.
*  <b>`dst_graph`</b>: the destination graph.
*  <b>`dst_scope`</b>: the destination scope.
*  <b>`src_scope`</b>: the source scope.
*  <b>`reuse_dst_scope`</b>: if True the dst_scope is re-used if it already exists.
    Otherwise, the scope is given a unique name based on the one given
    by appending an underscore followed by a digit (default).

##### Returns:

  A tuple `(sgv, info)` where:
    `sgv` is the transformed subgraph view;
    `info` is an instance of Transformer.ResultInfo containing
    information about the transform, including mapping between
    original and transformed tensors and operations.

##### Raises:


*  <b>`TypeError`</b>: if `dst_graph` is not a `tf.Graph`.
*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.copy_with_input_replacements(sgv, replacement_ts, dst_graph=None, dst_scope='', src_scope='', reuse_dst_scope=False)` {#copy_with_input_replacements}

Copy a subgraph, replacing some of its inputs.

Note a replacement only happens if the tensor to be replaced
is an input of the given subgraph. The inputs of a subgraph can
be queried using sgv.inputs.

##### Args:


*  <b>`sgv`</b>: the source subgraph-view. This argument is converted to a subgraph
    using the same rules as the function subgraph.make_view.
*  <b>`replacement_ts`</b>: dictionary mapping from original tensors to the
    replaced one.
*  <b>`dst_graph`</b>: the destination graph.
*  <b>`dst_scope`</b>: the destination scope.
*  <b>`src_scope`</b>: the source scope.
*  <b>`reuse_dst_scope`</b>: if True the dst_scope is re-used if it already exists.
    Otherwise, the scope is given a unique name based on the one given
    by appending an underscore followed by a digit (default).

##### Returns:

  A tuple `(sgv, info)` where:
    `sgv` is the transformed subgraph view;
    `info` is an instance of Transformer.ResultInfo containing
    information about the transform, including mapping between
    original and transformed tensors and operations.

##### Raises:


*  <b>`TypeError`</b>: if dst_graph is not a tf.Graph.
*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules as the function subgraph.make_view.


- - -

### `tf.contrib.graph_editor.graph_replace(target_ts, replacement_ts, dst_scope='', src_scope='', reuse_dst_scope=False)` {#graph_replace}

Create a new graph which compute the targets from the replaced Tensors.

##### Args:


*  <b>`target_ts`</b>: a single tf.Tensor or an iterable of tf.Tensor.
*  <b>`replacement_ts`</b>: dictionary mapping from original tensors to replaced tensors
*  <b>`dst_scope`</b>: the destination scope.
*  <b>`src_scope`</b>: the source scope.
*  <b>`reuse_dst_scope`</b>: if True the dst_scope is re-used if it already exists.
    Otherwise, the scope is given a unique name based on the one given
    by appending an underscore followed by a digit (default).

##### Returns:

  A single tf.Tensor or a list of target tf.Tensor, depending on
  the type of the input argument `target_ts`.
  The returned tensors are recomputed using the tensors from replacement_ts.

##### Raises:


*  <b>`ValueError`</b>: if the targets are not connected to replacement_ts.



## Module: match

- - -

### `tf.contrib.graph_editor.op_type(op_types, op=None)` {#op_type}

Check if an op is of the given type.

##### Args:


*  <b>`op_types`</b>: tuple of strings containing the types to check against.
    For instance: ("Add", "Const")
*  <b>`op`</b>: the operation to check (or None).

##### Returns:

  if op is not None, return True if the op is of the correct type.
  if op is None, return a lambda function which does the type checking.


- - -

### `class tf.contrib.graph_editor.OpMatcher` {#OpMatcher}

Graph match class.
- - -

#### `tf.contrib.graph_editor.OpMatcher.__call__(op)` {#OpMatcher.__call__}

Evaluate if the op matches or not.


- - -

#### `tf.contrib.graph_editor.OpMatcher.__init__(positive_filter)` {#OpMatcher.__init__}

Graph match constructor.


- - -

#### `tf.contrib.graph_editor.OpMatcher.control_input_ops(*args)` {#OpMatcher.control_input_ops}

Add input matches.


- - -

#### `tf.contrib.graph_editor.OpMatcher.input_ops(*args)` {#OpMatcher.input_ops}

Add input matches.


- - -

#### `tf.contrib.graph_editor.OpMatcher.output_ops(*args)` {#OpMatcher.output_ops}

Add output matches.




## Useful aliases

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

### `tf.contrib.graph_editor.sgv(*args, **kwargs)` {#sgv}

Create a SubGraphView from selected operations and passthrough tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    `tf.Operation` 3) (array of) `tf.Tensor`. Those objects will be converted
    into a list of operations and a list of candidate for passthrough tensors.
*  <b>`**kwargs`</b>: keyword graph is used 1) to check that the ops and ts are from
    the correct graph 2) for regular expression query

##### Returns:

  A subgraph view.

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a `tf.Graph`
    or if an argument in args is not an (array of) `tf.Tensor`
    or an (array of) `tf.Operation` or a string or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected.


- - -

### `tf.contrib.graph_editor.sgv_scope(scope, graph)` {#sgv_scope}

Make a subgraph from a name scope.

##### Args:


*  <b>`scope`</b>: the name of the scope.
*  <b>`graph`</b>: the `tf.Graph`.

##### Returns:

  A subgraph view representing the given scope.


- - -

### `tf.contrib.graph_editor.ts(*args, **kwargs)` {#ts}

Helper to select tensors.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    `tf.Tensor`. `tf.Operation` instances are silently ignored.
*  <b>`**kwargs`</b>: 'graph': `tf.Graph` in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if `positive_filter(elem)` is
      `True`. This is optional.
    'restrict_ts_regex': a regular expression is ignored if it doesn't start
      with the substring "(?#ts)".

##### Returns:

  A list of `tf.Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a `tf.Graph`
    or if an argument in args is not an (array of) `tf.Tensor`
    or an (array of) `tf.Operation` (silently ignored) or a string
    or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.


- - -

### `tf.contrib.graph_editor.ops(*args, **kwargs)` {#ops}

Helper to select operations.

##### Args:


*  <b>`*args`</b>: list of 1) regular expressions (compiled or not) or  2) (array of)
    `tf.Operation`. `tf.Tensor` instances are silently ignored.
*  <b>`**kwargs`</b>: 'graph': `tf.Graph` in which to perform the regex query.This is
    required when using regex.
    'positive_filter': an elem if selected only if `positive_filter(elem)` is
      `True`. This is optional.
    'restrict_ops_regex': a regular expression is ignored if it doesn't start
      with the substring "(?#ops)".

##### Returns:

  A list of `tf.Operation`.

##### Raises:


*  <b>`TypeError`</b>: if the optional keyword argument graph is not a `tf.Graph`
    or if an argument in args is not an (array of) `tf.Operation`
    or an (array of) `tf.Tensor` (silently ignored) or a string
    or a regular expression.
*  <b>`ValueError`</b>: if one of the keyword arguments is unexpected or if a regular
    expression is used without passing a graph as a keyword argument.


- - -

### `class tf.contrib.graph_editor.matcher` {#matcher}

Graph match class.
- - -

#### `tf.contrib.graph_editor.matcher.__call__(op)` {#matcher.__call__}

Evaluate if the op matches or not.


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



