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

*   `tf.contrib.graph_editor.make_list_of_op`
*   `tf.contrib.graph_editor.get_tensors`
*   `tf.contrib.graph_editor.make_list_of_t`
*   `tf.contrib.graph_editor.get_generating_ops`
*   `tf.contrib.graph_editor.get_consuming_ops`
*   `tf.contrib.graph_editor.ControlOutputs`
*   `tf.contrib.graph_editor.placeholder_name`
*   `tf.contrib.graph_editor.make_placeholder_from_tensor`
*   `tf.contrib.graph_editor.make_placeholder_from_dtype_and_shape`

## Module: select

*   `tf.contrib.graph_editor.filter_ts`
*   `tf.contrib.graph_editor.filter_ts_from_regex`
*   `tf.contrib.graph_editor.filter_ops`
*   `tf.contrib.graph_editor.filter_ops_from_regex`
*   `tf.contrib.graph_editor.get_name_scope_ops`
*   `tf.contrib.graph_editor.check_cios`
*   `tf.contrib.graph_editor.get_ops_ios`
*   `tf.contrib.graph_editor.compute_boundary_ts`
*   `tf.contrib.graph_editor.get_within_boundary_ops`
*   `tf.contrib.graph_editor.get_forward_walk_ops`
*   `tf.contrib.graph_editor.get_backward_walk_ops`
*   `tf.contrib.graph_editor.get_walks_intersection_ops`
*   `tf.contrib.graph_editor.get_walks_union_ops`
*   `tf.contrib.graph_editor.select_ops`
*   `tf.contrib.graph_editor.select_ts`
*   `tf.contrib.graph_editor.select_ops_and_ts`

## Module: subgraph

*   `tf.contrib.graph_editor.SubGraphView`
*   `tf.contrib.graph_editor.make_view`
*   `tf.contrib.graph_editor.make_view_from_scope`

## Module: reroute

*   `tf.contrib.graph_editor.swap_ts`
*   `tf.contrib.graph_editor.reroute_ts`
*   `tf.contrib.graph_editor.swap_inputs`
*   `tf.contrib.graph_editor.reroute_inputs`
*   `tf.contrib.graph_editor.swap_outputs`
*   `tf.contrib.graph_editor.reroute_outputs`
*   `tf.contrib.graph_editor.swap_ios`
*   `tf.contrib.graph_editor.reroute_ios`
*   `tf.contrib.graph_editor.remove_control_inputs`
*   `tf.contrib.graph_editor.add_control_inputs`

## Module: edit

*   `tf.contrib.graph_editor.detach_control_inputs`
*   `tf.contrib.graph_editor.detach_control_outputs`
*   `tf.contrib.graph_editor.detach_inputs`
*   `tf.contrib.graph_editor.detach_outputs`
*   `tf.contrib.graph_editor.detach`
*   `tf.contrib.graph_editor.connect`
*   `tf.contrib.graph_editor.bypass`

## Module: transform

*   `tf.contrib.graph_editor.replace_t_with_placeholder_handler`
*   `tf.contrib.graph_editor.keep_t_if_possible_handler`
*   `tf.contrib.graph_editor.assign_renamed_collections_handler`
*   `tf.contrib.graph_editor.transform_op_if_inside_handler`
*   `tf.contrib.graph_editor.copy_op_handler`
*   `tf.contrib.graph_editor.Transformer`
*   `tf.contrib.graph_editor.copy`
*   `tf.contrib.graph_editor.copy_with_input_replacements`
*   `tf.contrib.graph_editor.graph_replace`

## Useful aliases

*   `tf.contrib.graph_editor.ph`
*   `tf.contrib.graph_editor.sgv`
*   `tf.contrib.graph_editor.sgv_scope`
