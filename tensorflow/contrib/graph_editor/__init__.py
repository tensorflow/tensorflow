# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow Graph Editor.

The TensorFlow Graph Editor library allows for modification of an existing
tf.Graph instance in-place.

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
which to operate must always be explicitely given. This is the reason why
*graph=tf.get_default_graph()* is used in the code snippets above.

## Modules overview

* util: utility functions.
* select: various selection methods of TensorFlow tensors and operations.
* match: TensorFlow graph matching. Think of this as regular expressions for
  graphs (but not quite yet).
* reroute: various ways of rerouting tensors to different consuming ops like
  *swap* or *reroute_a2b*.
* subgraph: the SubGraphView class, which enables subgraph manipulations in a
  TensorFlow tf.Graph.
* edit: various editing functions operating on subgraphs like *detach*,
  *connect* or *bypass*.
* transform: the Transformer class, which enables transforming
  (or simply copying) a subgraph into another one.

## Module: util

@@make_list_of_op
@@get_tensors
@@make_list_of_t
@@get_generating_ops
@@get_consuming_ops
@@ControlOutputs
@@placeholder_name
@@make_placeholder_from_tensor
@@make_placeholder_from_dtype_and_shape

## Module: select

@@filter_ts
@@filter_ts_from_regex
@@filter_ops
@@filter_ops_from_regex
@@get_name_scope_ops
@@check_cios
@@get_ops_ios
@@compute_boundary_ts
@@get_within_boundary_ops
@@get_forward_walk_ops
@@get_backward_walk_ops
@@get_walks_intersection_ops
@@get_walks_union_ops
@@select_ops
@@select_ts
@@select_ops_and_ts

## Module: subgraph

@@SubGraphView
@@make_view
@@make_view_from_scope

## Module: reroute

@@swap_ts
@@reroute_a2b_ts
@@reroute_b2a_ts
@@swap_inputs
@@reroute_a2b_inputs
@@reroute_b2a_inputs
@@swap_outputs
@@reroute_a2b_outputs
@@reroute_b2a_outputs
@@swap
@@reroute_a2b
@@reroute_b2a
@@remove_control_inputs
@@add_control_inputs

## Module: edit

@@detach_control_inputs
@@detach_control_outputs
@@detach_inputs
@@detach_outputs
@@detach
@@connect
@@bypass

## Module: transform

@@replace_t_with_placeholder_handler
@@keep_t_if_possible_handler
@@assign_renamed_collections_handler
@@transform_op_if_inside_handler
@@copy_op_handler
@@transform_op_in_place
@@Transformer
@@copy
@@copy_with_input_replacements
@@graph_replace

## Module: match

@@op_type
@@OpMatcher

## Useful aliases

@@ph
@@sgv
@@sgv_scope
@@ts
@@ops
@@matcher
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.graph_editor import edit
from tensorflow.contrib.graph_editor import match
from tensorflow.contrib.graph_editor import reroute
from tensorflow.contrib.graph_editor import select
from tensorflow.contrib.graph_editor import subgraph
from tensorflow.contrib.graph_editor import transform
from tensorflow.contrib.graph_editor import util

# pylint: disable=wildcard-import
from tensorflow.contrib.graph_editor.edit import *
from tensorflow.contrib.graph_editor.match import *
from tensorflow.contrib.graph_editor.reroute import *
from tensorflow.contrib.graph_editor.select import *
from tensorflow.contrib.graph_editor.subgraph import *
from tensorflow.contrib.graph_editor.transform import *
from tensorflow.contrib.graph_editor.util import *
# pylint: enable=wildcard-import

# some useful aliases
ph = util.make_placeholder_from_dtype_and_shape
sgv = subgraph.make_view
sgv_scope = subgraph.make_view_from_scope
ts = select.select_ts
ops = select.select_ops
matcher = match.OpMatcher
