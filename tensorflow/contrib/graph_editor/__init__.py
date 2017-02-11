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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
# pylint: disable=g-bad-import-order
from tensorflow.contrib.graph_editor import subgraph as _subgraph
from tensorflow.contrib.graph_editor import util as _util
# pylint: enable=g-bad-import-order
ph = _util.make_placeholder_from_dtype_and_shape
sgv = _subgraph.make_view
sgv_scope = _subgraph.make_view_from_scope

del absolute_import
del division
del print_function
