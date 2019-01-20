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
"""Loader functionality for SavedModel with hermetic, language-neutral exports.

Load and restore capability for a SavedModel, which may include multiple meta
graph defs. Each SavedModel is associated with a single checkpoint. Each meta
graph def is saved with one or more tags, which are used to identify the exact
meta graph def to load.

The `load` operation requires the session in which to restore the graph
definition and variables, the tags used to identify the meta graph def to
load and the location of the SavedModel.

Upon a load, the subset of variables and assets supplied as part of the specific
meta graph def, will be restored into the supplied session. The values of the
variables though will correspond to the saved values from the first meta graph
added to the SavedModel using `add_meta_graph_and_variables(...)` in
`builder.py`.

Typical usage:

```python
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       ["foo-tag"],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...

with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph(["bar-tag", "baz-tag"],
                         assets_collection=bar_baz_assets)
...

builder.save()

...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, ["foo-tag"], export_dir)
  ...

```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.saved_model.loader_impl import load
from tensorflow.python.saved_model.loader_impl import maybe_saved_model_directory
# pylint: enable=unused-import
