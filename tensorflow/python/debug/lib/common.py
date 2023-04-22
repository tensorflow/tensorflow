# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Common values and methods for TensorFlow Debugger."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json

GRPC_URL_PREFIX = "grpc://"

# A key for a Session.run() call.
RunKey = collections.namedtuple("RunKey", ["feed_names", "fetch_names"])


def get_graph_element_name(elem):
  """Obtain the name or string representation of a graph element.

  If the graph element has the attribute "name", return name. Otherwise, return
  a __str__ representation of the graph element. Certain graph elements, such as
  `SparseTensor`s, do not have the attribute "name".

  Args:
    elem: The graph element in question.

  Returns:
    If the attribute 'name' is available, return the name. Otherwise, return
    str(fetch).
  """

  return elem.name if hasattr(elem, "name") else str(elem)


def get_flattened_names(feeds_or_fetches):
  """Get a flattened list of the names in run() call feeds or fetches.

  Args:
    feeds_or_fetches: Feeds or fetches of the `Session.run()` call. It maybe
      a Tensor, an Operation or a Variable. It may also be nested lists, tuples
      or dicts. See doc of `Session.run()` for more details.

  Returns:
    (list of str) A flattened list of fetch names from `feeds_or_fetches`.
  """

  lines = []
  if isinstance(feeds_or_fetches, (list, tuple)):
    for item in feeds_or_fetches:
      lines.extend(get_flattened_names(item))
  elif isinstance(feeds_or_fetches, dict):
    for key in feeds_or_fetches:
      lines.extend(get_flattened_names(feeds_or_fetches[key]))
  else:
    # This ought to be a Tensor, an Operation or a Variable, for which the name
    # attribute should be available. (Bottom-out condition of the recursion.)
    lines.append(get_graph_element_name(feeds_or_fetches))

  return lines


def get_run_key(feed_dict, fetches):
  """Summarize the names of feeds and fetches as a RunKey JSON string.

  Args:
    feed_dict: The feed_dict given to the `Session.run()` call.
    fetches: The fetches from the `Session.run()` call.

  Returns:
    A JSON Array consisting of two items. They first items is a flattened
    Array of the names of the feeds. The second item is a flattened Array of
    the names of the fetches.
  """
  return json.dumps(RunKey(get_flattened_names(feed_dict),
                           get_flattened_names(fetches)))
