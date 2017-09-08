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
#==============================================================================
"""Contains utility functions used by summary ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import re

from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging


def collect(val, collections, default_collections):
  """Adds keys to a collection.

  Args:
    val: The value to add per each key.
    collections: A collection of keys to add.
    default_collections: Used if collections is None.
  """
  if collections is None:
    collections = default_collections
  for key in collections:
    ops.add_to_collection(key, val)


_INVALID_TAG_CHARACTERS = re.compile(r'[^-/\w\.]')


def clean_tag(name):
  """Cleans a tag. Removes illegal characters for instance.

  Args:
    name: The original tag name to be processed.

  Returns:
    The cleaned tag name.
  """
  # In the past, the first argument to summary ops was a tag, which allowed
  # arbitrary characters. Now we are changing the first argument to be the node
  # name. This has a number of advantages (users of summary ops now can
  # take advantage of the tf name scope system) but risks breaking existing
  # usage, because a much smaller set of characters are allowed in node names.
  # This function replaces all illegal characters with _s, and logs a warning.
  # It also strips leading slashes from the name.
  if name is not None:
    new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
    new_name = new_name.lstrip('/')  # Remove leading slashes
    if new_name != name:
      tf_logging.info('Summary name %s is illegal; using %s instead.' %
                      (name, new_name))
      name = new_name
  return name


@contextlib.contextmanager
def summary_scope(name, family=None, default_name=None, values=None):
  """Enters a scope used for the summary and yields both the name and tag.

  To ensure that the summary tag name is always unique, we create a name scope
  based on `name` and use the full scope name in the tag.

  If `family` is set, then the tag name will be '<family>/<scope_name>', where
  `scope_name` is `<outer_scope>/<family>/<name>`. This ensures that `family`
  is always the prefix of the tag (and unmodified), while ensuring the scope
  respects the outer scope from this summary was created.

  Args:
    name: A name for the generated summary node.
    family: Optional; if provided, used as the prefix of the summary tag name.
    default_name: Optional; if provided, used as default name of the summary.
    values: Optional; passed as `values` parameter to name_scope.

  Yields:
    A tuple `(tag, scope)`, both of which are unique and should be used for the
    tag and the scope for the summary to output.
  """
  name = clean_tag(name)
  family = clean_tag(family)
  # Use family name in the scope to ensure uniqueness of scope/tag.
  scope_base_name = name if family is None else '{}/{}'.format(family, name)
  with ops.name_scope(scope_base_name, default_name, values=values) as scope:
    if family is None:
      tag = scope.rstrip('/')
    else:
      # Prefix our scope with family again so it displays in the right tab.
      tag = '{}/{}'.format(family, scope.rstrip('/'))
      # Note: tag is not 100% unique if the user explicitly enters a scope with
      # the same name as family, then later enter it again before summaries.
      # This is very contrived though, and we opt here to let it be a runtime
      # exception if tags do indeed collide.
    yield (tag, scope)
