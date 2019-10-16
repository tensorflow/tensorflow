# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Functions related to Python memory management."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# TODO(b/115366440): Delete this function when a custom OrderedDict is added
def dismantle_ordered_dict(ordered_dict):
  """Remove reference cycle in OrderedDict `ordered_dict`.

  Helpful for making sure the garbage collector doesn't need to run after
  using an OrderedDict.

  Args:
    ordered_dict: A `OrderedDict` object to destroy. This object is unusable
      after this function runs.
  """
  # OrderedDict, makes a simple reference loop
  # and hides it in an __attribute in some Python versions. We don't need to
  # throw an error if we can't find it, but if we do find it we can break the
  # loop to avoid creating work for the garbage collector.
  problematic_cycle = ordered_dict.__dict__.get("_OrderedDict__root", None)  # pylint: disable=protected-access
  if problematic_cycle:
    try:
      del problematic_cycle[0][:]
    except TypeError:
      # This is probably not one of the problematic Python versions. Continue
      # with the rest of our cleanup.
      pass
