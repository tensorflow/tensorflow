# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

r"""System for specifying garbage collection (GC) of path based data (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

This framework allows for GC of data specified by path names, for example files
on disk.  gc.Path objects each represent a single item stored at a path and may
be a base directory,
  /tmp/exports/0/...
  /tmp/exports/1/...
  ...
or a fully qualified file,
  /tmp/train-1.ckpt
  /tmp/train-2.ckpt
  ...

A gc filter function takes and returns a list of gc.Path items.  Filter
functions are responsible for selecting Path items for preservation or deletion.
Note that functions should always return a sorted list.

For example,
  base_dir = "/tmp"
  # Create the directories.
  for e in xrange(10):
    os.mkdir("%s/%d" % (base_dir, e), 0o755)

  # Create a simple parser that pulls the export_version from the directory.
  path_regex = "^" + re.escape(base_dir) + "/(\\d+)$"
  def parser(path):
    match = re.match(path_regex, path.path)
    if not match:
      return None
    return path._replace(export_version=int(match.group(1)))

  path_list = gc.get_paths("/tmp", parser)  # contains all ten Paths

  every_fifth = gc.mod_export_version(5)
  print(every_fifth(path_list))  # shows ["/tmp/0", "/tmp/5"]

  largest_three = gc.largest_export_versions(3)
  print(largest_three(all_paths))  # shows ["/tmp/7", "/tmp/8", "/tmp/9"]

  both = gc.union(every_fifth, largest_three)
  print(both(all_paths))  # shows ["/tmp/0", "/tmp/5",
                          #        "/tmp/7", "/tmp/8", "/tmp/9"]
  # Delete everything not in 'both'.
  to_delete = gc.negation(both)
  for p in to_delete(all_paths):
    gfile.rmtree(p.path)  # deletes:  "/tmp/1", "/tmp/2",
                                     # "/tmp/3", "/tmp/4", "/tmp/6",
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import heapq
import math
import os

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated

Path = collections.namedtuple('Path', 'path export_version')


@deprecated(None, 'Please implement your own file management or use Saver.')
def largest_export_versions(n):
  """Creates a filter that keeps the largest n export versions.

  Args:
    n: number of versions to keep.

  Returns:
    A filter function that keeps the n largest paths.
  """
  def keep(paths):
    heap = []
    for idx, path in enumerate(paths):
      if path.export_version is not None:
        heapq.heappush(heap, (path.export_version, idx))
    keepers = [paths[i] for _, i in heapq.nlargest(n, heap)]
    return sorted(keepers)

  return keep


@deprecated(None, 'Please implement your own file management or use Saver.')
def one_of_every_n_export_versions(n):
  """Creates a filter that keeps one of every n export versions.

  Args:
    n: interval size.

  Returns:
    A filter function that keeps exactly one path from each interval
    [0, n], (n, 2n], (2n, 3n], etc...  If more than one path exists in an
    interval the largest is kept.
  """
  def keep(paths):
    """A filter function that keeps exactly one out of every n paths."""

    keeper_map = {}  # map from interval to largest path seen in that interval
    for p in paths:
      if p.export_version is None:
        # Skip missing export_versions.
        continue
      # Find the interval (with a special case to map export_version = 0 to
      # interval 0.
      interval = math.floor(
          (p.export_version - 1) / n) if p.export_version else 0
      existing = keeper_map.get(interval, None)
      if (not existing) or (existing.export_version < p.export_version):
        keeper_map[interval] = p
    return sorted(keeper_map.values())

  return keep


@deprecated(None, 'Please implement your own file management or use Saver.')
def mod_export_version(n):
  """Creates a filter that keeps every export that is a multiple of n.

  Args:
    n: step size.

  Returns:
    A filter function that keeps paths where export_version % n == 0.
  """
  def keep(paths):
    keepers = []
    for p in paths:
      if p.export_version % n == 0:
        keepers.append(p)
    return sorted(keepers)
  return keep


@deprecated(None, 'Please implement your own file management or use Saver.')
def union(lf, rf):
  """Creates a filter that keeps the union of two filters.

  Args:
    lf: first filter
    rf: second filter

  Returns:
    A filter function that keeps the n largest paths.
  """
  def keep(paths):
    l = set(lf(paths))
    r = set(rf(paths))
    return sorted(list(l|r))
  return keep


@deprecated(None, 'Please implement your own file management or use Saver.')
def negation(f):
  """Negate a filter.

  Args:
    f: filter function to invert

  Returns:
    A filter function that returns the negation of f.
  """
  def keep(paths):
    l = set(paths)
    r = set(f(paths))
    return sorted(list(l-r))
  return keep


@deprecated(None, 'Please implement your own file name management.')
def get_paths(base_dir, parser):
  """Gets a list of Paths in a given directory.

  Args:
    base_dir: directory.
    parser: a function which gets the raw Path and can augment it with
      information such as the export_version, or ignore the path by returning
      None.  An example parser may extract the export version from a path
      such as "/tmp/exports/100" an another may extract from a full file
      name such as "/tmp/checkpoint-99.out".

  Returns:
    A list of Paths contained in the base directory with the parsing function
    applied.
    By default the following fields are populated,
      - Path.path
    The parsing function is responsible for populating,
      - Path.export_version
  """
  raw_paths = gfile.ListDirectory(base_dir)
  paths = []
  for r in raw_paths:
    p = parser(Path(os.path.join(compat.as_str_any(base_dir),
                                 compat.as_str_any(r)),
                    None))
    if p:
      paths.append(p)
  return sorted(paths)
