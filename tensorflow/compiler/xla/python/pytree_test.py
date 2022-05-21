# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for pytrees."""

import types

from absl.testing import absltest
import numpy as np

from tensorflow.compiler.xla.python import xla_extension

pytree = xla_extension.pytree

class SNWrap(types.SimpleNamespace):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

def tree_flatten(x : SNWrap):
  return pytree.flatten(x.__dict__, lambda a: a is not x.__dict__)

def tree_unflatten(aux, values):
  return SNWrap(**pytree.unflatten(aux, values))

pytree.register_node(SNWrap, tree_flatten, tree_unflatten)

def pytree_print(td, leaves):
  def node_visit(xs,node_data):
    if not node_data:
      labels = range(len(xs))
    elif isinstance(node_data, list):
      labels = node_data
    elif isinstance(node_data, xla_extension.PyTreeDef):
      out = node_data.walk(lambda x,d: d, 
                            lambda x:None, 
                            range(node_data.num_leaves), 
                            pass_node_data=True)
      assert len(xs) == len(out) 
      labels = out
    else:
      assert False

    return [*zip(map(str,labels),xs)]

  def print_with_paths(prefix, nodes):
    for (l,x) in nodes:
      p = prefix + '/' + l
      if isinstance(x, list):
        print_with_paths(p, x)
      else:
        print(p + ':', str(x[0]).replace('\n','\\n'))

  out = td.walk(node_visit, lambda x:(x,), leaves, pass_node_data=True)
  print_with_paths('', out)

class PyTreeTest(absltest.TestCase):

  def testWalk(self):
    y = SNWrap()
    y.y1 = {'y11': 'ya', 'y12': 'yb'}

    x = SNWrap()
    x.x1 = {'x11': 'a', 'x12': y, 'x13':'b'}
    x.x2 = {'x21': '2a', 'x22': y}

    obj = {'a1': 'leaf-a1',
           'a2':{'b1': [111,2,3], 
                 'b2': np.ones((2,3)), 
                 'b3': {'c1': 33, 
                        'c2': (44,55)}},
           'a3': x,
           'a4':'fred'}

    leaves,td = pytree.flatten(obj)

    out = td.walk(lambda n:sum(n), lambda x:1, leaves)
    self.assertEqual(out, len(leaves))

    out = td.walk(lambda n:n, lambda x:3, leaves)
    expect = (3, 
              ((3, 3, 3), 3, (3, (3, 3))), 
              ((3, ((3, 3),), 3), (3, ((3, 3),))), 
              3)
    self.assertEqual(out, expect)

    pytree_print(td,leaves)


if __name__ == "__main__":
  absltest.main()
