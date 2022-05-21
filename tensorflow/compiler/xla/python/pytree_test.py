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
    return pytree.flatten(x.__dict__, lambda a: a is not x.__dict__) # only flatten one step

def tree_unflatten(aux, values):
    return SNWrap(**pytree.unflatten(aux, values))

pytree.register_node(SNWrap, tree_flatten, tree_unflatten)

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

    out = td.walk(lambda n,_:sum(n), lambda x:1, leaves)
    self.assertEqual(out, len(leaves))

    out = td.walk(lambda n,_:n, lambda x:3, leaves)
    self.assertEqual(out, (3, ((3, 3, 3), 3, (3, (3, 3))), ((3, ((3, 3),), 3), (3, ((3, 3),))), 3))


if __name__ == "__main__":
  absltest.main()
