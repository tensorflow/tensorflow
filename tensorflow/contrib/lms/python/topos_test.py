# (C) Copyright IBM Corp. 2018. All Rights Reserved.
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
"""Tests for the LMS topos module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import lms
from tensorflow.python.platform import test
from tensorflow.python.platform.test import mock


class TOPOSTest(test.TestCase):

    @mock.patch('toposort.toposort')
    @mock.patch('tensorflow.contrib.lms.TOPOS._build_order_dict')
    @mock.patch('tensorflow.contrib.lms.TOPOS._reindex')
    @mock.patch('tensorflow.contrib.lms.TOPOS._clean_update_ops')
    @mock.patch('tensorflow.contrib.lms.TOPOS._clean_bw_ops')
    @mock.patch('tensorflow.contrib.lms.TOPOS._build_dependency_dict')
    def test_build(self, build_dep, clean_bw, clean_update, reindex, build_order, tps):
        grad_ops = {6, 7}
        topsosort_return = [{1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {10, 11, 12, 13}]
        tps.return_value = iter(topsosort_return)
        topo_test = lms.TOPOS({}, grad_ops)
        topo_test.build()
        #tps.assert_called_once_with(build_dep.return_value)
        self.assertTrue(clean_bw.called)
        self.assertTrue(clean_update.called)
        self.assertTrue(reindex.called)
        self.assertTrue(build_order.called)
        self.assertEqual(topo_test._bw_starting_order, 2)

    @mock.patch('tensorflow.contrib.graph_editor.util.get_consuming_ops')
    @mock.patch('tensorflow.contrib.graph_editor.util.get_generating_ops')
    @mock.patch('tensorflow.contrib.graph_editor.get_walks_intersection_ops')
    def test_build_dependency_dict(self, intersec, get_gen, get_cons):
        ops = [mock.Mock(name=('op%s' % x),
                         control_inputs=set(),
                         inputs=set(),
                         outputs=set())
               for x in range(5)]
        tensors = [mock.Mock(name=('ts%s' % x),
                             consuming_ops=set(),
                             generating_ops=set())
                   for x in range(4)]
        # Build mock graph
        seed_ops = {ops[0]}
        ops[0].outputs = {tensors[0], tensors[1]}
        tensors[0].consuming_ops = [ops[1], ops[2]]
        tensors[0].generating_ops = [ops[0]]
        tensors[1].consuming_ops = [ops[3]]
        tensors[1].generating_ops = [ops[0]]

        ops[1].inputs = {tensors[0]}
        ops[1].outputs = {tensors[2]}
        ops[2].inputs = {tensors[0]}
        ops[2].outputs = {tensors[3]}
        tensors[2].generating_ops = [ops[1]]
        tensors[3].generating_ops = [ops[2]]
        tensors[2].consuming_ops = [ops[4]]
        tensors[3].consuming_ops = [ops[4]]
        ops[3].inputs = {tensors[1]}

        ops[4].inputs = {tensors[2], tensors[3]}

        grad_ops = {}

        get_cons.side_effect = lambda x: x.consuming_ops
        get_gen.side_effect = lambda x: x.generating_ops
        intersec.return_value = ops
        topo_test = lms.TOPOS(seed_ops, grad_ops)
        ret = topo_test._build_dependency_dict()
        expected_dict = {ops[0]: set(),
                         ops[1]: {ops[0]},
                         ops[2]: {ops[0]},
                         ops[3]: {ops[0]},
                         ops[4]: {ops[1], ops[2]}}
        self.assertDictEqual(expected_dict, ret)

    def test_build_order_dict(self):
        grad_ops = {'g1', 'g2'}
        topo_test = lms.TOPOS({}, grad_ops)
        topo_test._topo_sort = { 0: {'a', 'b'},
                                 1: {'c', 'd'},
                                 2: {'e', 'f'},
                                 3: {'g1', 'g2'}}
        topo_test._build_order_dict()
        expected_val = { 'a': 0, 'b' : 0,
                         'c': 1, 'd' : 1,
                         'e': 2, 'f' : 2,
                         'g1': 3, 'g2' : 3}
        self.assertDictEqual(expected_val, topo_test._orders)

    def test_clean_bw_ops(self):
        grad_ops = {'g1', 'g2'}
        topo_test = lms.TOPOS({}, grad_ops)
        topo_test._topo_sort = { 0: {'a', 'b'},
                                 1: {'c', 'd', 'g1'},
                                 2: {'e', 'f', 'g1', 'g2'},
                                 3: {'g1', 'g2'}}
        topo_test._clean_bw_ops()
        expected_val = { 0: {'a', 'b'},
                         1: {'c', 'd'},
                         2: {'e', 'f'},
                         3: {'g1', 'g2'}}
        self.assertDictEqual(topo_test._topo_sort, expected_val)

    @mock.patch('tensorflow.contrib.graph_editor.get_forward_walk_ops')
    def test_clean_update_ops(self, fwd_walk):
        grad_ops = {'g1', 'g2'}
        topo_test = lms.TOPOS({}, grad_ops)
        topo_test._topo_sort = { 0: {'a', 'b'},
                                 1: {'c', 'd', 'g1'},
                                 2: {'e', 'f', 'g1', 'g2'},
                                 3: {'g1', 'g2'}}
        remove_bcf = lambda x, inclusive=False: {'b', 'c', 'f'}
        fwd_walk.side_effect = remove_bcf
        topo_test._clean_update_ops()
        expected = { 0: {'a'},
                     1: {'d', 'g1'},
                     2: {'e', 'g1', 'g2'},
                     3: {'g1', 'g2'}}
        self.assertDictEqual(expected, topo_test._topo_sort)

    def test_reindex(self):
        topo_test = lms.TOPOS({}, {})
        topo_test._topo_sort = { 0: {'a', 'b'},
                                 1: {},
                                 2: {'e', 'f', 'g1', 'g2'},
                                 3: {},
                                 4: {'a'}}
        topo_test._reindex()
        expected = { 0: {'a', 'b'},
                     1: {'e', 'f', 'g1', 'g2'},
                     2: {'a'}}
        self.assertDictEqual(expected, topo_test._topo_sort)

    def test_get_order(self):
        topo_test = lms.TOPOS({}, {})
        topo_test._orders = { 'a': 0, 'b' : 0,
                              'c': 1, 'd' : 1,
                              'e': 2, 'f' : 2,
                              'g1': 3, 'g2' : 3}
        self.assertEqual(topo_test.get_order('a'), 0)
        self.assertEqual(topo_test.get_order('f'), 2)
        self.assertEqual(topo_test.get_order('g1'), 3)
        self.assertEqual(topo_test.get_order('asdf'), -1)

    def test_get_ops(self):
        topo_test = lms.TOPOS({}, {})
        topo_test._topo_sort = { 0: {'a', 'b'},
                                 1: {'c', 'd', 'g1'},
                                 2: {'e', 'f', 'g1', 'g2'},
                                 3: {'g1', 'g2'}}
        ret = topo_test.get_ops(2)
        self.assertEqual(ret, {'e', 'f', 'g1', 'g2'})

    def test_size(self):
        topo_test = lms.TOPOS({}, {})
        topo_test._topo_sort = { 0: {'a', 'b'},
                                 1: {'c', 'd', 'g1'},
                                 2: {'e', 'f', 'g1', 'g2'},
                                 3: {'g1', 'g2'}}
        self.assertEqual(topo_test.size, 4)

    def test_bw_starting_order(self):
        topo_test = lms.TOPOS({}, {})
        topo_test._bw_starting_order = 100
        self.assertEqual(topo_test.bw_starting_order, 100)

if __name__ == '__main__':
  test.main()
