
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
"""Tests for LMS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import lms
from tensorflow.python.platform import test
from tensorflow.python.platform.test import mock


class LMSTest(test.TestCase):

    @mock.patch('tensorflow.contrib.lms.LMS._connect_ops')
    @mock.patch('tensorflow.contrib.graph_editor.sgv')
    @mock.patch('tensorflow.python.ops.array_ops.identity')
    def test_add_swapout(self, identity, sgv, connect_ops):
        graph = mock.Mock()
        lms_modifier = lms.LMS(graph=graph,
                               optimizer_scopes={'s1'})
        src_op = mock.Mock()
        ts0 = mock.Mock()
        swap_out = mock.Mock()
        sgv_ret = mock.Mock()
        sgv_ret.output_index.return_value = 123
        sgv.return_value = sgv_ret
        identity.return_value = swap_out
        ret = lms_modifier._add_swapout(src_op, ts0)
        identity.assert_called_once_with(ts0)
        sgv.assert_called_once_with(src_op, graph=graph)
        connect_ops.assert_called_once_with(src_op, swap_out.op,
                                            remap_outputs=True, idx=123)
        self.assertEqual(ret, swap_out.op)
        self.assertEqual(lms_modifier._excl_ops, {swap_out.op})

    @mock.patch('tensorflow.contrib.lms.LMS._connect_ops')
    @mock.patch('tensorflow.contrib.graph_editor.sgv')
    @mock.patch('tensorflow.python.ops.array_ops.identity')
    def test_add_swapin(self, identity, sgv, connect_ops):
        graph = mock.Mock()
        lms_modifier = lms.LMS({'s1'}, graph=graph)
        lms_modifier._topo_sort = mock.Mock()
        dest_op = mock.Mock()
        swapout_op = mock.Mock()
        ts0 = mock.Mock()
        swapin = mock.Mock()
        sgv_ret = mock.Mock()
        sgv_ret.input_index.return_value = 123
        sgv.return_value = sgv_ret
        identity.return_value = swapin
        ret = lms_modifier._add_swapin(swapout_op, dest_op, ts0)
        identity.assert_called_once_with(ts0)
        sgv.assert_called_once_with(dest_op, graph=graph)
        connect_calls = [mock.call(swapout_op, swapin.op),
                         mock.call(swapin.op, dest_op,
                                   remap_inputs=True, idx=123)]
        connect_ops.assert_has_calls(connect_calls)
        self.assertEqual(ret, swapin.op)
        self.assertEqual(lms_modifier._excl_ops, {swapin.op})

    def test_get_branch_ops(self):
        lms_test = lms.LMS({'s1'})
        lms_test._topo_sort = mock.Mock()
        lms_test._topo_sort.size = 6
        threshold = 3
        within_ops = {'f1', 'f2', 'f3', 'f4', 'f5', 'f6'}
        common_orders = {'f1': 1, 'f2': 2, 'f3': 3, 'f4': 4, 'f5': 5, 'f6': 6}
        lms_test._topo_sort.get_order = lambda x: common_orders.get(x, -1)
        ret = lms_test._get_branch_ops(within_ops, threshold)
        self.assertEqual(ret, {'f5', 'f6'})

    @mock.patch('tensorflow.contrib.graph_editor.sgv')
    @mock.patch('tensorflow.contrib.lms.LMS._connect_ops')
    @mock.patch('tensorflow.contrib.lms.LMS._add_control_dependency')
    @mock.patch('tensorflow.python.ops.array_ops.identity')
    def test_fuse_swapin_ops(self, identity, ctrl_dep, connect, sgv):
        lms_test = lms.LMS({'s1'}, graph=mock.Mock(), lb=5, ub=500)
        lms_test._topo_sort = mock.Mock()
        # Mock get_order to return the "order" value from the mock op
        lms_test._topo_sort.get_order = lambda x: x.order
        lms_test._topo_sort.size = 200
        swap_in = mock.Mock(name='swapin_ts', op=mock.Mock(name='swapin_op'))
        identity.return_value = swap_in
        sgv.return_value.input_index.return_value = 123
        src_op = mock.Mock(name='src_op')
        swapout_op = mock.Mock(name='swapout_op')
        bw_fr_ops = [mock.Mock(name='op1', order=15),
                     mock.Mock(name='op2', order=-1),
                     mock.Mock(name='op3', order=10),
                     mock.Mock(name='op4-earliest', order=5),
                     mock.Mock(name='op5', order=8),
                     mock.Mock(name='op6', order=-2)]
        ts0 = mock.Mock(name='ts0')
        earliest_op = bw_fr_ops[3]

        ret = lms_test._fuse_swapin_ops(src_op, swapout_op, set(bw_fr_ops),
                                        ts0)
        self.assertEqual(ret, {bw_fr_ops[1], bw_fr_ops[5]})
        connect_calls = [mock.call(swapout_op, swap_in.op),
                         mock.call(swap_in.op, bw_fr_ops[0], remap_inputs=True,
                                   idx=123),
                         mock.call(swap_in.op, bw_fr_ops[2], remap_inputs=True,
                                   idx=123),
                         mock.call(swap_in.op, bw_fr_ops[3], remap_inputs=True,
                                   idx=123),
                         mock.call(swap_in.op, bw_fr_ops[4], remap_inputs=True,
                                   idx=123)]
        connect.assert_has_calls(connect_calls, any_order=True)
        self.assertEqual(lms_test._excl_ops, {swap_in.op})
        ctrl_dep.assert_called_once_with(src_op, earliest_op, swap_in.op)

    @mock.patch('tensorflow.contrib.lms.LMS._find_new_src_op')
    @mock.patch('tensorflow.contrib.lms.LMS._fuse_swapin_ops')
    @mock.patch('tensorflow.contrib.lms.LMS._add_control_dependency')
    @mock.patch('tensorflow.contrib.lms.LMS._add_swapin')
    @mock.patch('tensorflow.contrib.lms.LMS._add_swapout')
    @mock.patch('tensorflow.contrib.lms.LMS._get_forward_walk_ops')
    @mock.patch('tensorflow.contrib.graph_editor.util.get_consuming_ops')
    def test_insert_swap_nodes(self, consuming_ops, fwd_walk_ops, swapout,
                               swapin, ctrldep, fuse_swapins, find_new_src):
        graph = mock.Mock()
        lms_test = lms.LMS({'s1'}, graph=graph, fuse_swapins=False)
        # Test op is excluded.
        # _insert_swap_nodes should return before accessing methods on
        # src_op which would blow up the test
        lms_test._excl_ops = {'a'}
        lms_test._insert_swap_nodes('a')
        lms_test._excl_ops ={}

        # Test included ops but op is not included.
        # _insert_swap_nodes should return before accessing methods on
        # src_op which would blow up the test
        inc_op = mock.Mock()
        lms_test._incl_ops = {inc_op}
        lms_test._insert_swap_nodes('a')

        # Test included ops and op is included.
        # _insert_swap_nodes should start iterating on the src outputs
        inc_op = mock.Mock()
        inc_op.outputs = ['a']
        lms_test._incl_ops = {inc_op}
        lms_test._grad_ops = {'b', 'c'}
        consuming_ops.return_value = {'b'}
        fwd_walk_ops.return_value = {}
        lms_test._insert_swap_nodes(inc_op)
        consuming_ops.assert_called_once_with('a')
        fwd_walk_ops.assert_called_once_with('b', inclusive=False)
        lms_test._incl_ops = {}
        consuming_ops.reset_mock()
        fwd_walk_ops.reset_mock()

        # Test creating swap out and swap in nodes
        src_op = mock.Mock()
        src_op.outputs = ['a', 'z']
        lms_test._grad_ops = {'b', 'c', 'g1', 'g2'}
        consuming_ops.side_effect = [{'b', 'f2', 'g2'} , {'c', 'g1', 'f3'}]
        swapout.side_effect = ['swapout_op1', 'swapout_op2']
        swapin.side_effect = ['swapin_op1', 'swapin_op2', 'swapin_op3',
                              'swapin_op4']
        fwd_walk_ops.return_value = {'d'}
        lms_test._topo_sort = mock.Mock()
        lms_test._topo_sort.get_order.side_effect = lambda x: 0
        lms_test._insert_swap_nodes(src_op)
        consuming_ops.assert_has_calls([mock.call('a'), mock.call('z')])
        fwd_calls = [mock.call('b', inclusive=False),
                     mock.call('g2', inclusive=False)]
        fwd_walk_ops.assert_has_calls(fwd_calls, any_order=True)
        swapout_calls = [mock.call(src_op, 'a'),
                         mock.call(src_op, 'z')]
        swapout.assert_has_calls(swapout_calls)
        swapin_calls = [mock.call('swapout_op1', 'b', 'a'),
                        mock.call('swapout_op1', 'g2', 'a'),
                        mock.call('swapout_op2', 'c', 'z'),
                        mock.call('swapout_op2', 'g1', 'z')]
        swapin.assert_has_calls(swapin_calls, any_order=True)
        ctrldep_calls = [mock.call(src_op, 'b', 'swapin_op1'),
                         mock.call(src_op, 'g2', 'swapin_op2')]
        ctrldep.assert_has_calls(ctrldep_calls, any_order=True)
        self.assertEqual(lms_test._incpu_count, 2)

        # Test calling _find_new_src_op
        lms_test = lms.LMS({'s1'}, graph=graph, fuse_swapins=False)
        src_op = mock.Mock()
        src_op.outputs = ['a', 'z']
        lms_test._grad_ops = {'b', 'c', 'g1', 'g2'}
        consuming_ops.side_effect = [{'b', 'f2', 'g2'} , {'c', 'g1', 'f3'}]
        swapout.side_effect = ['swapout_op1', 'swapout_op2']
        swapin.side_effect = ['swapin_op1', 'swapin_op4']
        fwd_walk_ops.return_value = {'d'}
        lms_test._topo_sort = mock.Mock()

        op_orders = {'b': 1, 'g2': -1, 'c': -1, 'g1': 1}
        get_order = lambda x: op_orders.get(x, 1)
        lms_test._topo_sort.get_order.side_effect = get_order
        # New src ops for recursion
        find_new_src.side_effect = [{'z1'}, {'z2'}]
        # Put ops 'z1' and 'z2' into excluded ops so the recursive call to
        # _insert_swap_nodes stops early.
        lms_test._excl_ops = {'z1', 'z2'}
        lms_test._insert_swap_nodes(src_op)
        consuming_ops.assert_has_calls([mock.call('a'), mock.call('z')])
        fwd_calls = [mock.call('b', inclusive=False),
                     mock.call('g2', inclusive=False)]
        fwd_walk_ops.assert_has_calls(fwd_calls, any_order=True)
        swapout_calls = [mock.call(src_op, 'a'),
                         mock.call(src_op, 'z')]
        swapout.assert_has_calls(swapout_calls)
        swapin_calls = [mock.call('swapout_op1', 'b', 'a'),
                        mock.call('swapout_op2', 'g1', 'z')]
        swapin.assert_has_calls(swapin_calls, any_order=True)
        ctrldep_calls = [mock.call(src_op, 'b', 'swapin_op1'),
                         mock.call(src_op, 'g1', 'swapin_op4')]
        ctrldep.assert_has_calls(ctrldep_calls, any_order=True)
        find_new_src.assert_has_calls([mock.call('g2'),
                                       mock.call('c')], any_order=True)
        self.assertEqual(lms_test._incpu_count, 2)

        # Test calling fuse_swapins
        consuming_ops.reset_mock()
        swapout.reset_mock()
        swapin.reset_mock()
        fwd_walk_ops.reset_mock()
        graph = mock.Mock()
        lms_test = lms.LMS({'s1'}, graph=graph, fuse_swapins=True)
        lms_test._topo_sort = mock.Mock()
        lms_test._topo_sort.get_order.side_effect = lambda x: 0
        lms_test._grad_ops = {'b', 'c', 'g1', 'g2'}
        consuming_ops.side_effect = [{'b', 'f2', 'g2'} , {'c', 'g1', 'f3'}]
        swapout.side_effect = ['swapout_op1', 'swapout_op2']
        swapin.side_effect = ['swapin_op1', 'swapin_op2', 'swapin_op3',
                              'swapin_op4']
        fwd_walk_ops.return_value = {'d'}
        fuse_swapins.return_value = ['fuse1', 'fuse2']
        lms_test._insert_swap_nodes(src_op)
        fuse_calls = [mock.call(src_op, "swapout_op1", {'b', 'g2'}, 'a'),
                      mock.call(src_op, "swapout_op2", {'c', 'g1'}, 'z')]
        fuse_swapins.assert_has_calls(fuse_calls)

        # Test stop swapping out once max number of tensors to swap is hit
        consuming_ops.reset_mock()
        lms_test = lms.LMS({'s1'}, graph=graph, fuse_swapins=False)
        lms_test._n_tensors = 10
        lms_test._incpu_count = 10
        lms_test._insert_swap_nodes(src_op)
        self.assertFalse(consuming_ops.called)

    @mock.patch('tensorflow.contrib.graph_editor.util.get_consuming_ops')
    def test_find_new_src_op(self, get_cons):
        ts = mock.Mock()
        lms_test = lms.LMS({'s1'})
        lms_test._topo_sort = ts
        ts.bw_starting_order = 5
        ts.get_order.side_effect = lambda x: x.order
        get_cons.side_effect = lambda x: x.consumers

        op_w_order = mock.Mock(name='op_w_order', order=45, outputs=[])
        frontier1 = mock.Mock(name='fwd1', order=-1, outputs=[])
        frontier2_out = mock.Mock(name='fwd2_out', consumers=[op_w_order])
        frontier2 = mock.Mock(name='fwd2', order=-1, outputs=[frontier2_out])
        orig_src_out = mock.Mock(name='orig_out', consumers=[frontier1,
                                                             frontier2])
        original_op = mock.Mock(name='original_op', outputs=[orig_src_out])
        new_src_ops = lms_test._find_new_src_op(original_op)
        self.assertEqual(new_src_ops, {frontier2})


    @mock.patch('tensorflow.contrib.graph_editor.connect')
    @mock.patch('tensorflow.contrib.graph_editor.sgv')
    def test_connect_ops(self, sgv, connect):
        graph = mock.Mock()
        lms_test = lms.LMS({'s1'}, graph=graph)
        src_op = mock.Mock()
        dest_op = mock.Mock()
        dcf = mock.Mock()
        sgv.side_effect = ['a', 'b']
        lms_test._connect_ops(src_op, dest_op, remap_inputs=False,
                              remap_outputs=False, disconnect_first=dcf)
        sgv_calls = [mock.call(src_op, graph=graph),
                     mock.call(dest_op, graph=graph)]
        sgv.assert_has_calls(sgv_calls)
        connect.assert_called_once_with('a', 'b', dcf)

        # Test remaps
        connect.reset_mock()
        sgv.reset_mock()
        src_sgv = mock.Mock()
        dest_sgv = mock.Mock()
        sgv.side_effect = [src_sgv, dest_sgv]
        lms_test._connect_ops(src_op, dest_op, remap_inputs=True,
                              remap_outputs=True, idx=2, disconnect_first=dcf)
        src_sgv.remap_outputs.assert_called_once_with([2])
        dest_sgv.remap_inputs.assert_called_once_with([2])
        connect.assert_called_once_with(src_sgv.remap_outputs.return_value,
                                        dest_sgv.remap_inputs.return_value,
                                        dcf)

    @mock.patch('tensorflow.contrib.graph_editor.get_name_scope_ops')
    def test_filter_scopes_and_types(self, get_name_scope_ops):
        op1 = mock.Mock()
        op2 = mock.Mock()
        op3 = mock.Mock()
        op4 = mock.Mock()
        op1.type = 'a'
        op2.type = 'b'
        op3.type = 'a'
        op3.type = 'c'
        op4.type = 'd'
        within_ops = {op1, op2, op3, op4}
        lms_test = lms.LMS({'s1'})
        get_name_scope_ops.return_value = [op4]
        ret = lms_test._filter_scopes_and_types(within_ops, {'s1', 's2'},
                                                {'a', 'c'})
        get_name_scope_ops.assert_has_calls([mock.call(within_ops, 's1'),
                                             mock.call(within_ops, 's2')],
                                             any_order=True)
        self.assertEqual(ret, {op1, op3, op4})

    @mock.patch('tensorflow.contrib.graph_editor.make_list_of_op')
    @mock.patch('tensorflow.contrib.graph_editor.filter_ops_from_regex')
    def test_build_gradient_ops(self, filter_ops, make_list):
        graph = mock.Mock()
        filter_ops.side_effect = [['a', 'b', 'c'], ['d']]
        lms_test = lms.LMS({'s1', 's2'}, graph=graph)
        lms_test._build_gradient_ops()
        self.assertEqual(lms_test._grad_ops, {'a', 'b', 'c', 'd'})
        self.assertEqual(2, make_list.call_count)
        self.assertEqual(2, filter_ops.call_count)

    @mock.patch('tensorflow.contrib.graph_editor.get_forward_walk_ops')
    def test_get_forward_walk_ops(self, fwd_walk):
        lms_test = lms.LMS({'s1'})
        ops_dict = {mock.sentinel.op1: [mock.sentinel.op1, mock.sentinel.op10],
                    mock.sentinel.op2: [mock.sentinel.op2, mock.sentinel.op20],
                    mock.sentinel.op3: [mock.sentinel.op3, mock.sentinel.op30]}
        lms_test._ops_dict = ops_dict

        # Test op in ops_dict
        ret = lms_test._get_forward_walk_ops(mock.sentinel.op1)
        self.assertEqual(ret, [mock.sentinel.op1, mock.sentinel.op10])
        # Test inclusive=False
        ret = lms_test._get_forward_walk_ops(mock.sentinel.op1,
                                             inclusive=False)
        self.assertEqual(ret, [mock.sentinel.op10])

        # Test op not in ops_dict
        fwd_walk.return_value = [mock.sentinel.op, mock.sentinel.ret]
        ret = lms_test._get_forward_walk_ops(mock.sentinel.op)
        self.assertEqual(ret, [mock.sentinel.op, mock.sentinel.ret])
        # Test inclusive=False
        lms_test._ops_dict = ops_dict
        fwd_walk.return_value = [mock.sentinel.op, mock.sentinel.ret]
        ret = lms_test._get_forward_walk_ops(mock.sentinel.op,
                                             inclusive=False)
        self.assertEqual(ret, [mock.sentinel.ret])

    @mock.patch('tensorflow.contrib.lms.LMS._do_action')
    @mock.patch('tensorflow.contrib.lms.TOPOS.build')
    @mock.patch('tensorflow.contrib.lms.LMS._filter_scopes_and_types')
    @mock.patch('tensorflow.contrib.graph_editor.get_forward_walk_ops')
    @mock.patch('tensorflow.contrib.lms.LMS._get_forward_walk_ops')
    @mock.patch('tensorflow.contrib.lms.LMS._get_seed_ops')
    @mock.patch('tensorflow.contrib.lms.LMS._build_gradient_ops')
    def test_run(self, grad, seed, fwd_walk, tf_fwd_walk, filter, build, action):
        # Test mainline through
        seed_ops = [mock.Mock() for x in range(5)]
        grad_ops = [mock.Mock() for x in range(6)]
        fwd_walk.return_value = [mock.Mock() for x in range(3)] + grad_ops
        seed.return_value = seed_ops
        lms_test = lms.LMS({'s1'}, graph=mock.Mock())

        def fake_build_gradient_ops():
            lms_test._grad_ops = set(grad_ops)

        grad.side_effect = fake_build_gradient_ops
        lms_test.topos = mock.Mock()
        lms_test.run()
        self.assertTrue(grad.called)
        self.assertTrue(seed.called)
        self.assertTrue(fwd_walk.call_count, len(seed_ops))
        reachable = set(fwd_walk.return_value) - set(grad_ops)
        filter.assert_has_calls([mock.call(reachable, mock.ANY, mock.ANY),
                                 mock.call(reachable, mock.ANY, mock.ANY)])
        self.assertTrue(build.called)
        action.assert_called_once_with(seed_ops)

        # Test passing a graph in run and verify it overwrites a graph passed
        # on the constructor
        new_graph = mock.Mock()
        lms_test.run(graph=new_graph)
        self.assertEqual(lms_test._graph, new_graph)

        # Test no graph
        grad.reset_mock()
        lms_test = lms.LMS({'s1'})
        self.assertRaises(ValueError, lms_test.run)

        # Test n_tensors = 0
        lms_test = lms.LMS({'s1'}, n_tensors=0)
        lms_test.run(new_graph)
        self.assertFalse(grad.called)

        # Test n_tensors = -1
        action.reset_mock()
        lms_test = lms.LMS({'s1'}, n_tensors=-1)
        lms_test.run(new_graph)
        self.assertTrue(action.called)
        # n_tensors gets set to when passed as -1
        self.assertEqual(lms_test._n_tensors, 0)

    @mock.patch('tensorflow.contrib.lms.LMS._insert_swap_nodes')
    @mock.patch('tensorflow.contrib.graph_editor.util.get_consuming_ops')
    def test_do_action(self, get_cons, swap):
        lms_test = lms.LMS({'s1'})
        src_ops = [mock.Mock() for x in range(2)]
        grad_op1 = mock.Mock()
        lms_test._grad_ops = {grad_op1}
        for op in src_ops:
            op.outputs = ['o' for x in range(2)]
        dup_op = mock.Mock()
        get_cons_side_effect = [[mock.Mock() for x in range(3)] + [dup_op],
                                [mock.Mock() for x in range(2)] + [dup_op],
                                [mock.Mock() for x in range(2)],
                                [mock.Mock() for x in range(3)]]
        # Set outputs of first level consuming ops
        for op_list in get_cons_side_effect:
            for op in op_list:
                op.outputs=['a']
        for x in range(11):
            get_cons_side_effect.append([grad_op1])

        get_cons.side_effect = get_cons_side_effect
        # Now add gradient ops for the remaining calls to get_consuming_ops
        # Set all consuming ops to have a gradient as
        lms_test._do_action(src_ops)
        # There should be 13 calls to _insert_swap_nodes.  The original 2
        # nodes, the 10 from the side_effect ranges above and ONE call on the
        # dup_op
        self.assertEqual(swap.call_count, 13)

        self.assertEqual(get_cons.call_count, 15)

        # Test when NOT swapping all possible tensors
        swap.reset_mock()
        get_cons.reset_mock()
        lms_test = lms.LMS(optimizer_scopes={'s1'}, n_tensors=7)

        def fake_swap(op):
            lms_test._incpu_count += len(op.outputs)

        swap.side_effect = fake_swap
        lms_test._grad_ops = {grad_op1}
        get_cons.side_effect = get_cons_side_effect
        lms_test._do_action(src_ops)

        # There should only be 5 calls to insert swap nodes.  The first
        # seed ops will swap 2 tensors each, and 3 other ops will each swap 1
        self.assertEqual(swap.call_count, 5)

    @mock.patch('tensorflow.contrib.graph_editor.filter_ops_from_regex')
    @mock.patch('tensorflow.contrib.graph_editor.make_list_of_op')
    @mock.patch('tensorflow.contrib.graph_editor.get_forward_walk_ops')
    @mock.patch('tensorflow.contrib.graph_editor.util.get_consuming_ops')
    def test_get_seed_ops(self, get_consuming, get_fwd_walk, make_list,
                          filter_ops):
        graph = mock.Mock()

        # Test with starting scope
        lms_test = lms.LMS({'s1'}, graph=graph, starting_scope='sc')
        filter_ops.side_effect = lambda x, y: {'a', 'b'}
        ret = lms_test._get_seed_ops()
        filter_ops.assert_called_once_with(make_list.return_value, "^sc")
        make_list.assert_called_once_with(graph)

        # Test with starting op names
        filter_ops.reset_mock()
        make_list.reset_mock()
        name_ops = [mock.Mock(name='op1'), mock.Mock(name='op2')]
        filter_ops.side_effect = [[name_ops[0]], [name_ops[1]]]
        lms_test = lms.LMS({'s1'}, graph=graph, starting_op_names={'a', 'b'})
        ret = lms_test._get_seed_ops()
        filter_ops.assert_has_calls([mock.call(make_list.return_value, "^a$"),
                                    mock.call(make_list.return_value, "^b$")],
                                    any_order=True)
        make_list.assert_called_once_with(graph)
        self.assertEqual(set(ret), set(name_ops))

        # Test building seed ops with graph traversal
        # setup graph operations and their outputs
        graph_ops = [mock.Mock() for x in range(6)]
        for op in graph_ops:
            op.outputs = ['mock_output']
        graph.get_operations.return_value = graph_ops

        # Consuming ops for the graph operations, make most of them have a
        # gradient op consuming.
        grad_ops = {mock.sentinel.gradient1, mock.sentinel.gradient2}
        non_grad_ops= graph_ops
        get_consuming.side_effect = [[mock.sentinel.gradient1],
                                     [mock.sentinel.gradient1],
                                     [mock.sentinel.gradient2],
                                     ['nothing'],
                                     [mock.sentinel.gradient2],
                                     ['nothing2']]
        get_fwd_walk.side_effect = [[],
                                    [graph_ops[0], graph_ops[2], graph_ops[4]],
                                    [graph_ops[0], graph_ops[1], graph_ops[4]],
                                    [graph_ops[1], graph_ops[2]]]
        lms_test = lms.LMS({'s1'}, graph=graph)
        lms_test._grad_ops = grad_ops
        ret = lms_test._get_seed_ops()
        self.assertEqual(get_consuming.call_count, 6)
        walk_calls = [mock.call(graph_ops[0], within_ops=non_grad_ops, inclusive=False),
                      mock.call(graph_ops[1], within_ops=non_grad_ops, inclusive=False),
                      mock.call(graph_ops[2], within_ops=non_grad_ops, inclusive=False),
                      mock.call(graph_ops[4], within_ops=non_grad_ops, inclusive=False)]
        get_fwd_walk.assert_has_calls(walk_calls, any_order=True)
        # There will be two seed ops. However, since sets are used in
        # _get_seed_ops and we are mocking get_forward_walk_ops, we can't
        # check for specific mock operations.
        self.assertEqual(len(ret), 2)

    @mock.patch('tensorflow.contrib.graph_editor.add_control_inputs')
    @mock.patch('tensorflow.contrib.lms.LMS._do_direct_order')
    @mock.patch('tensorflow.contrib.lms.LMS._do_chain_rule')
    def test_add_control_dependency(self, do_chain, do_direct, add_ctrl_input):
        # Test when lb is reset and chain rule
        lms_test = lms.LMS({'s1'}, ctrld_strategy="chain_rule", lb=10, ub=20)

        lms_test._topo_sort = mock.Mock()
        lms_test._topo_sort.get_order.side_effect = [24, 15]
        fw_op = mock.sentinel.fw_op
        bw_op = mock.sentinel.orig_bw_op
        swapin_op = mock.Mock()
        do_chain.return_value = [mock.sentinel.ctl_op, 123]
        lms_test._add_control_dependency(fw_op, bw_op, swapin_op)
        do_chain.assert_called_once_with(fw_op, bw_op, 1, 20)
        add_ctrl_input.assert_called_once_with(swapin_op,
                                               mock.sentinel.ctl_op)

        # Test chain rule
        lms_test._topo_sort.get_order.side_effect = [26, 15]
        do_chain.reset_mock()
        add_ctrl_input.reset_mock()
        lms_test._add_control_dependency(fw_op, bw_op, swapin_op)
        do_chain.assert_called_once_with(fw_op, bw_op,
                                         10, 20)
        add_ctrl_input.assert_called_once_with(swapin_op,
                                               mock.sentinel.ctl_op)

        # Test with direct_order
        do_chain.reset_mock()
        add_ctrl_input.reset_mock()
        lms_test = lms.LMS({'s1'}, ctrld_strategy="direct_order", lb=10, ub=20)

        lms_test._topo_sort = mock.Mock()
        lms_test._topo_sort.get_order.side_effect = [26, 15]
        do_chain.return_value = None
        do_direct.return_value = [mock.sentinel.ctl_op, 567]
        lms_test._add_control_dependency(fw_op, bw_op, swapin_op)
        add_ctrl_input.assert_called_once_with(swapin_op,
                                               mock.sentinel.ctl_op)
        do_direct.assert_called_once_with(fw_op, mock.sentinel.orig_bw_op,
                                          10, 20)

        # Test direct order when fw_op is a gradient op
        do_direct.reset_mock()
        do_chain.reset_mock()
        add_ctrl_input.reset_mock()
        lms_test = lms.LMS({'s1'}, ctrld_strategy="chain_rule", lb=10, ub=20)
        lms_test._grad_ops = {fw_op}
        lms_test._topo_sort = mock.Mock()
        lms_test._topo_sort.get_order.side_effect = [26, 15]
        do_chain.return_value = None
        do_direct.return_value = [mock.sentinel.ctl_op, 567]
        lms_test._add_control_dependency(fw_op, bw_op, swapin_op)
        add_ctrl_input.assert_called_once_with(swapin_op,
                                               mock.sentinel.ctl_op)
        # Note we expect _do_direct_order to be called even though the
        # control dependency strategy was set to "chain_rule" because fw_op is
        # a gradient op.
        do_direct.assert_called_once_with(fw_op, mock.sentinel.orig_bw_op,
                                          10, 20)
        self.assertEqual(do_chain.call_count, 0)

        # Test direct order when fw_op is a gradient op
        do_direct.reset_mock()
        do_chain.reset_mock()
        add_ctrl_input.reset_mock()
        lms_test = lms.LMS({'s1'}, ctrld_strategy="chain_rule", lb=10, ub=20)
        lms_test._grad_ops = {fw_op}
        lms_test._topo_sort = mock.Mock()
        lms_test._topo_sort.get_order.side_effect = [26, 15]
        do_chain.return_value = None
        do_direct.return_value = [mock.sentinel.ctl_op, 567]
        lms_test._add_control_dependency(fw_op, bw_op, swapin_op)
        add_ctrl_input.assert_called_once_with(swapin_op,
                                               mock.sentinel.ctl_op)
        # Note we expect _do_direct_order to be called even though the
        # control dependency strategy was set to "chain_rule" because fw_op is
        # a gradient op.
        do_direct.assert_called_once_with(fw_op, mock.sentinel.orig_bw_op,
                                          10, 20)
        self.assertEqual(do_chain.call_count, 0)

    @mock.patch('tensorflow.contrib.graph_editor.util.get_consuming_ops')
    @mock.patch('tensorflow.contrib.lms.LMS._do_direct_order')
    def test_do_chain_rule(self, direct_order, cons_ops):
        lms_test = lms.LMS({'s1'})
        lms_test._topo_sort = mock.Mock()

        # Test calling _do_direct_order when the bw_op is close to the
        # boundary
        lms_test._topo_sort.get_order.side_effect = [1, 5]
        lms_test._topo_sort.bw_starting_order = 10
        fwd_op = mock.Mock(name='fwdop')
        bw_op = mock.Mock(name='bwop')
        lms_test._do_chain_rule(fwd_op, bw_op, 4, 10)
        direct_order.assert_called_once_with(fwd_op, bw_op, 4, 10)

        # Test going through one layer
        lms_test._topo_sort = mock.Mock()
        # Mock get_order to return the "order" value from the mock op
        lms_test._topo_sort.get_order = lambda x: x.order
        # Mock get_consuming_ops to return the mock outputs' consumers
        cons_ops.side_effect = lambda x: x.consumers
        grad_op = mock.Mock(name='grad', order=7)
        grad_op.name = 'grad'
        lms_test._grad_ops = {grad_op}
        layer2 = [mock.Mock(name='l2a',
                            outputs=[mock.Mock(name='secondToLast',
                                               outputs=[mock.Mock(
                                                   name='a')],
                                               consumers=[grad_op])]),
                  mock.Mock(name='l2b', outputs=[])]
        layer2b = mock.Mock(outputs=[])
        layer1 = [mock.Mock(name='l1a', outputs=[mock.Mock(consumers=layer2)]),
                  mock.Mock(name='l1b',
                            outputs=[mock.Mock(consumers=[layer2b])])]
        fwd_op = mock.Mock(name='fwdop', order=5,
                           outputs=[mock.Mock(name='fwdop_out1',
                                              consumers=layer1),
                                    mock.Mock(name='fwdop_out2',
                                              consumers=[], outputs=[])])
        bw_op = mock.Mock(name='bwop', order=50)
        lms_test._topo_sort.bw_starting_order = 1
        ret = lms_test._do_chain_rule(fwd_op, bw_op, 1, 10)
        self.assertEqual(ret, (grad_op, 7))

    @mock.patch('tensorflow.contrib.graph_editor.get_forward_walk_ops')
    def test_do_direct_order(self, get_fwd_walk):
        lms_test = lms.LMS({'s1'})
        lms_test._topo_sort = mock.Mock()
        # Mock get_order to return the "order" value from the mock op
        lms_test._topo_sort.get_order.side_effect = lambda x: x.order
        expected_ret = mock.Mock(name='expected_op')
        expected_ret.name = 'expected_op'
        get_ops = mock.Mock(name='get_ops')
        gor = [[]]
        get_ops.side_effect = [[1, 2, 3], [4, 5], [6, expected_ret]]
        lms_test._topo_sort.get_ops = get_ops
        fw_op = mock.Mock(name='fwd_op', order=5)
        src_op = mock.Mock(name='src_op', order=50)
        fw_walk = lambda x: {'a', src_op} if x is expected_ret else {'b', 'c'}
        get_fwd_walk.side_effect = fw_walk
        ret = lms_test._do_direct_order(fw_op, src_op, 3, 100)
        self.assertEqual(ret, (expected_ret, 44))

if __name__ == '__main__':
  test.main()
