# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

import itertools
import networkx as nx
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.contrib.ipu.python import sharded_optimizer as so
from tensorflow.python.framework import ops

def tensor_memory_use(t):
  return t.shape.num_elements() * t.dtype.size

def get_shard_from_colocation(op):
  g = op.graph
  for c in op.colocation_groups():
    coloc_op = g.get_operation_by_name(c.decode('utf-8')[5:])
    if so.has_attr(coloc_op, so._XLA_SHARDING):
      attr = coloc_op.get_attr(so._XLA_SHARDING);
      return attr
  return None

def children(op):
  return set(op for out in op.outputs for op in out.consumers())

def dependencies(roots):
  working = roots
  op_set = set()
  while len(working) > 0:
    o = working[0]
    if type(o) is ops.Tensor:
      o = o.op
    working = working[1:]
    op_set.add(o)
    working += [t.op for t in o.inputs if t.op not in op_set]
    working += [c for c in o.control_inputs if c not in op_set]
  return op_set

def convert_ops_to_nx(fwd_ops, bwd_ops=None):
  bwd_inputs = [t for op in bwd_ops for t in op.inputs]
  graph = nx.DiGraph()
  dictionary = dict()
  variables_seen=[]
  for op in fwd_ops:
    if op.type == 'ReadVariableOp' and op.inputs[0].name not in variables_seen:
      parameter_mem = np.sum([tensor_memory_use(t) for t in op.outputs])
      variables_seen.append(op.inputs[0].name)
    else:
      parameter_mem = 0
    bwd_links = [t for t in op.outputs if t in bwd_inputs]
    if bwd_links != [] and op.type != 'ReadVariableOp':
      saved_mem = np.sum([tensor_memory_use(t) for t in bwd_links])
    else:
      saved_mem = 0
    bwd_links = {t.name: {'size': tensor_memory_use(t),
                          'shape': t.shape.as_list()} for t in bwd_links}
    has_bwd_links = bwd_links != {}
    graph.add_node(op.name, bwd_links=bwd_links, saved_mem=saved_mem,
                   has_bwd_links=has_bwd_links, parameter_mem=parameter_mem)
    dictionary[op.name] = op

  for op in fwd_ops:
    for c_op in children(op):
      if c_op in fwd_ops:
        graph.add_edges_from([(op.name, c_op.name)])

  return graph, dictionary

def calculate_memory(graph, nodes):
  total_mem = 0
  for n in nodes:
    total_mem += graph.nodes[n]['saved_mem']
    total_mem += graph.nodes[n]['parameter_mem']
  return total_mem

def set_ipu_shard(op, index):
  proto = xla_data_pb2.OpSharding(
    type=xla_data_pb2.OpSharding.MAXIMAL, tile_assignment_devices=[index])

  attr_value = attr_value_pb2.AttrValue(s=proto.SerializeToString())
  op._set_attr(so._XLA_SHARDING, attr_value)

def is_splitting_edge(G_fwd, edge, input_node, output_node):
  G = nx.DiGraph(G_fwd)
  G.remove_edge(edge[0], edge[1])
  W = list(nx.weakly_connected_components(G))
  if len(W)==2:
    if not any([input_node in c and output_node in c for c in W]):
      return True
  return False

def find_all_subgraphs(graph, splitting_edges, input_node, output_node):
  graph = nx.DiGraph(graph)
  for edge in splitting_edges:
    graph.remove_edge(edge[0], edge[1])
  W = list(nx.weakly_connected_components(graph))

  subgraphs = []
  edges = []
  next_node = input_node
  while len(W)>0:
    index = next((i for i, w in enumerate(W) if next_node in w))
    sg = W.pop(index)
    subgraphs += [graph.subgraph(sg)]
    if len(W)>0:
      #find edge in subgraph
      edge = next((e for e in splitting_edges if e[0] in sg))
      next_node = edge[1]
      edges += [edge]

  assert output_node in sg, "output node must be in final subgraph"

  return subgraphs, edges

def automatic_sharding(num_shards, input_ts, loss_ts, train_ops=None):
  """Automatically set shards for all connected nodes in graph.

  Args:
    num_shards: number of shards to split graph over
    input_ts: tensor closest to the datafeed in graph
    loss_ts: tensor closest to the loss in graph
    train_ops: an operation or list of operations which are returned by
               Optimizer.minimize()
  """

  loss_op = loss_ts.op

  roots = [loss_op]
  if train_ops:
    roots += train_ops

  op_list = list(filter(lambda o : 'IPU' in o.device, dependencies(roots)))

  fwd_ops = []
  bwd_ops = []

  assert len(op_list) > 0

  for op in op_list:
    op_name = str(op.name.lower())
    if 'gradient' in op_name:
      bwd_ops.append(op)
    else:
      fwd_ops.append(op)

  fwd_ops = list(fwd_ops)
  bwd_ops = list(bwd_ops)

  if input_ts.op not in fwd_ops:
    input_op = [op for op in input_ts.consumers() if op in fwd_ops][0]
  else:
    input_op = input_ts

  graph, dictionary = convert_ops_to_nx(fwd_ops, bwd_ops)

  # check graph is a single weakly connected component
  # if not find the component with the loss op in and use that
  W = list(nx.weakly_connected_components(graph))
  if len(W)>1:
    for g in W:
      if loss_op.name in g:
        graph = graph.subgraph(g)


    fwd_ops = [op for op in fwd_ops if op.name in graph.nodes]

  assert nx.number_weakly_connected_components(graph)==1

  graph_fwd = graph.subgraph([op.name for op in fwd_ops])

  # find all graph edges that split the graph into two subgraphs where the input
  # and output are not in the same subgraph
  splitting_edges = [edge for edge in graph_fwd.edges
                     if is_splitting_edge(graph_fwd, edge, input_op.name,
                                          loss_op.name)]

  # given the splitting edges found find all of the subgraphs created and order
  # them
  subgraphs, edges = find_all_subgraphs(graph_fwd, splitting_edges,
                                        input_op.name, loss_op.name)


  subgraph_mem = [calculate_memory(graph_fwd, g) for g in subgraphs]

  # Split the ordered subgraphs into n groups and calulate the memory for each
  # possible combination
  #
  # Choose the best grouping based on:
  #       1. min max memory
  #       2. variance of memory
  # could use minimum data transfered between IPUs?
  min_max_mem = np.inf
  for ind in itertools.combinations(range(len(edges)), num_shards-1):
    ind_pad = [0] + [i+1 for i in ind] + [len(subgraph_mem)]
    mem = [np.sum(subgraph_mem[ind_pad[i]:ind_pad[i+1]])
           for i in range(len(ind)+1)]
    max_mem = np.max(mem)
    if max_mem < min_max_mem:
      best_ind = [ind]
      best_mem = [mem]
      min_max_mem = max_mem
    elif max_mem == min_max_mem:
      best_ind += [ind]
      best_mem += [mem]

  min_var = np.inf
  for ind, mem in zip(best_ind, best_mem):
    var_mem = np.var(mem)
    if var_mem < min_var:
      best_ind = [ind]
      best_mem = [mem]
      min_var = var_mem
    elif var_mem == min_var:
      best_ind += [ind]
      best_mem += [mem]

  # if still tied choose the first option in the list
  best_ind = best_ind[0]

  ind_pad = [0] + [i+1 for i in best_ind] + [len(subgraph_mem)]
  per_shard_subgraphs = [graph_fwd.subgraph([g0 for g in
                                             subgraphs[ind_pad[i]:ind_pad[i+1]]
                                             for g0 in g.nodes])
                         for i in range(len(ind)+1)]

  for op in fwd_ops:
    shard_set = False
    for i, g in enumerate(per_shard_subgraphs):
      if op.name in g:
        set_ipu_shard(op, i)
        shard_set = True
    assert shard_set, "%s not in any graph split" % op.name

  tf_graph = loss_ts.graph

  for op in filter(lambda o : o not in fwd_ops, op_list):
    attr = get_shard_from_colocation(op)
    if not attr:
      for child in children(op):
        attr = get_shard_from_colocation(child)

    if attr:
      op._set_attr(so._XLA_SHARDING, attr_value_pb2.AttrValue(s=attr))
