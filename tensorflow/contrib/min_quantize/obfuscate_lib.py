from __future__ import absolute_import, division, print_function

import io
import re

from google.protobuf.text_format import MessageToString, Parse
from tensorflow.contrib.min_quantize.quantized_pb2 import QuantizedGraph, QuantizedItem
from tensorflow.core.framework.graph_pb2 import GraphDef

__all__ = [
  'obfuscate_graph_def',
  'obfuscate_quantized_graph',
]


def obfuscate_graph_def(graph_def, keeps=None):
  """
  obfuscate node names in graph
  :param graph_def: graph to obfuscate
  :param keeps: list of node name or (node_name, target_name) tuple
  :return: (obfuscated graph def, mapping)
  """
  mapper = _NodeNameMapper(keeps)
  for node in graph_def.node:
    mapper.map(node.name)
  return _replace_graph_node_names(graph_def, mapper.mapping), mapper.mapping


def obfuscate_quantized_graph(quantized_graph, keeps=None):
  """
  obfuscate node names in graph
  :param quantized_graph: QuantizedGraph
  :param keeps: list of node name or (node_name, target_name) tuple
  :return: (obfuscated quantized graph, mapping)
  """
  graph, mapping = obfuscate_graph_def(quantized_graph.graph, keeps)
  items = []
  for item in quantized_graph.items:
    name = mapping.get(item.name, None)
    if name is None:
      items.append(item)
    else:
      new_item = QuantizedItem()
      new_item.CopyFrom(item)
      new_item.name = name
      items.append(new_item)
  new_quantized_graph = QuantizedGraph()
  new_quantized_graph.graph.CopyFrom(graph)
  new_quantized_graph.items.extend(items)
  return new_quantized_graph, mapping


_node_name_regex_tpl = r'(s: "loc:@|input: "|name: "\^?)(?P<name>{})[:"]'


def _replace_graph_node_names(graph, mapping):
  # get all nodes, sort by node name length
  all_nodes = [node.name for node in graph.node if len(node.name) > 0]
  all_nodes.sort(key=lambda k: len(k), reverse=True)

  # regex, match all node name
  all_nodes_regex = re.compile(_node_name_regex_tpl.format('|'.join(all_nodes)))

  # old graph text
  graph_text = MessageToString(graph)

  # replace all node name
  obfuscated_graph_text = io.StringIO()
  last_match_end = 0
  while True:
    match = all_nodes_regex.search(graph_text, last_match_end)
    if match is None:
      break

    # prefix
    match_beg, match_end = match.span('name')
    obfuscated_graph_text.write(graph_text[last_match_end:match_beg])
    last_match_end = match_end

    # node name
    node_name = graph_text[match_beg:match_end]
    obfuscated_graph_text.write(mapping.get(node_name, node_name))

  obfuscated_graph_text.write(graph_text[last_match_end:])

  obfuscated_graph = GraphDef()
  Parse(obfuscated_graph_text.getvalue(), obfuscated_graph)
  obfuscated_graph_text.close()
  return obfuscated_graph


class _NodeNameMapper:
  def __init__(self, keeps, pool=None):
    self._used_names = set()
    self._mapping = self._init_mapping(keeps)
    self._seed = 0
    self._pool = _NodeNameMapper._default_pool() if pool is None else pool

  @property
  def mapping(self):
    return self._mapping

  def map(self, name):
    if name in self._mapping:
      return self._mapping[name]

    obfuscated_name = self._next_valid_name()
    self._mapping[name] = obfuscated_name
    return obfuscated_name

  @staticmethod
  def _default_pool():
    lower = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    upper = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    num = [chr(c) for c in range(ord('0'), ord('9') + 1)]
    return lower + upper + num

  @staticmethod
  def _init_mapping(keeps):
    mapping = {}
    if keeps is None:
      return mapping
    for item in keeps:
      if isinstance(item, str):
        if item in mapping.keys():
          raise RuntimeError('src {} dup in keeps'.format(item))
        if item in mapping.values():
          raise RuntimeError('dst {} dup in keeps'.format(item))
        mapping[item] = item
      elif isinstance(item, tuple):
        if len(item) != 2:
          raise RuntimeError('{} length should be 2'.format(item))
        if not isinstance(item[0], str) or not isinstance(item[1], str):
          raise RuntimeError('{} should be tuple of str'.format(item))
        if item[0] in mapping.keys():
          raise RuntimeError('src {} dup in keeps'.format(item[0]))
        if item[1] in mapping.values():
          raise RuntimeError('dst {} dup in keeps'.format(item[1]))
        mapping[item[0]] = item[1]
    return mapping

  def _next_name(self):
    seed = self._seed
    self._seed += 1

    pool_size = len(self._pool)
    s = io.StringIO()
    while True:
      s.write(self._pool[seed % pool_size])
      seed //= pool_size
      if seed == 0:
        break
    name = s.getvalue()
    s.close()
    return name

  def _next_valid_name(self):
    while True:
      name = self._next_name()
      if name in self._used_names:
        continue
      return name

  def _add_map(self, src, dst):
    self._mapping[src] = dst
    self._used_names.add(dst)
