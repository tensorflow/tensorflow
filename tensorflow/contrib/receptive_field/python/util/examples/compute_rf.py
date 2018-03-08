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
"""Computes Receptive Field (RF) information given a graph protobuf.

For an example of usage, see accompanying file compute_rf.sh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from google.protobuf import text_format

from tensorflow.contrib.receptive_field import receptive_field_api as receptive_field
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

cmd_args = None


def _load_graphdef(path):
  """Helper function to load GraphDef from file.

  Args:
    path: Path to pbtxt file.

  Returns:
    graph_def: A GraphDef object.
  """
  graph_def = graph_pb2.GraphDef()
  pbstr = gfile.Open(path).read()
  text_format.Parse(pbstr, graph_def)
  return graph_def


def main(unused_argv):

  graph_def = _load_graphdef(cmd_args.graph_path)

  (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
   effective_padding_x, effective_padding_y
  ) = receptive_field.compute_receptive_field_from_graph_def(
      graph_def, cmd_args.input_node, cmd_args.output_node)

  logging.info('Receptive field size (horizontal) = %s', receptive_field_x)
  logging.info('Receptive field size (vertical) = %s', receptive_field_y)
  logging.info('Effective stride (horizontal) = %s', effective_stride_x)
  logging.info('Effective stride (vertical) = %s', effective_stride_y)
  logging.info('Effective padding (horizontal) = %s', effective_padding_x)
  logging.info('Effective padding (vertical) = %s', effective_padding_y)

  f = gfile.GFile('%s' % cmd_args.output_path, 'w')
  f.write('Receptive field size (horizontal) = %s\n' % receptive_field_x)
  f.write('Receptive field size (vertical) = %s\n' % receptive_field_y)
  f.write('Effective stride (horizontal) = %s\n' % effective_stride_x)
  f.write('Effective stride (vertical) = %s\n' % effective_stride_y)
  f.write('Effective padding (horizontal) = %s\n' % effective_padding_x)
  f.write('Effective padding (vertical) = %s\n' % effective_padding_y)
  f.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--graph_path', type=str, default='', help='Graph path (pbtxt format).')
  parser.add_argument(
      '--output_path',
      type=str,
      default='',
      help='Path to output text file where RF information will be written to.')
  parser.add_argument(
      '--input_node', type=str, default='', help='Name of input node.')
  parser.add_argument(
      '--output_node', type=str, default='', help='Name of output node.')
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
