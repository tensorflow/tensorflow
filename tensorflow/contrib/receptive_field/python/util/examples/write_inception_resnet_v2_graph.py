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
"""Simple script to write Inception-ResNet-v2 model to graph file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import app
from nets import inception

cmd_args = None


def main(unused_argv):
  # Model definition.
  g = ops.Graph()
  with g.as_default():
    images = array_ops.placeholder(
        dtypes.float32, shape=(1, None, None, 3), name='input_image')
    inception.inception_resnet_v2_base(images)

  graph_io.write_graph(g.as_graph_def(), cmd_args.graph_dir,
                       cmd_args.graph_filename)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--graph_dir',
      type=str,
      default='/tmp',
      help='Directory where graph will be saved.')
  parser.add_argument(
      '--graph_filename',
      type=str,
      default='graph.pbtxt',
      help='Filename of graph that will be saved.')
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
