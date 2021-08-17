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
r"""Prints a header file to be used with SELECTIVE_REGISTRATION.

An example of command-line usage is:
  bazel build tensorflow/python/tools:print_selective_registration_header && \
  bazel-bin/tensorflow/python/tools/print_selective_registration_header \
    --graphs=path/to/graph.pb > ops_to_register.h

Then when compiling tensorflow, include ops_to_register.h in the include search
path and pass -DSELECTIVE_REGISTRATION and -DSUPPORT_SELECTIVE_REGISTRATION
 - see core/framework/selective_registration.h for more details.

When compiling for Android:
  bazel build -c opt --copt="-DSELECTIVE_REGISTRATION" \
    --copt="-DSUPPORT_SELECTIVE_REGISTRATION" \
    //tensorflow/tools/android/inference_interface:libtensorflow_inference.so \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --crosstool_top=//external:android/crosstool --cpu=armeabi-v7a
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from absl import app
from tensorflow.python.tools import selective_registration_header_lib

FLAGS = None


def main(unused_argv):
  graphs = FLAGS.graphs.split(',')
  print(
      selective_registration_header_lib.get_header(graphs,
                                                   FLAGS.proto_fileformat,
                                                   FLAGS.default_ops))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--graphs',
      type=str,
      default='',
      help='Comma-separated list of paths to model files to be analyzed.',
      required=True)
  parser.add_argument(
      '--proto_fileformat',
      type=str,
      default='rawproto',
      help='Format of proto file, either textproto, rawproto or ops_list. The '
      'ops_list is the file contains the list of ops in JSON format. Ex: '
      '"[["Add", "BinaryOp<CPUDevice, functor::add<float>>"]]".')
  parser.add_argument(
      '--default_ops',
      type=str,
      default='NoOp:NoOp,_Recv:RecvOp,_Send:SendOp',
      help='Default operator:kernel pairs to always include implementation for.'
      'Pass "all" to have all operators and kernels included; note that this '
      'should be used only when it is useful compared with simply not using '
      'selective registration, as it can in some cases limit the effect of '
      'compilation caches')

  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
