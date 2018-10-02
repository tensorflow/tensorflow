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
"""Run the python doc generator and fail if there are any broken links."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import textwrap

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import googletest
from tensorflow.python.platform import resource_loader
from tensorflow.tools.docs import generate_lib


class Flags(object):
  resource_root = resource_loader.get_root_dir_with_all_resources()
  src_dir = os.path.join(googletest.GetTempDir(), 'input')
  os.mkdir(src_dir)
  base_dir = os.path.join(resource_root, 'tensorflow/')
  output_dir = os.path.join(googletest.GetTempDir(), 'output')
  os.mkdir(output_dir)


class BuildDocsTest(googletest.TestCase):

  def testBuildDocs(self):
    doc_generator = generate_lib.DocGenerator()

    doc_generator.set_py_modules([('tf', tf), ('tfdbg', tf_debug)])

    try:
      status = doc_generator.build(Flags())
    except RuntimeError as e:
      if not e.args[0].startswith('Modules nested too deep'):
        raise

      msg = textwrap.dedent("""\
          %s

          ****************************************************************
          If this test fails here, you have most likely introduced an
          unsealed module. Make sure to use `remove_undocumented` or similar
          utilities to avoid leaking symbols. See above for more information
          on the exact point of failure.
          ****************************************************************
          """ % e.args[0])

      raise RuntimeError(msg)

    if status:
      self.fail('Found %s Errors!' % status)


if __name__ == '__main__':
  googletest.main()
