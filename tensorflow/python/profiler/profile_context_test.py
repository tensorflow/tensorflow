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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.profiler import option_builder

# pylint: disable=g-bad-import-order
from tensorflow.python.profiler import profile_context
from tensorflow.python.profiler.internal import model_analyzer_testlib as lib

builder = option_builder.ProfileOptionBuilder


class ProfilerContextTest(test.TestCase):

  def testBasics(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), "dump")
    opts = builder(builder.time_and_memory()
                  ).with_file_output(outfile).build()

    x = lib.BuildFullModel()

    profile_str = None
    profile_step50 = os.path.join(test.get_temp_dir(), "profile_50")
    with profile_context.ProfileContext(test.get_temp_dir()) as pctx:
      pctx.add_auto_profiling("op", options=opts, profile_steps=[15, 50, 100])
      with session.Session() as sess:
        sess.run(variables.global_variables_initializer())
        total_steps = 101 if test.is_gpu_available() else 50
        for i in range(total_steps):
          sess.run(x)
          if i == 14 or i == 99:
            self.assertTrue(gfile.Exists(outfile))
            gfile.Remove(outfile)
          if i == 49:
            self.assertTrue(gfile.Exists(profile_step50))
            with gfile.Open(outfile, "r") as f:
              profile_str = f.read()
            gfile.Remove(outfile)

    with lib.ProfilerFromFile(
        os.path.join(test.get_temp_dir(), "profile_50")) as profiler:
      profiler.profile_operations(options=opts)
      with gfile.Open(outfile, "r") as f:
        self.assertEqual(profile_str, f.read())


if __name__ == "__main__":
  test.main()
