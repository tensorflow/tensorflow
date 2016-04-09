# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Tests for contrib.copy_graph.python.util.copy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import tensor_util


class CopyVariablesTest(tf.test.TestCase):

    def testVariableCopy(self):
        graph1 = tf.Graph()
        graph2 = tf.Graph()

        with graph1.as_default():
            #Define a Variable in graph1
            some_var = tf.Variable(2)
            #Initialize the Variable
            self.test_session.run(tf.initialize_all_variables())

        #Make a copy of some_var in the defsult scope in graph2
        copy1 = tf.contrib.copy_graph.copy_variable_to_graph(
            some_var, graph2)

        #Make another copy with different scope
        copy2 = tf.contrib.copy_graph.copy_variable_to_graph(
            some_var, graph2, "test_scope")

        #Initialize both the copies
        with graph2.as_default():
            self.test_session.run(tf.initialize_all_variables())

        #Ensure values in all three variables are the same
        v1 = self.test_session.run(some_var)
        v2 = self.test_session.run(copy1)
        v3 = self.test_session.run(copy2)

        assert v1 == v2 == v3 == 2
