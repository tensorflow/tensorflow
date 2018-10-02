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
"""Tests for contrib.copy_graph.python.util.copy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.copy_graph.python.util import copy_elements
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

graph1 = ops.Graph()
graph2 = ops.Graph()


class CopyVariablesTest(test.TestCase):

  def testVariableCopy(self):

    with graph1.as_default():
      #Define a Variable in graph1
      some_var = variables.VariableV1(2)
      #Initialize session
      sess1 = session_lib.Session()
      #Initialize the Variable
      variables.global_variables_initializer().run(session=sess1)

    #Make a copy of some_var in the defsult scope in graph2
    copy1 = copy_elements.copy_variable_to_graph(some_var, graph2)

    #Make another copy with different scope
    copy2 = copy_elements.copy_variable_to_graph(some_var, graph2, "test_scope")

    #Initialize both the copies
    with graph2.as_default():
      #Initialize Session
      sess2 = session_lib.Session()
      #Initialize the Variables
      variables.global_variables_initializer().run(session=sess2)

    #Ensure values in all three variables are the same
    v1 = some_var.eval(session=sess1)
    v2 = copy1.eval(session=sess2)
    v3 = copy2.eval(session=sess2)

    assert isinstance(copy1, variables.Variable)
    assert isinstance(copy2, variables.Variable)
    assert v1 == v2 == v3 == 2


class CopyOpsTest(test.TestCase):

  def testOpsCopy(self):

    with graph1.as_default():
      #Initialize a basic expression y = ax + b
      x = array_ops.placeholder("float")
      a = variables.VariableV1(3.0)
      b = constant_op.constant(4.0)
      ax = math_ops.multiply(x, a)
      y = math_ops.add(ax, b)
      #Initialize session
      sess1 = session_lib.Session()
      #Initialize the Variable
      variables.global_variables_initializer().run(session=sess1)

    #First, initialize a as a Variable in graph2
    a1 = copy_elements.copy_variable_to_graph(a, graph2)

    #Initialize a1 in graph2
    with graph2.as_default():
      #Initialize session
      sess2 = session_lib.Session()
      #Initialize the Variable
      variables.global_variables_initializer().run(session=sess2)

    #Initialize a copy of y in graph2
    y1 = copy_elements.copy_op_to_graph(y, graph2, [a1])

    #Now that y has been copied, x must be copied too.
    #Get that instance
    x1 = copy_elements.get_copied_op(x, graph2)

    #Compare values of y & y1 for a sample input
    #and check if they match
    v1 = y.eval({x: 5}, session=sess1)
    v2 = y1.eval({x1: 5}, session=sess2)

    assert v1 == v2


if __name__ == "__main__":
  test.main()
