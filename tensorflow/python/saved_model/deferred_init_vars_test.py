# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for model loading without checkpoints."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import save


class DeferredInitModuleVariablesTest(test.TestCase):

  def test_deferred_init_module_variables(self):
    """Defer initialization of variables in a module to the load stage."""

    class MyModule(module.Module):

      def __init__(self, size):
        super().__init__()
        self.size = size
        # variable initialized by a Tensor-compatible value
        self.w1 = variables.Variable(
            constant_op.constant(1., shape=[self.size]), trainable=False)
        # variable initialized by a function
        self.w2 = variables.Variable(
            lambda: constant_op.constant(2., shape=[self.size]))
        # variable instantiated lazily in call()
        self.w3 = None

      def call(self):
        if self.w3 is None:
          self.w3 = variables.Variable(
              constant_op.constant(3., shape=[self.size]))
        for w in (self.w1, self.w2, self.w3):
          w.assign_add(constant_op.constant(1., shape=[self.size]))
        return self.w1, self.w2, self.w3

    def export_initializer(initial_value, export_dir):

      class Initializer(module.Module):

        @def_function.function(input_signature=[])
        def call(self):
          if callable(initial_value):
            return initial_value()
          return initial_value

      save.save(Initializer(), export_dir)

    def create_and_save_module(weight_size):

      initial_values = {}  # For storing initial_value of created variables

      def variable_creator(next_creator, **kwargs):
        variable = next_creator(**kwargs)
        initial_values[variable.name] = kwargs['initial_value']
        return variable

      export_dir = self.create_tempdir().full_path

      with ops.Graph().as_default():
        with variable_scope.variable_creator_scope(variable_creator):
          exported = MyModule(weight_size)
          exported.call = def_function.function(input_signature=[])(
              exported.call)

          module_dir = f'{export_dir}/module'
          file_io.recursive_create_dir(module_dir)
          save.save_and_return_nodes(
              exported, module_dir, experimental_skip_checkpoint=True)

      # Save the initializer of the created variables.
      for variable_name, initial_value in initial_values.items():
        export_initializer(initial_value,
                           f'{export_dir}/variables/{variable_name}')

      return export_dir

    def load_and_run_module(export_dir, weight_size):

      # pylint: disable=unused-argument
      def layer_variable_creator(next_creator, **kwargs):
        variable_dir = f'{export_dir}/variables/{kwargs["name"]}:0'
        initializer = load.load(variable_dir)
        kwargs['initial_value'] = initializer.call
        variable = resource_variable_ops.ResourceVariable(**kwargs)
        return variable

      with ops.Graph().as_default():
        with variable_scope.variable_creator_scope(layer_variable_creator):
          imported = load.load(
              f'{export_dir}/module',
              options=load_options.LoadOptions(
                  experimental_skip_checkpoint=True))
        outputs = imported.call()

        with self.cached_session() as sess:
          variables.global_variables_initializer().run()
          # Check if variables work as expected across multiple iterations.
          for i in range(3):
            np_outputs = sess.run(outputs)
            for j, np_output in enumerate(np_outputs):
              self.assertAllClose(np_output, np.full(weight_size, i + j + 2))

    # The size of the serialized content (both module and variables) stays
    # small even with a large weight_size as the initial values are not stored
    # in checkpoints.
    weight_size = 1024
    export_dir = create_and_save_module(weight_size)
    load_and_run_module(export_dir, weight_size)


if __name__ == '__main__':
  test.main()
