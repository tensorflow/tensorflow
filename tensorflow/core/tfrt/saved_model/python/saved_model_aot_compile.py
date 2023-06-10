#  Copyright 2023 The TensorFlow Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test .py file for pybind11 files for AotOptions and AotCompileSavedModel, currently unable to test due to nullptr in AotOptions."""


from absl import app
from tensorflow.core.tfrt.graph_executor.python import _pywrap_graph_execution_options
from tensorflow.core.tfrt.saved_model.python import _pywrap_saved_model_aot_compile


def main(unused_argv):
  if not _pywrap_saved_model_aot_compile:
    return
  try:
    # Test for creating an instance of GraphExecutionOptions
    test = _pywrap_graph_execution_options.GraphExecutionOptions()
    print(test)

    # Executes AoTOptions and AotCompileSavedModel for Wrapping Tests
    _pywrap_saved_model_aot_compile.AotOptions()

    # TODO(cesarmagana): Once AotCompileSavedModel is complete
    # update this test script to read from CNS
    _pywrap_saved_model_aot_compile.AotCompileSavedModel("random")

  # Could also do except status.StatusNotOk if testing for AotCompileSavedModel
  except Exception as exception:  # pylint: disable=broad-exception-caught
    print(exception)


if __name__ == "__main__":
  app.run(main)
