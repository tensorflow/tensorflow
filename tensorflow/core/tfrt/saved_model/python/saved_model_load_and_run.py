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

"""Test .py file for pybind11 files for SavedModelImpl functions LoadSvaedModel & Run."""

from absl import app
from tensorflow.core.tfrt.saved_model.python import _pywrap_saved_model


def main(unused_argv):
  if not _pywrap_saved_model:
    return
  try:
    # Try to run Load and Run functions
    _pywrap_saved_model.LoadSavedModel()
    _pywrap_saved_model.Run(_pywrap_saved_model.LoadSavedModel())
    # //TODO(malikys): load real saved_model data for testing

  except Exception as exception:  # pylint: disable=broad-exception-caught
    print(exception)


if __name__ == "__main__":
  app.run(main)
