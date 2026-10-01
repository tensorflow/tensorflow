# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the pip package builder."""

import json
import os
import tempfile
import unittest
from unittest import mock

from tensorflow.tools.pip_package import build_pip_package


class BuildPipPackageTest(unittest.TestCase):

  def test_build_wheel_collaborator_environment(self):
    cases = (
        (None, None, None),
        (None, "inherited", "inherited"),
        ("False", None, None),
        ("False", "inherited", "inherited"),
        ("True", None, "True"),
        ("True", "inherited", "True"),
    )
    for collab, inherited, expected in cases:
      with self.subTest(collab=collab, inherited=inherited):
        with tempfile.TemporaryDirectory() as cwd, mock.patch.dict(os.environ):
          os.environ.pop("collaborator_build", None)
          if inherited is not None:
            os.environ["collaborator_build"] = inherited
          setup_path = os.path.join(
              cwd, "tensorflow", "tools", "pip_package", "setup.py"
          )
          os.makedirs(os.path.dirname(setup_path))
          with open(setup_path, "w", encoding="utf-8") as setup_py:
            setup_py.write(
                "import json\n"
                "import os\n"
                'with open("environment.json", "w", encoding="utf-8") as f:\n'
                '  json.dump(os.environ.get("collaborator_build"), f)\n'
            )
          kwargs = {} if collab is None else {"collab": collab}
          build_pip_package.build_wheel(
              dir_path=os.path.join(cwd, "dist"),
              cwd=cwd,
              project_name="tensorflow_test",
              platform="test_platform",
              **kwargs,
          )
          with open(
              os.path.join(cwd, "environment.json"), encoding="utf-8"
          ) as output:
            self.assertEqual(expected, json.load(output))


if __name__ == "__main__":
  unittest.main()
