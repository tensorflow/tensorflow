# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Binary for extracting API information for a set of Python sources."""
from collections.abc import Sequence

from absl import app
from absl import flags

from tensorflow.python.tools.api.generator2.extractor import parser
from tensorflow.python.tools.api.generator2.shared import exported_api

_OUTPUT = flags.DEFINE_string("output", "", "File to output contents to.")
_DECORATOR = flags.DEFINE_string(
    "decorator",
    "",
    "Full path to Python decorator function used for exporting API.",
)
_API_NAME = flags.DEFINE_string(
    "api_name",
    "",
    "Prefix for all exported symbols and docstrings.",
)


def main(argv: Sequence[str]) -> None:
  exporter = exported_api.ExportedApi()
  p = parser.Parser(exporter, _DECORATOR.value, _API_NAME.value)
  for arg in argv[1:]:
    p.process_file(arg)

  exporter.write(_OUTPUT.value)


if __name__ == "__main__":
  app.run(main)
