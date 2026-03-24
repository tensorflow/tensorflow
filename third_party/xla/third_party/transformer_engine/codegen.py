# Copyright 2025 The OpenXLA Authors.
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
"""Codegen script for Transformer Engine."""

from absl import app
from absl import flags

_TEMPLATE_FILE = flags.DEFINE_string(
    'template_file', None, 'Path to the template file.', required=True
)
_DATA_FILE = flags.DEFINE_string(
    'data_file', None, 'Path to the data file.', required=True
)
_STRING_NAME = flags.DEFINE_string(
    'string_name', None, 'String name to use in the template.', required=True
)


def main(_):
  with open(_TEMPLATE_FILE.value, 'rt') as f, open(_DATA_FILE.value, 'rt') as g:
    template = f.read()
    data = g.read()
  template = template.replace('@STRING_NAME@', _STRING_NAME.value)
  template = template.replace('@STRING@', data)
  print(template)


if __name__ == '__main__':
  app.run(main)
