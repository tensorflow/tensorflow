# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Container for origin source code information before AutoGraph compilation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple


class CodeLocation(namedtuple('CodeLocation', ('file_path', 'line_number'))):
  """Location of a line of code.

  Attributes:
    file_path: text, the full path to the file containing the code.
    line_number: Int, the 1-based line number of the code in its file.
  """
  pass


class OriginInfo(
    namedtuple('OriginInfo', ('file_path', 'function_name', 'line_number',
                              'column_offset', 'source_code_line'))):
  """Container for information about the source code before conversion.

  Instances of this class contain information about the source code that
  transformed code originated from. Examples include:
    * line number
    * file name
    * original user code
  """

  def as_frame(self):
    """Makes a traceback frame tuple.

    Returns:
      A tuple of (file_path, line_number, function_name, source_code_line).
    """
    return (self.file_path, self.line_number, self.function_name,
            self.source_code_line)
