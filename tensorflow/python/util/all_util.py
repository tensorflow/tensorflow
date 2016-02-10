# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Generate __all__ from a module docstring."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys

_reference_pattern = re.compile(r'^@@(\w+)$', flags=re.MULTILINE)


def make_all(module_name):
  """Generate `__all__` from a module's docstring.

  Usage: `make_all(__name__)`.  The caller module must have a docstring,
  and `__all__` will contain all symbols with `@@` references.

  Args:
    module_name: The name of the module (usually `__name__`).

  Returns:
    A list suitable for use as `__all__`.
  """
  doc = sys.modules[module_name].__doc__
  return [m.group(1) for m in _reference_pattern.finditer(doc)]


__all__ = ['make_all']
