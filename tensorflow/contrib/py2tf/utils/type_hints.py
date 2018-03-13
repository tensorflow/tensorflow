# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""No-op utilities that provide static type hints.

These are used when the data type is not known at creation, for instance in the
case of empty lists.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def set_element_type(entity, dtype, shape=None):
  """Indicates that the entity is expected hold items of specified type.

  This function is a no-op. Its presence merely marks the data type of its
  argument. The staged TensorFlow ops will reflect and assert this data type.

  Args:
    entity: A Tensor or TensorArray.
    dtype: TensorFlow dtype value to assert for entity.
    shape: Optional shape to assert for entity.
  Returns:
    The value of entity, unchanged.
  """
  del dtype
  del shape
  return entity
