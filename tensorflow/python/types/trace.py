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
"""Function Tracing Type."""

import abc
from typing import Optional, Sequence


class TraceType(abc.ABC):
  """Represents the type of object(s) for Function Tracing purposes.

  `TraceType` is an abstract class that other classes might inherit from to
  provide information regarding associated class(es) for the purposes of
  Function Tracing. The typing logic provided through this mechanism will be
  used to make decisions regarding usage of cached functions and retracing.
  """

  @abc.abstractmethod
  def is_subtype_of(self, other: "TraceType") -> bool:
    pass

  @abc.abstractmethod
  def most_specific_common_supertype(
      self, others: Sequence["TraceType"]) -> Optional["TraceType"]:
    pass

  @abc.abstractmethod
  def __hash__(self) -> int:
    pass

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    pass
