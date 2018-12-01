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
"""API class for dense (approximate) kernel mappers.

See ./random_fourier_features.py for a concrete instantiation of this class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six


class InvalidShapeError(Exception):
  """Exception thrown when a tensor's shape deviates from an expected shape."""


@six.add_metaclass(abc.ABCMeta)
class DenseKernelMapper(object):
  """Abstract class for a kernel mapper that maps dense inputs to dense outputs.

  This class is abstract. Users should not create instances of this class.
  """

  @abc.abstractmethod
  def map(self, input_tensor):
    """Main Dense-Tensor-In-Dense-Tensor-Out (DTIDTO) map method.

    Should be implemented by subclasses.
    Args:
      input_tensor: The dense input tensor to be mapped using the (approximate)
      kernel mapper.
    """
    raise NotImplementedError('map is not implemented for {}.'.format(self))

  @abc.abstractproperty
  def name(self):
    """Returns the name of the kernel mapper."""
    pass

  @abc.abstractproperty
  def output_dim(self):
    """Returns the output dimension of the mapping."""
    pass
