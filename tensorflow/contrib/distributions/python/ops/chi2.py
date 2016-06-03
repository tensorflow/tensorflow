# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""The Chi2 distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import gamma
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class Chi2(gamma.Gamma):
  """The Chi2 distribution with degrees of freedom df.

  The PDF of this distribution is:

  ```pdf(x) = (x^(df/2 - 1)e^(-x/2))/(2^(k/2)Gamma(k/2)), x > 0```

  Note that the Chi2 distribution is a special case of the Gamma distribution,
  with Chi2(df) = Gamma(df/2, 1/2).
  """

  def __init__(self, df, name="Chi2"):
    with ops.op_scope([df], name, "init"):
      df = ops.convert_to_tensor(df)
      self._df = df
      super(Chi2, self).__init__(alpha=df / 2,
                                 beta=math_ops.cast(0.5, dtype=df.dtype))

  @property
  def df(self):
    return self._df
