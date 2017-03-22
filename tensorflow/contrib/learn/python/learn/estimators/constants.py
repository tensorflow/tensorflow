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

"""Constants regarding Estimators.

This file is obsoleted in the move of Estimator to core.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ProblemType(object):
  """Enum-like values for the type of problem that the model solves.

  These values are used when exporting the model to produce the appropriate
  signature function for serving.

  The following values are supported:
    UNSPECIFIED: Produces a predict signature_fn.
    CLASSIFICATION: Produces a classify signature_fn.
    LINEAR_REGRESSION: Produces a regression signature_fn.
    LOGISTIC_REGRESSION: Produces a classify signature_fn.
  """
  UNSPECIFIED = 0
  CLASSIFICATION = 1
  LINEAR_REGRESSION = 2
  LOGISTIC_REGRESSION = 3
