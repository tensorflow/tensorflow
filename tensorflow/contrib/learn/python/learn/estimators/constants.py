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

"""Constants regarding Estimators (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ProblemType(object):
  """Enum-like values for the type of problem that the model solves.

  THIS CLASS IS DEPRECATED.

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


# CollectionDef key for the input feature keys.
# TODO(b/34388557): This is a stopgap; please follow the bug to learn of changes
COLLECTION_DEF_KEY_FOR_INPUT_FEATURE_KEYS = "input_feature_keys"
