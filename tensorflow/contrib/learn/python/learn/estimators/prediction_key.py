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
"""Enum for model prediction keys (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

This file is obsoleted in the move of Estimator to core.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class PredictionKey(object):
  """THIS CLASS IS DEPRECATED."""

  CLASSES = "classes"
  PROBABILITIES = "probabilities"
  LOGITS = "logits"
  LOGISTIC = "logistic"
  SCORES = "scores"
  TOP_K = "top_k"
  GENERIC = "output"
