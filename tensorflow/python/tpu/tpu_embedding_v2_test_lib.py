# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Library module for TPU Embedding mid level API test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops import init_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2_utils


class EmbeddingTestBase(test.TestCase):
  """Base embedding test class for use on CPU and TPU."""

  def _create_initial_data(self):
    """Create the common test data used by both TPU and CPU."""

    self.embedding_values = np.array(list(range(32)), dtype=np.float64)
    self.initializer = init_ops_v2.Constant(self.embedding_values)
    # Embedding for video initialized to
    # 0 1 2 3
    # 4 5 6 7
    # ...
    self.table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=8,
        dim=4,
        initializer=self.initializer,
        combiner='sum',
        name='video')
    # Embedding for user initialized to
    # 0 1
    # 2 3
    # 4 5
    # 6 7
    # ...
    self.table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=16,
        dim=2,
        initializer=self.initializer,
        combiner='mean',
        name='user')
    self.feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched'),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='favorited'),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, name='friends'))

    self.batch_size = 2
    self.data_batch_size = 4

    # One (global) batch of inputs
    # sparse tensor for watched:
    # row 0: 0
    # row 1: 0, 1
    # row 2: 0, 1
    # row 3: 1
    self.feature_watched_indices = [[0, 0], [1, 0], [1, 1],
                                    [2, 0], [2, 1], [3, 0]]
    self.feature_watched_values = [0, 0, 1, 0, 1, 1]
    self.feature_watched_row_lengths = [1, 2, 2, 1]
    # sparse tensor for favorited:
    # row 0: 0, 1
    # row 1: 1
    # row 2: 0
    # row 3: 0, 1
    self.feature_favorited_indices = [[0, 0], [0, 1], [1, 0],
                                      [2, 0], [3, 0], [3, 1]]
    self.feature_favorited_values = [0, 1, 1, 0, 0, 1]
    self.feature_favorited_row_lengths = [2, 1, 1, 2]
    # sparse tensor for friends:
    # row 0: 3
    # row 1: 0, 1, 2
    # row 2: 3
    # row 3: 0, 1, 2
    self.feature_friends_indices = [[0, 0], [1, 0], [1, 1], [1, 2],
                                    [2, 0], [3, 0], [3, 1], [3, 2]]
    self.feature_friends_values = [3, 0, 1, 2, 3, 0, 1, 2]
    self.feature_friends_row_lengths = [1, 3, 1, 3]
