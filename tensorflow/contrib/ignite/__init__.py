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
"""IgniteDataset that allows to get data from Apache Ignite.

Apache Ignite is a memory-centric distributed database, caching, and
processing platform for transactional, analytical, and streaming workloads,
delivering in-memory speeds at petabyte scale. This contrib package
contains an integration between Apache Ignite and TensorFlow. The
integration is based on tf.data from TensorFlow side and Binary Client
Protocol from Apache Ignite side. It allows to use Apache Ignite as a
datasource for neural network training, inference and all other
computations supported by TensorFlow. Ignite Dataset is based on Apache
Ignite Binary Client Protocol:
https://apacheignite.readme.io/v2.6/docs/binary-client-protocol.

@@IgniteDataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.ignite.python.ops.ignite_dataset_ops import IgniteDataset
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "IgniteDataset",
]

remove_undocumented(__name__)
