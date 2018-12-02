# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ignite File System for checkpointing and communication with TensorBoard.

Apache Ignite is a memory-centric distributed database, caching, and
processing platform for transactional, analytical, and streaming workloads,
delivering in-memory speeds at petabyte scale. In addition to database
functionality Apache Ignite provides a distributed file system called
IGFS (https://ignite.apache.org/features/igfs.html). IGFS delivers a similar
functionality to Hadoop HDFS, but only in-memory. In fact, in addition to
its own APIs, IGFS implements Hadoop FileSystem API and can be transparently
plugged into Hadoop or Spark deployments. This contrib package contains an
integration between IGFS and TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.ignite.python.ops import ignite_op_loader  # pylint: disable=unused-import
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

file_system_library = os.path.join(resource_loader.get_data_files_path(),
                                   "../../_ignite_ops.so")
load_library.load_file_system_library(file_system_library)
