# Copyright 2017 Graphcore Ltd
#

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
# =============================================================================
"""
A driver for the Graphcore IPU device, interfacing through the poplar library.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging

def load_poplar():
    """Loads the Tensorflow poplar plugin
    """
    try:
        filename = resource_loader.get_path_to_datafile('libpoplar_plugin.so')
        load_library.load_op_library(filename)
    except errors.NotFoundError:
        logging.warning('%s file could not be loaded.', filename)

