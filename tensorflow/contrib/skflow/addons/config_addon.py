"""Configuration Addon."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import division, print_function, absolute_import

import tensorflow as tf

class ConfigAddon(object):
    """This class specifies the specific configurations for a session.

    Parameters:
        num_cores: Number of cores to be used. (default: 4)
        verbose: Controls the verbosity, possible values:
                 0: the algorithm and debug information is muted.
                 1: trainer prints the progress.
                 2: log device placement is printed.
        gpu_memory_fraction: Fraction of GPU memory used by the process on
            each GPU uniformly on the same machine.
   """

    def __init__(self, num_cores=4, verbose=1, gpu_memory_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        self.config = tf.ConfigProto(log_device_placement=(verbose > 1),
                                     inter_op_parallelism_threads=num_cores,
                                     intra_op_parallelism_threads=num_cores,
                                     gpu_options=gpu_options)
    