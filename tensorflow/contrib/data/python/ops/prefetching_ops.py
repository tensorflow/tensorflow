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
"""Python wrapper for prefetching_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import gen_prefetching_ops
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_prefetching_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("../../_prefetching_ops.so"))


# TODO(rohanj): Add a python class that constructs resource in the __init__
# method and provides a get_next() that calls the prefetch op.
def function_buffering_resource(string_arg,
                                target_device,
                                shared_name,
                                f,
                                buffer_size,
                                thread_pool_size=1,
                                container="",
                                name=None):
  return gen_prefetching_ops.function_buffering_resource(
      string_arg=string_arg,
      target_device=target_device,
      shared_name=shared_name,
      f=f,
      buffer_size=buffer_size,
      thread_pool_size=thread_pool_size,
      container=container,
      name=name)


def function_buffering_resource_get_next(function_buffer_resource,
                                         output_types,
                                         name=None):
  return gen_prefetching_ops.function_buffering_resource_get_next(
      function_buffer_resource=function_buffer_resource,
      output_types=output_types,
      name=name)
