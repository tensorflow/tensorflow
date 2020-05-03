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
"""Experimental API for controlling threading in `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.data.util import options
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.ThreadingOptions")
class ThreadingOptions(options.OptionsBase):
  """Represents options for dataset threading.

  You can set the threading options of a dataset through the
  `experimental_threading` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.ThreadingOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 10
  dataset = dataset.with_options(options)
  ```
  """

  max_intra_op_parallelism = options.create_option(
      name="max_intra_op_parallelism",
      ty=int,
      docstring=
      "If set, it overrides the maximum degree of intra-op parallelism.")

  private_threadpool_size = options.create_option(
      name="private_threadpool_size",
      ty=int,
      docstring=
      "If set, the dataset will use a private threadpool of the given size.")
