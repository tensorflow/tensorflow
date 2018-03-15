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
"""Basic loop for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.basic_train_loop")
def basic_train_loop(supervisor, train_step_fn, args=None,
                     kwargs=None, master=""):
  """Basic loop to train a model.

  Calls `train_step_fn` in a loop to train a model.  The function is called as:

  ```python
  train_step_fn(session, *args, **kwargs)
  ```

  It is passed a `tf.Session` in addition to `args` and `kwargs`.  The function
  typically runs one training step in the session.

  Args:
    supervisor: `tf.train.Supervisor` to run the training services.
    train_step_fn: Callable to execute one training step.  Called
      repeatedly as `train_step_fn(session, *args **kwargs)`.
    args: Optional positional arguments passed to `train_step_fn`.
    kwargs: Optional keyword arguments passed to `train_step_fn`.
    master: Master to use to create the training session.  Defaults to
      `""` which causes the session to be created in the local process.
  """
  if args is None:
    args = []
  if kwargs is None:
    kwargs = {}
  should_retry = True
  while should_retry:
    try:
      should_retry = False
      with supervisor.managed_session(master) as sess:
        while not supervisor.should_stop():
          train_step_fn(sess, *args, **kwargs)
    except errors.AbortedError:
      # Always re-run on AbortedError as it indicates a restart of one of the
      # distributed tensorflow servers.
      should_retry = True
