# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Simple binary for sleep."""

import math
import sys
import time

from absl import app
import numpy as np
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.sleep import sleep_op


def stack50(op, delay):
  """Create a tf.stack of 50 sleep ops.

  Args:
    op: The sleep op, either sleep_op.SyncSleep or sleep_op.AsyncSleep.
    delay: Each op should finish at least float `delay` seconds after it starts.
  """
  n = 50
  delays = delay + tf.range(0, n, dtype=float) / 10000.0
  start_t = time.time()
  func = tf.function(lambda: tf.stack([op(delays[i]) for i in range(n)]))
  r_numpy = func().numpy()
  end_t = time.time()
  print('')
  print('Total time = %5.3f seconds using %s' % (end_t - start_t, str(op)))
  print('Returned values from the ops:')
  np.set_printoptions(precision=4, suppress=True)
  print(r_numpy)
  sys.stdout.flush()


def main(argv):
  del argv  # not used
  delay_seconds = 1.0
  print("""
Using synchronous sleep op with each of 50 ops sleeping for about %0.2f seconds,
so total time is about %0.2f * ceil(50 / NUMBER_OF_THREADS). 16 is a typical
number of threads, which would be %0.2f seconds. The actual time will be
a little greater.
""" % (delay_seconds, delay_seconds, delay_seconds * math.ceil(50.0 / 16.0)))
  stack50(sleep_op.SyncSleep, delay_seconds)

  print("""
Using asynchronous sleep op with each of 50 ops sleeping only as much as
necessary so they finish after at least %0.2f seconds. Time that
an op spends blocked waiting to finish counts as all or part of its delay.
The returned values show how long each ops sleeps or 0 if the op does not
need to sleep. The expected total time will be a little greater than
the requested delay of %0.2f seconds.
""" % (delay_seconds, delay_seconds))
  stack50(sleep_op.AsyncSleep, delay_seconds)


if __name__ == '__main__':
  app.run(main)
