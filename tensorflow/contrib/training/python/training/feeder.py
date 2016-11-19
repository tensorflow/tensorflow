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


"""A mechanism to strongly decouple input generation from consumption.

This helper handles the plumbing in order to set up a feeder task to
push generated inputs to a pool of remote consumers; or to run an
identical feeding mechanism in a seperate thread in the same process.

Example usage for distributed feeding:

```
# In the consumer job:
dtypes = [tf.int32, tf.string]
shapes = [[5], []]

with tf.Graph().as_default():
  feeder = tf.contrib.training.Feeder(dtypes, shapes)
  int_inputs, str_inputs = feeder.get_many_fed_tensors(batch_size=10)

  # ... go on to use inputs and a training/eval/etc loop as usual ...


# In the feeder job:
with tf.Graph().as_default():
  input_ints = tf.constant([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
  input_strs = tf.constant(['one_x', 'two_x'])

  # Important: constructor arguments must be the same as in the consumer job!
  feeder = tf.contrib.training.Feeder(dtypes, shapes)

  feeder.set_many_fed_tensors([input_ints, input_strs])

  feeder.add_remote_devices(
      ['/job:consumer/replica:0', '/job:consumer/replica:1'])
  # ...or use the add_remote_replicas helper.

  feeder.run_feeding_forever(lambda: tf.Session(FLAGS.master))
```

For feeding in-process, a Feeder acts similarly to a Queue, with a
QueueRunner automatically registered:

```
dtypes = [tf.int32, tf.string]
shapes = [[5], []]

# ... in main():
with tf.Graph().as_default():
  feeder = tf.contrib.training.Feeder(dtypes, shapes)

  feeder.set_many_fed_tensors([tf.constant([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]]),
                               tf.constant(['one_x', 'two_x'])])

  int_inputs, str_inputs = feeder.get_many_fed_tensors(batch_size=10)

  # ... go on to use inputs and a training/eval/etc loop as usual ...
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import threading

from tensorflow.contrib.training.python.training import failure_tolerator
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner


class Feeder(object):
  """Helper to manage the plumbing for externally-fed graphs."""
  REMOTE_QUEUE_RUNNERS = 'feeder_remote_queue_runners'

  def __init__(
      self, dtypes, shapes=None, capacity=10, shared_name='feeding_queue'):
    self._dtypes = dtypes
    self._shapes = shapes
    self._shared_name = shared_name
    self._capacity = capacity
    self._local_q = data_flow_ops.FIFOQueue(capacity=self._capacity,
                                            dtypes=self._dtypes,
                                            shapes=self._shapes,
                                            name=self._shared_name,
                                            shared_name=self._shared_name)
    self._num_remote_feeds = 0

    # Fake do-nothing operation that's used to prevent remote queues
    # from being closed, and as a workaround for b/32749157
    self._fake_op = array_ops.constant('dummy close', name='feeder_fake_op').op
    self._feeding_event = threading.Event()

  def get_fed_tensors(self):
    """Returns fed tensor values."""
    return self._local_q.dequeue()

  def get_many_fed_tensors(self, batch_size):
    """Returns a batch of fed tensor values."""
    return self._local_q.dequeue_many(batch_size)

  def set_fed_tensors(self, tensors):
    """Sets fed tensors."""
    enq_op = self._local_q.enqueue(tensors)
    queue_runner.add_queue_runner(queue_runner.QueueRunner(
        self._local_q, [enq_op]))

  def set_many_fed_tensors(self, tensors):
    """Sets batches fed tensors."""
    enq_op = self._local_q.enqueue_many(tensors)
    queue_runner.add_queue_runner(queue_runner.QueueRunner(
        self._local_q, [enq_op]))

  def add_remote_device(self, remote_device):
    """Requests that fed values are sent to `remote_device`."""
    local_value = self.get_fed_tensors()

    self._num_remote_feeds += 1

    with ops.device(None):  # Bypass any existing device() calls
      with ops.device(remote_device):
        remote_q = data_flow_ops.FIFOQueue(capacity=self._capacity,
                                           dtypes=self._dtypes,
                                           shapes=self._shapes,
                                           name=self._shared_name,
                                           shared_name=self._shared_name)
        remote_enq_op = remote_q.enqueue(local_value)

    # Add a remote queue runner to feed the remote queue.
    self._add_remote_queue_runner(remote_q, [remote_enq_op])

  def add_remote_devices(self, devices):
    for d in devices:
      self.add_remote_device(d)

  def add_remote_replicas(self, job_name, replica_count, feeder_task_num=None,
                          replicas_per_feeder=None,
                          base_device_spec=None):
    """Adds feeding for a range of replicas from `job_name`.

    Args:
      job_name: The job name portion of the remote jobs
      replica_count: The total number of remote jobs
      feeder_task_num: Optional; if there is more than one feeder job
        in the flock this is the task # of the current process.
      replicas_per_feeder: How many replicas each feeder job should
        push to. If present, `feeder_task_num` is required.
      base_device_spec: Optional base device spec. If present, then
        each replica device spec is derived from `base_device_spec`,
        with the job and replica properties set.
    Raises:
      ValueError: On invalid arguments.
    """
    if replicas_per_feeder is not None and feeder_task_num is None:
      raise ValueError(
          'Must set feeder_task_num if replicas_per_feeder is provided.')

    if replicas_per_feeder is None:
      replicas_per_feeder = replica_count
      feeder_task_num = 0

    if isinstance(base_device_spec, device.DeviceSpec):
      device_spec = copy.copy(base_device_spec)
    else:
      device_spec = device.DeviceSpec.from_string(base_device_spec or '')

    device_spec.job = job_name

    start_index = feeder_task_num * replicas_per_feeder
    end_index = start_index + replicas_per_feeder

    for idx in range(start_index, end_index):
      device_spec.replica = (idx % replica_count)
      self.add_remote_device(device_spec.to_string())

  def run_feeding_forever(self,
                          sess_callback,
                          outer_coordinator=None,
                          tolerator=None,
                          start_queue_runners=True):
    """Runs feeding forever.

    This method exits only if `outer_coordinator` has a stop requested
    or if a remote feed encounters an un-tolerated error. The most
    likely cause of `outer_coordinator` stopping besides a manual call
    to `request_stop()` is a `QueueRunner` thread reaching the end of
    its queue or encountering an error.

    Returns only after joining `outer_coordinator`.

    Args:
      sess_callback: A function which, when called, returns a Session
        to use for feeding. Can be called multiple times due to retries.
      outer_coordinator: If present, a `Coordinator` which the feeding
        process will respect. Will be created if omitted.
      tolerator: If present, a `failure_tolerator.FailureTolerator` which is
        used to manage retries of feeding the remote devices.
      start_queue_runners: Whether to start queue runners before
        beginning to feed the remote devices. Defaults to True. If
        False and no other mechanism is used to start queue runners, this
        method will hang forever without doing work.

    """
    # We use /two/ coordinators: one which runs normal queue
    # runners (outer_coordinator), and one which runs the remote
    # enqueues (using an inner coordinator) with retries and failure
    # tolerance. By using two coordinators, errors
    # encountered while running the remote enqueue ops don't cause the
    # outer_coordinator to be shut down.
    if outer_coordinator is None:
      outer_coordinator = coordinator.Coordinator()

    # Start the outer queue runners:
    if start_queue_runners:
      session = sess_callback()
      # Work around b/32749157 by running an operation before proceeding --
      # this way the session used for queue runners will be fully established
      # before we create another session with the same target.
      session.run(self._fake_op)
      queue_runner.start_queue_runners(sess=session,
                                       coord=outer_coordinator)

    if self._num_remote_feeds == 0:
      self._feeding_event.set()
      outer_coordinator.join()
      return
    else:
      try:
        self._feed_remote_queues_forever(
            sess_callback, outer_coordinator, tolerator)
      finally:
        self._feeding_event.set()
        outer_coordinator.join()

  def wait_until_feeding(self, timeout=None):
    """Waits until run_feeding_forever() is entered.

    Does not return until it is safe to create new sessions against
    the same target as the feeder is using; see b/32749157.

    Args:
      timeout: An optional timeout in seconds.
    Returns:
      True if feeding has begun; False if the timeout was reached.
    """
    return self._feeding_event.wait(timeout=timeout)

  def _feed_remote_queues_forever(
      self, sess_callback, outer_coordinator, tolerator):
    if tolerator is None:
      tolerator = failure_tolerator.FailureTolerator(limit=5)

    # In a retry loop, keep the remote queue runners going:
    while True:
      if outer_coordinator.should_stop():
        return

      inner_coordinator = coordinator.Coordinator()

      # Make sure inner_coordinator stops when outer_coordinator does:
      _link_coordinators(inner_coordinator, outer_coordinator)

      # Create a fresh session to use for remote queues:
      inner_session = sess_callback()
      inner_session.run(self._fake_op)  # Work around b/32749157, as above

      queue_runner.start_queue_runners(sess=inner_session,
                                       coord=inner_coordinator,
                                       collection=Feeder.REMOTE_QUEUE_RUNNERS)

      self._feeding_event.set()  # Notify that feeding has begun

      try:
        with tolerator.forgive():
          # Wait for a stop to be requested.
          inner_coordinator.wait_for_stop()

          # TODO(shoutis): If outer_coordinator.should_stop(), it
          # would be nice to interrupt the remote queue runners (which
          # may be blocked if their remote queue is full) -- but
          # there's no way currently; see b/32774422.

          # Cause any exceptions from the remote queue runners to be
          # reraised immediately, without waiting for their associated
          # threads to terminate like join() would. This means a retry
          # can begin immediately after any remote device fails,
          # rather than having to wait for any pending enqueues to
          # other remote devices to finish first.
          inner_coordinator.raise_requested_exception()

          # If this line is reached, there was a graceful shutdown
          # requested.

          # Request the outer coordinator to stop. Since
          # outer_coordinator.request_stop() is the currently only way
          # for inner_coordinator() to finish without failure, this is
          # redundant, but it's harmless and defends against infinite
          # hangs should code changes make it possible for
          # inner_coordinator to finish in other ways.
          outer_coordinator.request_stop()

          return
      except Exception as e:
        # Pass non-forgiven errors along to outer_coordinator:
        outer_coordinator.request_stop(e)
        raise

  def _add_remote_queue_runner(self, queue, enq_ops):
    """Adds a remote queue runner to the graph.

    These queue runners differ from the standard in two ways: First,
    they never close their queue. Second, they are added to the
    `Feeder.REMOTE_QUEUE_RUNNERS` collection, rather than
    `ops.GraphKeys.QUEUE_RUNNERS`, so they can be started/stopped
    seperately.

    Args:
      queue: The queue.
      enq_ops: A list of ops which perform enqueues (each on its own thread).
    """

    runner = queue_runner.QueueRunner(
        queue,
        enq_ops,
        cancel_op=self._fake_op,
        close_op=self._fake_op)
    queue_runner.add_queue_runner(
        runner, collection=Feeder.REMOTE_QUEUE_RUNNERS)


def _link_coordinators(inner_coord, outer_coord, start=True, wait_time=5):
  """Returns a thread which stops `inner_coord` whenever `outer_coord` stops.

  The thread is also registered with `inner_coord`.

  Args:
    inner_coord: The `Coordinator` to stop.
    outer_coord: The `Coordinator` to watch for stopping.
    start: Whether to start the thread before returning.
    wait_time: The number of seconds for each `outer_coord.wait_for_stop` call.
  Returns:
    A `Thread` which links the coordinators.
  """
  def _link_thread():
    while True:
      if inner_coord.should_stop():
        # The inner coordinator is stopping, so this thread's done.
        return

      if outer_coord.wait_for_stop(wait_time):
        # The outer coordinator stopped; we should stop the inner.
        with inner_coord.stop_on_exception():
          # Causes a re-raise, but without waiting for registered threads
          outer_coord.raise_requested_exception()
          inner_coord.request_stop()
          return

  result = threading.Thread(target=_link_thread)
  inner_coord.register_thread(result)
  if start:
    result.start()
  return result
