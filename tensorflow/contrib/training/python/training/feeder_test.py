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
"""Tests for tf.contrib.training.feeder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import portpicker
from tensorflow.contrib.training.python.training import feeder as feeder_lib
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import server_lib


_PORTS = set()


def _pick_unused_port():
  """For some reason portpicker returns the same port sometimes."""
  while True:
    p = portpicker.pick_unused_port()
    if p not in _PORTS:
      break
  _PORTS.add(p)
  return p


class FeederThread(object):
  # Helper class, wrapping a feeder and making sure it's located on the proper
  # device

  def __init__(self, test_case, coord, servers, job, task_num, prefix=''):
    self.graph = ops.Graph()
    self.coord = coord
    self.server = servers[job][task_num]
    self.remote_devices = []

    # Just because we do tf.session(X) doesn't mean ops will located
    # on the X task; wrapping all feeder creation/interaction in an
    # extra tf.device(X) ensures that any ops that don't provider
    # their own tf.device() wrapper will be placed on the correct "local"
    # feeder task. A session can and does put ops that have no device
    # assignment onto any of the tasks it knows about, not just the
    # task passed as its target= argument!
    self.device = '/job:%s/task:%d' % (job, task_num)
    self.prefix = prefix
    self.thread = test_case.checkedThread(target=self._feed_thread)

    with self.graph.as_default(), ops.device(self.device):
      self.feeder = feeder_lib.Feeder(
          [dtypes_lib.string, dtypes_lib.string], [[], []], capacity=1)
      self.feeder.set_many_fed_tensors(self._get_feed_values())

  def _get_feed_values(self):
    # Return some feeding strings, possibly prefixed.
    return [
        constant_op.constant(
            ['%s%s' % (self.prefix, x) for x in ['a0', 'a1', 'a2']]),
        constant_op.constant(
            ['%s%s' % (self.prefix, x) for x in ['b0', 'b1', 'b2']])
    ]

  def add_remote_device(self, dev):
    with self.graph.as_default(), ops.device(self.device):
      self.feeder.add_remote_device(dev)

  def start(self):
    self.thread.start()
    self.feeder.wait_until_feeding()  # wait until it's up & feeding
    if self.coord.should_stop():
      self.coord.join()  # rethrows errors encountered in run_feeding_forever

  def join(self):
    self.thread.join()

  def _session(self):
    return session_lib.Session(target=self.server.target)

  def _feed_thread(self):
    with self.coord.stop_on_exception():
      with self.graph.as_default(), ops.device(self.device):
        self.feeder.run_feeding_forever(self._session, self.coord)


class FeederTest(test.TestCase):
  # Tests for Feeder

  def _create_local_cluster(self, **kargs):
    """Creates a local cluster."""
    cluster_dict = {}
    for (k, v) in kargs.items():
      cluster_dict[k] = [
          'localhost:%d' % _pick_unused_port() for _ in range(v)
      ]

    # Launch servers:
    servers = {}
    for (k, v) in kargs.items():
      servers[k] = [
          server_lib.Server(
              cluster_dict, job_name=k, task_index=idx, start=True)
          for idx in range(v)
      ]
    return servers

  def testFeederActsLikeQueue(self):
    # Tests that a feeder acts like a queue
    feeder = feeder_lib.Feeder(
        dtypes=[dtypes_lib.string, dtypes_lib.string],
        shapes=[[], []],
        capacity=10)

    feeder.set_many_fed_tensors([
        constant_op.constant(['a0', 'a1', 'a2']),
        constant_op.constant(['b0', 'b1', 'b2'])
    ])

    out_a, out_b = feeder.get_fed_tensors()

    with self.test_session() as session:
      coord = coordinator.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coord)

      a, b = session.run([out_a, out_b])
      self.assertEquals(b'a0', a)
      self.assertEquals(b'b0', b)
      a = session.run(out_a)  # Omit b!
      self.assertEquals(b'a1', a)
      a, b = session.run([out_a, out_b])
      self.assertEquals(b'a2', a)
      self.assertEquals(b'b2', b)  # queued together
      a, b = session.run([out_a, out_b])  # loops around
      self.assertEquals(b'a0', a)
      self.assertEquals(b'b0', b)  # queued together

    coord.request_stop()
    coord.join()

  def testFeederSeparateThread(self):
    # Start a feeder on a seperate thread, but with a shared local queue
    servers = self._create_local_cluster(worker=1)
    coord = coordinator.Coordinator()
    feed_thread = FeederThread(self, coord, servers, 'worker', 0)
    feed_thread.start()

    with ops.Graph().as_default():
      with ops.device('/job:worker/task:0'):
        feeder = feeder_lib.Feeder(
            dtypes=[dtypes_lib.string, dtypes_lib.string],
            shapes=[[], []],
            capacity=1)

        out_a, out_b = feeder.get_fed_tensors()

      with session_lib.Session(servers['worker'][0].target) as session:
        a, b = session.run([out_a, out_b])
        self.assertEquals(b'a0', a)
        self.assertEquals(b'b0', b)
        a = session.run(out_a)  # Omit b!
        self.assertEquals(b'a1', a)

    coord.request_stop()
    coord.join()
    feed_thread.join()

  def testOneEachFeeding(self):
    # One feeder, one consumer
    servers = self._create_local_cluster(consumer=1, feeder=1)

    coord = coordinator.Coordinator()
    feeder_thread = FeederThread(self, coord, servers, 'feeder', 0)
    feeder_thread.add_remote_device('/job:consumer/task:0')
    feeder_thread.start()

    with ops.Graph().as_default():
      with ops.device('/job:consumer/task:0'):
        feeder = feeder_lib.Feeder(
            dtypes=[dtypes_lib.string, dtypes_lib.string],
            shapes=[[], []],
            capacity=1)

        out_a, out_b = feeder.get_fed_tensors()

      with session_lib.Session(servers['consumer'][0].target) as session:
        a, b = session.run([out_a, out_b])
        self.assertEquals(b'a0', a)
        self.assertEquals(b'b0', b)
        a = session.run(out_a)  # Omit b!
        self.assertEquals(b'a1', a)

    coord.request_stop()
    coord.join()
    feeder_thread.join()

  def testMultipleProducersAndConsumers(self):
    # Three feeders, three consumers.
    servers = self._create_local_cluster(consumer=3, feeder=3)

    coord = coordinator.Coordinator()

    # Start the three feeders:
    f0 = FeederThread(self, coord, servers, 'feeder', 0, prefix='feed0_')
    f0.add_remote_device('/job:consumer/task:0')
    f0.add_remote_device('/job:consumer/task:1')
    f0.start()

    f1 = FeederThread(self, coord, servers, 'feeder', 1, prefix='feed1_')
    f1.add_remote_device('/job:consumer/task:2')
    f1.add_remote_device('/job:consumer/task:0')
    f1.start()

    f2 = FeederThread(self, coord, servers, 'feeder', 2, prefix='feed2_')
    f2.add_remote_device('/job:consumer/task:1')
    f2.add_remote_device('/job:consumer/task:2')
    f2.start()

    # Three consumers.
    def _run_consumer(task, expected_keys):
      server = servers['consumer'][task]
      # Runs until everything in expected_keys has been seen at least once;
      # fails if any prefix not in expected_keys shows up
      with ops.Graph().as_default(), ops.device('/job:consumer/task:%d' % task):
        feeder = feeder_lib.Feeder(
            dtypes=[dtypes_lib.string, dtypes_lib.string],
            shapes=[[], []],
            capacity=1)

        out_a, out_b = feeder.get_fed_tensors()
        counts = collections.Counter()
        with session_lib.Session(server.target) as sess:
          while True:
            a, b = sess.run([out_a, out_b])
            counts[a[:-1]] += 1
            counts[b[:-1]] += 1

            self.assertTrue(a[:-1] in expected_keys)
            self.assertTrue(b[:-1] in expected_keys)

            if all(counts[k] > 0 for k in expected_keys):
              return

    _run_consumer(0, [b'feed0_a', b'feed0_b', b'feed1_a', b'feed1_b'])
    _run_consumer(1, [b'feed0_a', b'feed0_b', b'feed2_a', b'feed2_b'])
    _run_consumer(2, [b'feed1_a', b'feed1_b', b'feed2_a', b'feed2_b'])

    coord.request_stop()
    coord.join()

    f0.join()
    f1.join()
    f2.join()

  def testAddRemoteReplicas(self):
    with ops.Graph().as_default():
      for idx in range(3):
        with ops.name_scope('replica_%d' % idx):
          feeder = feeder_lib.Feeder(
              dtypes=[dtypes_lib.string, dtypes_lib.string],
              shapes=[[], []],
              capacity=10)

          feeder.add_remote_replicas(
              'consumer',
              replica_count=3,
              feeder_task_num=idx,
              replicas_per_feeder=2,
              base_device_spec='/device:cpu:0')

      # Examine ops...
      op_types_by_scope_and_device = collections.defaultdict(
          lambda: collections.defaultdict(collections.Counter))

      for op in ops.get_default_graph().get_operations():
        scope = '/'.join(op.name.split('/')[:-1])
        dev = op.device

        op_types_by_scope_and_device[scope][dev][op.type] += 1

      expected_ops = collections.Counter(
          {'QueueEnqueueV2': 1, 'FIFOQueueV2': 1})
      expected_enq_devices = [('replica_0', [
          '/job:consumer/replica:0/device:cpu:0',
          '/job:consumer/replica:1/device:cpu:0',
      ]), ('replica_1', [
          '/job:consumer/replica:2/device:cpu:0',
          '/job:consumer/replica:0/device:cpu:0',
      ]), ('replica_2', [
          '/job:consumer/replica:1/device:cpu:0',
          '/job:consumer/replica:2/device:cpu:0',
      ])]

      for scope, devs in expected_enq_devices:
        for dev in devs:
          self.assertEqual(expected_ops,
                           op_types_by_scope_and_device[scope][dev])


if __name__ == '__main__':
  test.main()
