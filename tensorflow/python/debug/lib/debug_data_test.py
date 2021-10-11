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
"""Tests for tfdbg module debug_data."""
import os
import platform
import tempfile

import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class DeviceNamePathConversionTest(test_util.TensorFlowTestCase):

  def testDeviceNameToDevicePath(self):
    self.assertEqual(
        debug_data.METADATA_FILE_PREFIX + debug_data.DEVICE_TAG +
        ",job_ps,replica_1,task_2,cpu_0",
        debug_data.device_name_to_device_path("/job:ps/replica:1/task:2/cpu:0"))

  def testDevicePathToDeviceName(self):
    self.assertEqual(
        "/job:ps/replica:1/task:2/cpu:0",
        debug_data.device_path_to_device_name(
            debug_data.METADATA_FILE_PREFIX + debug_data.DEVICE_TAG +
            ",job_ps,replica_1,task_2,cpu_0"))


class HasNanOrInfTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._dummy_datum = dummy_datum = debug_data.DebugTensorDatum(
        "/foo", "bar_0_DebugIdentity_42")

  def testNaN(self):
    a = np.array([np.nan, np.nan, 7.0])
    self.assertTrue(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testInf(self):
    a = np.array([np.inf, np.inf, 7.0])
    self.assertTrue(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testNanAndInf(self):
    a = np.array([np.inf, np.nan, 7.0])
    self.assertTrue(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testNoNanOrInf(self):
    a = np.array([0.0, 0.0, 7.0])
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testEmpty(self):
    a = np.array([])
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testInconvertibleTensorProto(self):
    self.assertFalse(debug_data.has_inf_or_nan(
        self._dummy_datum,
        debug_data.InconvertibleTensorProto(tensor_pb2.TensorProto(),
                                            initialized=False)))
    self.assertFalse(debug_data.has_inf_or_nan(
        self._dummy_datum,
        debug_data.InconvertibleTensorProto(tensor_pb2.TensorProto(),
                                            initialized=True)))

  def testDTypeComplexWorks(self):
    a = np.array([1j, 3j, 3j, 7j], dtype=np.complex128)
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

    b = np.array([1j, 3j, 3j, 7j, np.nan], dtype=np.complex128)
    self.assertTrue(debug_data.has_inf_or_nan(self._dummy_datum, b))

  def testDTypeIntegerWorks(self):
    a = np.array([1, 3, 3, 7], dtype=np.int16)
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testDTypeStringGivesFalse(self):
    """isnan and isinf are not applicable to strings."""

    a = np.array(["s", "p", "a", "m"])
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testDTypeObjectGivesFalse(self):
    dt = np.dtype([("spam", np.str_, 16), ("eggs", np.float64, (2,))])
    a = np.array([("spam", (8.0, 7.0)), ("eggs", (6.0, 5.0))], dtype=dt)
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))


class DebugTensorDatumTest(test_util.TensorFlowTestCase):

  def testDebugDatum(self):
    dump_root = "/tmp/tfdbg_1"
    debug_dump_rel_path = (
        debug_data.METADATA_FILE_PREFIX + debug_data.DEVICE_TAG +
        ",job_localhost,replica_0,task_0,cpu_0" +
        "/ns1/ns2/node_a_1_2_DebugIdentity_1472563253536385")

    datum = debug_data.DebugTensorDatum(dump_root, debug_dump_rel_path)

    self.assertEqual("DebugIdentity", datum.debug_op)
    self.assertEqual("ns1/ns2/node_a_1", datum.node_name)
    self.assertEqual(2, datum.output_slot)
    self.assertEqual("ns1/ns2/node_a_1:2", datum.tensor_name)
    self.assertEqual(1472563253536385, datum.timestamp)
    self.assertEqual("ns1/ns2/node_a_1:2:DebugIdentity", datum.watch_key)
    self.assertEqual(
        os.path.join(dump_root, debug_dump_rel_path), datum.file_path)
    self.assertEqual(
        "{DebugTensorDatum (/job:localhost/replica:0/task:0/cpu:0) "
        "%s:%d @ %s @ %d}" % (datum.node_name,
                              datum.output_slot,
                              datum.debug_op,
                              datum.timestamp), str(datum))
    self.assertEqual(
        "{DebugTensorDatum (/job:localhost/replica:0/task:0/cpu:0) "
        "%s:%d @ %s @ %d}" % (datum.node_name,
                              datum.output_slot,
                              datum.debug_op,
                              datum.timestamp), repr(datum))

  def testDumpSizeBytesIsNoneForNonexistentFilePath(self):
    dump_root = "/tmp/tfdbg_1"
    debug_dump_rel_path = "ns1/ns2/node_foo_1_2_DebugIdentity_1472563253536385"
    datum = debug_data.DebugTensorDatum(dump_root, debug_dump_rel_path)

    self.assertIsNone(datum.dump_size_bytes)


class DebugDumpDirTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._dump_root = tempfile.mktemp()
    os.mkdir(self._dump_root)

  def tearDown(self):
    # Tear down temporary dump directory.
    file_io.delete_recursively(self._dump_root)

  def _makeDataDirWithMultipleDevicesAndDuplicateNodeNames(self):
    cpu_0_dir = os.path.join(
        self._dump_root,
        debug_data.METADATA_FILE_PREFIX + debug_data.DEVICE_TAG +
        ",job_localhost,replica_0,task_0,cpu_0")
    gpu_0_dir = os.path.join(
        self._dump_root,
        debug_data.METADATA_FILE_PREFIX + debug_data.DEVICE_TAG +
        ",job_localhost,replica_0,task_0,device_GPU_0")
    gpu_1_dir = os.path.join(
        self._dump_root,
        debug_data.METADATA_FILE_PREFIX + debug_data.DEVICE_TAG +
        ",job_localhost,replica_0,task_0,device_GPU_1")
    os.makedirs(cpu_0_dir)
    os.makedirs(gpu_0_dir)
    os.makedirs(gpu_1_dir)
    open(os.path.join(
        cpu_0_dir, "node_foo_1_2_DebugIdentity_1472563253536386"), "wb")
    open(os.path.join(
        gpu_0_dir, "node_foo_1_2_DebugIdentity_1472563253536385"), "wb")
    open(os.path.join(
        gpu_1_dir, "node_foo_1_2_DebugIdentity_1472563253536387"), "wb")

  def testDebugDumpDir_nonexistentDumpRoot(self):
    with self.assertRaisesRegex(IOError, "does not exist"):
      debug_data.DebugDumpDir(tempfile.mktemp() + "_foo")

  def testDebugDumpDir_invalidFileNamingPattern(self):
    # File name with too few underscores should lead to an exception.
    device_dir = os.path.join(
        self._dump_root,
        debug_data.METADATA_FILE_PREFIX + debug_data.DEVICE_TAG +
        ",job_localhost,replica_0,task_0,cpu_0")
    os.makedirs(device_dir)
    open(os.path.join(device_dir, "node1_DebugIdentity_1234"), "wb")

    with self.assertRaisesRegex(ValueError,
                                "does not conform to the naming pattern"):
      debug_data.DebugDumpDir(self._dump_root)

  def testDebugDumpDir_validDuplicateNodeNamesWithMultipleDevices(self):
    self._makeDataDirWithMultipleDevicesAndDuplicateNodeNames()

    graph_cpu_0 = graph_pb2.GraphDef()
    node = graph_cpu_0.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/cpu:0"
    graph_gpu_0 = graph_pb2.GraphDef()
    node = graph_gpu_0.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:0"
    graph_gpu_1 = graph_pb2.GraphDef()
    node = graph_gpu_1.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:1"

    dump_dir = debug_data.DebugDumpDir(
        self._dump_root,
        partition_graphs=[graph_cpu_0, graph_gpu_0, graph_gpu_1])

    self.assertItemsEqual(
        ["/job:localhost/replica:0/task:0/cpu:0",
         "/job:localhost/replica:0/task:0/device:GPU:0",
         "/job:localhost/replica:0/task:0/device:GPU:1"], dump_dir.devices())
    self.assertEqual(1472563253536385, dump_dir.t0)
    self.assertEqual(3, dump_dir.size)

    with self.assertRaisesRegex(ValueError, r"Invalid device name: "):
      dump_dir.nodes("/job:localhost/replica:0/task:0/device:GPU:2")
    self.assertItemsEqual(["node_foo_1", "node_foo_1", "node_foo_1"],
                          dump_dir.nodes())
    self.assertItemsEqual(
        ["node_foo_1"],
        dump_dir.nodes(device_name="/job:localhost/replica:0/task:0/cpu:0"))

  def testDuplicateNodeNamesInGraphDefOfSingleDeviceRaisesException(self):
    self._makeDataDirWithMultipleDevicesAndDuplicateNodeNames()
    graph_cpu_0 = graph_pb2.GraphDef()
    node = graph_cpu_0.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/cpu:0"
    graph_gpu_0 = graph_pb2.GraphDef()
    node = graph_gpu_0.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:0"
    graph_gpu_1 = graph_pb2.GraphDef()
    node = graph_gpu_1.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:1"
    node = graph_gpu_1.node.add()  # Here is the duplicate.
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:1"

    with self.assertRaisesRegex(ValueError, r"Duplicate node name on device "):
      debug_data.DebugDumpDir(
          self._dump_root,
          partition_graphs=[graph_cpu_0, graph_gpu_0, graph_gpu_1])

  def testDebugDumpDir_emptyDumpDir(self):
    dump_dir = debug_data.DebugDumpDir(self._dump_root)

    self.assertIsNone(dump_dir.t0)
    self.assertEqual([], dump_dir.dumped_tensor_data)

  def testDebugDumpDir_usesGfileGlob(self):
    if platform.system() == "Windows":
      self.skipTest("gfile.Glob is not used on Windows.")

    self._makeDataDirWithMultipleDevicesAndDuplicateNodeNames()

    def fake_gfile_glob(glob_pattern):
      del glob_pattern
      return []

    with test.mock.patch.object(
        gfile, "Glob", side_effect=fake_gfile_glob, autospec=True) as fake:
      debug_data.DebugDumpDir(self._dump_root)
      expected_calls = [
          test.mock.call(os.path.join(
              self._dump_root,
              (debug_data.METADATA_FILE_PREFIX +
               debug_data.CORE_METADATA_TAG + "*"))),
          test.mock.call(os.path.join(
              self._dump_root,
              (debug_data.METADATA_FILE_PREFIX +
               debug_data.FETCHES_INFO_FILE_TAG + "*"))),
          test.mock.call(os.path.join(
              self._dump_root,
              (debug_data.METADATA_FILE_PREFIX +
               debug_data.FEED_KEYS_INFO_FILE_TAG + "*"))),
          test.mock.call(os.path.join(
              self._dump_root,
              (debug_data.METADATA_FILE_PREFIX +
               debug_data.DEVICE_TAG + "*")))]
      fake.assert_has_calls(expected_calls, any_order=True)


if __name__ == "__main__":
  googletest.main()
