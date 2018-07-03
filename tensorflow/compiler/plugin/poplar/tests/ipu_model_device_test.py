# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re

import test_utils as tu

from tensorflow.python.client import session as session_lib
from tensorflow.python.platform import googletest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class IpuIpuModelTest(test_util.TensorFlowTestCase):

    def testIpuModelDevice(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [2,2], name="a")
            pb = array_ops.placeholder(np.float32, [2,2], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL

        with session_lib.Session(
                config=config_pb2.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            result = sess.run(output, fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])

    def testIpuModelDeviceWithNoReport(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [2,2], name="a")
            pb = array_ops.placeholder(np.float32, [2,2], name="b")
            output = pa + pb

        with ops.device('cpu'):
            with ops.control_dependencies([output]):
                report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL

        with session_lib.Session(
                config=config_pb2.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            sess.run(report, fd)

            result, rep = sess.run([output, report], fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])
            self.assertTrue(len(rep) == 0)

    def testIpuModelDeviceWithReport(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [2,2], name="a")
            pb = array_ops.placeholder(np.float32, [2,2], name="b")
            output = pa + pb

        with ops.device('cpu'):
            with ops.control_dependencies([output]):
                report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True

        with session_lib.Session(
                config=config_pb2.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            sess.run(report, fd)

            result, rep = sess.run([output, report], fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])
            self.assertTrue(len(rep) == 3)
            evts = tu.extract_all_events(rep)
            self.assertEqual(evts[0].type, IpuTraceEvent.COMPILE_BEGIN)
            self.assertEqual(evts[1].type, IpuTraceEvent.COMPILE_END)
            self.assertEqual(evts[2].type, IpuTraceEvent.EXECUTE)

    def testIpuModelDeviceWithMultipleReport(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [2,2], name="a")
            pb = array_ops.placeholder(np.float32, [2,2], name="b")
            out1 = pa + pb
            out2 = pa - pb

        with ops.device('cpu'):
            with ops.control_dependencies([out1, out2]):
                report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True

        with session_lib.Session(
                config=config_pb2.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            sess.run(report, fd)

            result = sess.run(out1, fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])

            result, rep = sess.run([out2, report], fd)
            self.assertAllClose(result, [[1.,0.],[-2.,-2.]])

            # 2x compile_begin, 2x compile_end, 2x load engine
            self.assertTrue(len(rep) == 6)

    def testIpuModelDeviceMultipleIPUs(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [480], name="a")
            pb = array_ops.placeholder(np.float32, [480], name="b")
            output = pa + pb

        with ops.device('cpu'):
            with ops.control_dependencies([output]):
                report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True
        dev.ipu_model_config.num_ipus = 2
        dev.ipu_model_config.tiles_per_ipu = 4

        with session_lib.Session(
                config=config_pb2.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: np.zeros([480]), pb: np.zeros([480])}
            sess.run(report, fd)

            result, rep = sess.run([output, report], fd)
            self.assertAllClose(result, np.zeros([480]))
            self.assertTrue(len(rep) == 3)

            s = tu.extract_all_strings_from_event_trace(rep)
            l = s.split("\n")
            l = [x for x in l if re.search(" *Num tiles computing *: *8", x)]
            self.assertTrue(len(l) == 1)

    def testIpu(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [480], name="a")
            pb = array_ops.placeholder(np.float32, [480], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True
        dev.ipu_model_config.num_ipus = 1

        try:
            with session_lib.Session(
                    config=config_pb2.ConfigProto(ipu_options=opts)) as sess:
                fd = {pa: np.zeros([480]), pb: np.zeros([480])}
                sess.run(output, fd)
        except errors.InternalError:
            pass

    def testIpuWithSpecificTiles(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [480], name="a")
            pb = array_ops.placeholder(np.float32, [480], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True
        dev.ipu_model_config.num_ipus = 1
        dev.ipu_model_config.tiles_per_ipu = 4

        try:
            with session_lib.Session(
                    config=config_pb2.ConfigProto(ipu_options=opts)) as sess:
                fd = {pa: np.zeros([480]), pb: np.zeros([480])}
                sess.run(output, fd)
        except errors.InternalError:
            pass

    def testEngineCompilationOptions(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [480], name="a")
            pb = array_ops.placeholder(np.float32, [480], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True
        dev.ipu_model_config.num_ipus = 1

        opt = dev.compilation_options.add()
        opt.option = "some_option"
        opt.value = "some_value"

        try:
            with session_lib.Session(
                    config=config_pb2.ConfigProto(ipu_options=opts)) as sess:
                fd = {pa: np.zeros([480]), pb: np.zeros([480])}
                sess.run(output, fd)
        except errors.UnknownError:
            pass

    def testIpuSimulatorDevice(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [16], name="a")
            pb = array_ops.placeholder(np.float32, [16], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_SIMULATOR

        try:
            with session_lib.Session(
                    config=config_pb2.ConfigProto(ipu_options=opts)) as sess:
                fd = {pa: np.zeros([16]), pb: np.zeros([16])}
                res = sess.run(output, fd)

                self.assertAllClose(res, np.zeros([16]))
        except errors.InternalError as e:
            self.assertTrue(e.message.startswith("Failed to create session"))

    def testNamedOperations(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float32, [2,2], name="a")
            pb = array_ops.placeholder(np.float32, [2,2], name="b")
            with ops.name_scope('my_ops'):
                out = math_ops.add(pa, pb, 'my_add_op')

        with ops.device('cpu'):
            report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True

        with session_lib.Session(
                config=config_pb2.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            sess.run(report, fd)

            result = sess.run(out, fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])

            rep = sess.run(report, fd)
            s = tu.extract_all_strings_from_event_trace(rep)
            cs_list = tu.get_compute_sets_from_report(s)

            print(cs_list)

            ok = ['my_ops/my_add_op/add']

            self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
