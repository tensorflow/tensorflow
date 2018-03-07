# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.core.protobuf import config_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.core.framework import summary_pb2
from tensorflow.python.ops.summary_ops import tensor_summary

import contextlib
import re

@contextlib.contextmanager
def ipu_session():
  opts = config_pb2.IPUOptions()
  dev = opts.device_config.add()
  dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
  dev.profiling.enable_compilation_trace = True
  dev.profiling.enable_io_trace = True
  dev.profiling.enable_execution_trace = True
  with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:
    yield sess

def get_compute_sets_from_report(report):
  lines = report.split('\n')
  return [x for x in lines if re.search('(\d+ execution.?)', x)]

def check_all_compute_sets_in_list(cs_list, whitelist):
  for cs in cs_list:
    if len([x for x in whitelist if cs.startswith(x)]) == 0:
      return False
  return True

def ipu_compile_summary(name, op_list, collections=None):

  with ops.device("cpu"):
    with ops.control_dependencies(op_list):
      reports = gen_ipu_ops.ipu_summary()

      summary_metadata = summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="ipu"))

      t_summary = tensor_summary(name=name, tensor=reports,
                                 summary_metadata=summary_metadata,
                                 collections=collections)
  return t_summary
