# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops.summary_ops import tensor_summary

import contextlib
import re
import fnmatch

@contextlib.contextmanager
def ipu_session(compilation_trace=True, io_trace=False, execution_trace=True):
  opts = config_pb2.IPUOptions()
  dev = opts.device_config.add()
  dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
  dev.profiling.enable_compilation_trace = compilation_trace
  dev.profiling.enable_io_trace = io_trace
  dev.profiling.enable_execution_trace = execution_trace
  dev.profiling.enable_poplar_reports_text = True
  with session_lib.Session(
      config=config_pb2.ConfigProto(ipu_options=opts)) as sess:
    yield sess

def get_compute_sets_from_report(report):
  lines = report.split('\n')
  cs = [x for x in lines if re.search('  Step #\d+:', x)]
  cs = [x.split(":")[1].strip() for x in cs]
  cs = [x.split()[0] for x in cs]
  return cs

def check_all_compute_sets_in_list(cs_list, whitelist):
  wl = [x+'*' for x in whitelist]
  if len(cs_list) < len(wl):
    return False
  for cs in cs_list:
    if len([x for x in wl if fnmatch.fnmatch(cs, x)]) == 0:
      return False
  return True

def extract_all_strings_from_event_trace(events):
  result = ""
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result = result + evt.data_str.decode('utf-8')
  return result

def extract_all_types_from_event_trace(events):
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result += [evt.type]
  return result

def extract_all_events(events):
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result += [evt]
  return result

def extract_all_io_events(events):
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type in [IpuTraceEvent.HOST_TO_DEVICE_TRANSFER,
                    IpuTraceEvent.DEVICE_TO_HOST_TRANSFER]:
      result += [(evt.type, evt.data_str.decode('utf-8'))]
  return result

def ipu_compile_summary(name, op_list, collections=None):

  with ops.device("cpu"):
    with ops.control_dependencies(op_list):
      reports = gen_ipu_ops.ipu_event_trace()

      summary_metadata = summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="ipu"))

      t_summary = tensor_summary(name=name, tensor=reports,
                                 summary_metadata=summary_metadata,
                                 collections=collections)
  return t_summary
