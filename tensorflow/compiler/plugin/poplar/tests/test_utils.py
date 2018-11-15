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
import fnmatch
import json
import re

@contextlib.contextmanager
def ipu_session(compilation_trace=True, io_trace=False, execution_trace=True,
                report_every_nth_execution=0, text_report=True, sharded=False,
                compile_ipu_code=False):
  opts = config_pb2.IPUOptions()
  opts.profiling.enable_compilation_trace = compilation_trace
  opts.profiling.enable_io_trace = io_trace
  opts.profiling.enable_execution_trace = execution_trace
  opts.profiling.enable_poplar_reports_text = text_report
  opts.profiling.report_every_nth_execution = report_every_nth_execution
  opts.ipu_model_config.enable_ipu_model = True
  opts.ipu_model_config.compile_ipu_code = compile_ipu_code

  if sharded:
    opts.enable_sharding = True

    dev = opts.device_config.add()
    dev.auto_count = 2

  with session_lib.Session(
      config=config_pb2.ConfigProto(ipu_options=opts)) as sess:
    yield sess

def get_total_memory_from_report(report):
  lines = report.split('\n')
  found = False
  for x in lines:
    if not found:
      m = re.search('Memory Usage\s+:', x)
      if m:
        found = True
    else:
      m = re.search('Total\s*:\s+(\d+) bytes', x)
      if m:
        return int(m.group(1))
  return None

def get_compute_sets_from_report(report):
  lines = report.split('\n')
  cs = [x for x in lines if re.search('  Step #\d+:', x)]
  cs = [x.split(":")[1].strip() for x in cs]
  cs = [x.split()[0] for x in cs]
  return cs


def get_maximum_tile_size_from_events(report):
  lines = report.split('\n')
  for l in lines:
    if l.startswith('Max tile memory'):
        m = re.match(r'Max tile memory \(tile \d+\): (\d+) \(\d+% of mean\)', l)
        if m:
          return int(m.group(1))
  return None

def check_compute_sets_not_in_blacklist(cs_list, bl):
  result = True
  fail_list = []
  for x in bl:
    matches = [cs for cs in cs_list if fnmatch.fnmatch(cs, x)]
    if len(matches) > 0:
      fail_list += matches
      result = False
  if not result:
    print("Compute sets present: " + str(fail_list))
  return result

def check_whitelist_entries_in_compute_sets(cs_list, whitelist):
  result = True
  fail_list = []
  wl = [x+'*' for x in whitelist]
  for cs in cs_list:
    if len([x for x in wl if fnmatch.fnmatch(cs, x)]) == 0:
      fail_list += [ cs ]
      result = False
  if not result:
    print("Failed to match " + str(fail_list))
  return result

def check_compute_sets_in_whitelist_entries(cs_list, whitelist):
  result = True
  fail_list = []
  wl = [x+'*' for x in whitelist]
  for x in wl:
    if len([cs for cs in cs_list if fnmatch.fnmatch(cs, x)]) == 0:
      fail_list += [ x ]
      result = False
  if not result:
    print("Failed to match " + str(fail_list))
  return result

def check_all_compute_sets_and_list(cs_list, whitelist):
  return (check_whitelist_entries_in_compute_sets(cs_list, whitelist) and
          check_compute_sets_in_whitelist_entries(cs_list, whitelist))

def extract_all_strings_from_event_trace(events):
  result = ""
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    try:
      if evt.type == IpuTraceEvent.COMPILE_BEGIN:
        pass
      if evt.type == IpuTraceEvent.COMPILE_END:
        result = result + evt.compile_end.compilation_report.decode('utf-8')
      if evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
        result = result + evt.data_transfer.data_transfer.decode('utf-8')
      if evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
        result = result + evt.data_transfer.data_transfer.decode('utf-8')
      if evt.type == IpuTraceEvent.LOAD_ENGINE:
        pass
      if evt.type == IpuTraceEvent.EXECUTE:
        result = result + evt.execute.execution_report.decode('utf-8')
    except UnicodeDecodeError:
      pass
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
      try:
        payload = json.loads(evt.data_transfer.data_transfer.decode('utf-8'))
        for t in payload["tensors"]:
          result += [(evt.type, t["name"])]
      except UnicodeDecodeError:
        pass
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
