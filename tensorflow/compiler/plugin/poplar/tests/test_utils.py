# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import fnmatch
import json as js
import numpy as np
import re

from tensorflow.core.framework import summary_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.summary.summary import tensor_summary

@contextlib.contextmanager
def ipu_session(compilation_trace=True, io_trace=False, execution_trace=True,
                report_every_nth_execution=0, text_report=True, sharded=False,
                compile_ipu_code=False, enable_ipu_events=False):
  opts = config_pb2.IPUOptions()
  opts.profiling.enable_ipu_trace_events = (compilation_trace or io_trace or
                                            execution_trace or
                                            enable_ipu_events)
  opts.profiling.enable_compilation_trace = compilation_trace
  opts.profiling.enable_io_trace = io_trace
  opts.profiling.enable_execution_trace = execution_trace
  opts.profiling.enable_poplar_reports_text = text_report
  opts.profiling.report_every_nth_execution = report_every_nth_execution
  opts.ipu_model_config.enable_ipu_model = True
  opts.ipu_model_config.compile_ipu_code = compile_ipu_code

  if sharded:
    dev = opts.device_config.add()
    dev.auto_count = 2

  with session_lib.Session(
      config=config_pb2.ConfigProto(ipu_options=opts)) as sess:
    yield sess


@contextlib.contextmanager
def ipu_shard(index):

  ipus = []
  if hasattr(index, '__iter__'):
    ipus = index
  else:
    ipus = [index]

  proto = xla_data_pb2.OpSharding(
    type=xla_data_pb2.OpSharding.MAXIMAL, tile_assignment_devices=ipus)

  attr_value = attr_value_pb2.AttrValue(s=proto.SerializeToString())
  attrs = {"_XlaSharding": attr_value}

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access

def get_total_memory_from_report(report):
  lines = report.split('\n')
  found = False
  for l in lines:
    if not found:
      m = re.search('Memory Usage:', l)
      if m:
        found = True
    else:
      m = re.search('Including Gaps: +([\d,]+) B', l)
      if m:
        return int(m.group(1).replace(',', ''))
  return None

def get_compute_sets_from_report(report):
  lines = report.split('\n')
  cs = [x for x in lines if re.search(' OnTileExecute .*: ', x)]
  cs = [x.split(":")[1].strip() for x in cs]
  cs = [x.split()[0] for x in cs]
  return cs

def get_maximum_tile_size_from_events(report):
  print(report)
  lines = report.split('\n')
  found = False
  for l in lines:
    if not found:
      m = re.search('Memory Usage:', l)
      if m:
        found = True
    else:
      m = re.match(r' +Maximum.*: ([\d,]+) .*on tile [\d,]+', l)
      if m:
        return int(m.group(1).replace(',', ''))
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

def get_compute_sets_from_json_report(event):
  if event.type == IpuTraceEvent.COMPILE_END:
    rep = js.loads(event.compile_end.compilation_report.decode('utf-8'))
    return rep['computeSets']['names']
  else:
    return []

def get_all_global_exchange_from_json_report(event):
  if event.type == IpuTraceEvent.COMPILE_END:
    rep = js.loads(event.compile_end.compilation_report.decode('utf-8'))
    return [p['name'] for p in rep['programs'] if p['type'] == 'GlobalExchange']
  else:
    return []

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

def extract_all_execute_events(events):
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type == IpuTraceEvent.EXECUTE:
      result += [evt]
  return result

def extract_all_io_events(events):
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type in [IpuTraceEvent.HOST_TO_DEVICE_TRANSFER,
                    IpuTraceEvent.DEVICE_TO_HOST_TRANSFER]:
      try:
        payload = js.loads(evt.data_transfer.data_transfer.decode('utf-8'))
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
