# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Formats and displays profiling information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re

from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import source_utils


SORT_OPS_BY_OP_NAME = "node"
SORT_OPS_BY_OP_TIME = "op_time"
SORT_OPS_BY_EXEC_TIME = "exec_time"
SORT_OPS_BY_START_TIME = "start_time"
SORT_OPS_BY_LINE = "line"


class ProfileDatum(object):
  """Profile data point."""

  def __init__(self, node_exec_stats, file_line, op_type):
    """Constructor.

    Args:
      node_exec_stats: `NodeExecStats` proto.
      file_line: A `string` formatted as <file_name>:<line_number>.
      op_type: (string) Operation type.
    """
    self.node_exec_stats = node_exec_stats
    self.file_line = file_line
    self.op_type = op_type
    self.start_time = self.node_exec_stats.all_start_micros
    self.op_time = (self.node_exec_stats.op_end_rel_micros -
                    self.node_exec_stats.op_start_rel_micros)

  @property
  def exec_time(self):
    """Measures compute function execution time plus pre- and post-processing."""
    return self.node_exec_stats.all_end_rel_micros


class ProfileDataTableView(object):
  """Table View of profiling data."""

  def __init__(self, profile_datum_list, time_unit=cli_shared.TIME_UNIT_US):
    """Constructor.

    Args:
      profile_datum_list: List of `ProfileDatum` objects.
      time_unit: must be in cli_shared.TIME_UNITS.
    """
    self._profile_datum_list = profile_datum_list
    self.formatted_start_time = [
        datum.start_time for datum in profile_datum_list]
    self.formatted_op_time = [
        cli_shared.time_to_readable_str(datum.op_time,
                                        force_time_unit=time_unit)
        for datum in profile_datum_list]
    self.formatted_exec_time = [
        cli_shared.time_to_readable_str(
            datum.node_exec_stats.all_end_rel_micros,
            force_time_unit=time_unit)
        for datum in profile_datum_list]

    self._column_names = ["Node",
                          "Start Time (us)",
                          "Op Time (%s)" % time_unit,
                          "Exec Time (%s)" % time_unit,
                          "Filename:Lineno(function)"]
    self._column_sort_ids = [SORT_OPS_BY_OP_NAME, SORT_OPS_BY_START_TIME,
                             SORT_OPS_BY_OP_TIME, SORT_OPS_BY_EXEC_TIME,
                             SORT_OPS_BY_LINE]

  def value(self, row, col):
    if col == 0:
      return self._profile_datum_list[row].node_exec_stats.node_name
    elif col == 1:
      return self.formatted_start_time[row]
    elif col == 2:
      return self.formatted_op_time[row]
    elif col == 3:
      return self.formatted_exec_time[row]
    elif col == 4:
      return self._profile_datum_list[row].file_line
    else:
      raise IndexError("Invalid column index %d." % col)

  def row_count(self):
    return len(self._profile_datum_list)

  def column_count(self):
    return len(self._column_names)

  def column_names(self):
    return self._column_names

  def column_sort_id(self, col):
    return self._column_sort_ids[col]


def _list_profile_filter(
    profile_datum, node_name_regex, file_name_regex, op_type_regex,
    op_time_interval, exec_time_interval):
  """Filter function for list_profile command.

  Args:
    profile_datum: A `ProfileDatum` object.
    node_name_regex: Regular expression pattern object to filter by name.
    file_name_regex: Regular expression pattern object to filter by file.
    op_type_regex: Regular expression pattern object to filter by op type.
    op_time_interval: `Interval` for filtering op time.
    exec_time_interval: `Interval` for filtering exec time.

  Returns:
    True if profile_datum should be included.
  """
  if not node_name_regex.match(
      profile_datum.node_exec_stats.node_name):
    return False
  if profile_datum.file_line is not None and not file_name_regex.match(
      profile_datum.file_line):
    return False
  if profile_datum.op_type is not None and not op_type_regex.match(
      profile_datum.op_type):
    return False
  if op_time_interval is not None and not op_time_interval.contains(
      profile_datum.op_time):
    return False
  if exec_time_interval and not exec_time_interval.contains(
      profile_datum.node_exec_stats.all_end_rel_micros):
    return False
  return True


def _list_profile_sort_key(profile_datum, sort_by):
  """Get a profile_datum property to sort by in list_profile command.

  Args:
    profile_datum: A `ProfileDatum` object.
    sort_by: (string) indicates a value to sort by.
      Must be one of SORT_BY* constants.

  Returns:
    profile_datum property to sort by.
  """
  if sort_by == SORT_OPS_BY_OP_NAME:
    return profile_datum.node_exec_stats.node_name
  elif sort_by == SORT_OPS_BY_LINE:
    return profile_datum.file_line
  elif sort_by == SORT_OPS_BY_OP_TIME:
    return profile_datum.op_time
  elif sort_by == SORT_OPS_BY_EXEC_TIME:
    return profile_datum.node_exec_stats.all_end_rel_micros
  else:  # sort by start time
    return profile_datum.node_exec_stats.all_start_micros


class ProfileAnalyzer(object):
  """Analyzer for profiling data."""

  def __init__(self, graph, run_metadata):
    """ProfileAnalyzer constructor.

    Args:
      graph: (tf.Graph) Python graph object.
      run_metadata: A `RunMetadata` protobuf object.

    Raises:
      ValueError: If run_metadata is None.
    """
    self._graph = graph
    if not run_metadata:
      raise ValueError("No RunMetadata passed for profile analysis.")
    self._run_metadata = run_metadata
    self._arg_parsers = {}
    ap = argparse.ArgumentParser(
        description="List nodes profile information.",
        usage=argparse.SUPPRESS)
    ap.add_argument(
        "-d",
        "--device_name_filter",
        dest="device_name_filter",
        type=str,
        default="",
        help="filter device name by regex.")
    ap.add_argument(
        "-n",
        "--node_name_filter",
        dest="node_name_filter",
        type=str,
        default="",
        help="filter node name by regex.")
    ap.add_argument(
        "-t",
        "--op_type_filter",
        dest="op_type_filter",
        type=str,
        default="",
        help="filter op type by regex.")
    # TODO(annarev): allow file filtering at non-stack top position.
    ap.add_argument(
        "-f",
        "--file_name_filter",
        dest="file_name_filter",
        type=str,
        default="",
        help="filter by file name at the top position of node's creation "
             "stack that does not belong to TensorFlow library.")
    ap.add_argument(
        "-e",
        "--execution_time",
        dest="execution_time",
        type=str,
        default="",
        help="Filter by execution time interval "
             "(includes compute plus pre- and post -processing time). "
             "Supported units are s, ms and us (default). "
             "E.g. -e >100s, -e <100, -e [100us,1000ms]")
    ap.add_argument(
        "-o",
        "--op_time",
        dest="op_time",
        type=str,
        default="",
        help="Filter by op time interval (only includes compute time). "
             "Supported units are s, ms and us (default). "
             "E.g. -e >100s, -e <100, -e [100us,1000ms]")
    ap.add_argument(
        "-s",
        "--sort_by",
        dest="sort_by",
        type=str,
        default=SORT_OPS_BY_START_TIME,
        help=("the field to sort the data by: (%s | %s | %s | %s | %s)" %
              (SORT_OPS_BY_OP_NAME, SORT_OPS_BY_START_TIME,
               SORT_OPS_BY_OP_TIME, SORT_OPS_BY_EXEC_TIME, SORT_OPS_BY_LINE)))
    ap.add_argument(
        "-r",
        "--reverse",
        dest="reverse",
        action="store_true",
        help="sort the data in reverse (descending) order")
    ap.add_argument(
        "--time_unit",
        dest="time_unit",
        type=str,
        default=cli_shared.TIME_UNIT_US,
        help="Time unit (" + " | ".join(cli_shared.TIME_UNITS) + ")")

    self._arg_parsers["list_profile"] = ap

  def list_profile(self, args, screen_info=None):
    """Command handler for list_profile.

    List per-operation profile information.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """
    del screen_info

    parsed = self._arg_parsers["list_profile"].parse_args(args)
    op_time_interval = (command_parser.parse_time_interval(parsed.op_time)
                        if parsed.op_time else None)
    exec_time_interval = (
        command_parser.parse_time_interval(parsed.execution_time)
        if parsed.execution_time else None)
    node_name_regex = re.compile(parsed.node_name_filter)
    file_name_regex = re.compile(parsed.file_name_filter)
    op_type_regex = re.compile(parsed.op_type_filter)

    output = debugger_cli_common.RichTextLines([""])
    device_name_regex = re.compile(parsed.device_name_filter)
    data_generator = self._get_profile_data_generator()
    device_count = len(self._run_metadata.step_stats.dev_stats)
    for index in range(device_count):
      device_stats = self._run_metadata.step_stats.dev_stats[index]
      if device_name_regex.match(device_stats.device):
        profile_data = [
            datum for datum in data_generator(device_stats)
            if _list_profile_filter(
                datum, node_name_regex, file_name_regex, op_type_regex,
                op_time_interval, exec_time_interval)]
        profile_data = sorted(
            profile_data,
            key=lambda datum: _list_profile_sort_key(datum, parsed.sort_by),
            reverse=parsed.reverse)
        output.extend(
            self._get_list_profile_lines(
                device_stats.device, index, device_count,
                profile_data, parsed.sort_by, parsed.reverse, parsed.time_unit))
    return output

  def _get_profile_data_generator(self):
    """Get function that generates `ProfileDatum` objects.

    Returns:
      A function that generates `ProfileDatum` objects.
    """
    node_to_file_line = {}
    node_to_op_type = {}
    for op in self._graph.get_operations():
      file_line = ""
      for trace_entry in reversed(op.traceback):
        filepath = trace_entry[0]
        file_line = "%s:%d(%s)" % (
            os.path.basename(filepath), trace_entry[1], trace_entry[2])
        if not source_utils.guess_is_tensorflow_py_library(filepath):
          break
      node_to_file_line[op.name] = file_line
      node_to_op_type[op.name] = op.type

    def profile_data_generator(device_step_stats):
      for node_stats in device_step_stats.node_stats:
        if node_stats.node_name == "_SOURCE" or node_stats.node_name == "_SINK":
          continue
        yield ProfileDatum(
            node_stats,
            node_to_file_line.get(node_stats.node_name, ""),
            node_to_op_type.get(node_stats.node_name, ""))
    return profile_data_generator

  def _get_list_profile_lines(
      self, device_name, device_index, device_count,
      profile_datum_list, sort_by, sort_reverse, time_unit):
    """Get `RichTextLines` object for list_profile command for a given device.

    Args:
      device_name: (string) Device name.
      device_index: (int) Device index.
      device_count: (int) Number of devices.
      profile_datum_list: List of `ProfileDatum` objects.
      sort_by: (string) Identifier of column to sort. Sort identifier
          must match value of SORT_OPS_BY_OP_NAME, SORT_OPS_BY_EXEC_TIME,
          SORT_OPS_BY_MEMORY or SORT_OPS_BY_LINE.
      sort_reverse: (bool) Whether to sort in descending instead of default
          (ascending) order.
      time_unit: time unit, must be in cli_shared.TIME_UNITS.

    Returns:
      `RichTextLines` object containing a table that displays profiling
      information for each op.
    """
    profile_data = ProfileDataTableView(profile_datum_list, time_unit=time_unit)

    # Calculate total time early to calculate column widths.
    total_op_time = sum(datum.op_time for datum in profile_datum_list)
    total_exec_time = sum(datum.node_exec_stats.all_end_rel_micros
                          for datum in profile_datum_list)
    device_total_row = [
        "Device Total", "",
        cli_shared.time_to_readable_str(total_op_time,
                                        force_time_unit=time_unit),
        cli_shared.time_to_readable_str(total_exec_time,
                                        force_time_unit=time_unit)]

    # Calculate column widths.
    column_widths = [
        len(column_name) for column_name in profile_data.column_names()]
    for col in range(len(device_total_row)):
      column_widths[col] = max(column_widths[col], len(device_total_row[col]))
    for col in range(len(column_widths)):
      for row in range(profile_data.row_count()):
        column_widths[col] = max(
            column_widths[col], len(str(profile_data.value(row, col))))
      column_widths[col] += 2  # add margin between columns

    # Add device name.
    output = debugger_cli_common.RichTextLines(["-"*80])
    device_row = "Device %d of %d: %s" % (
        device_index + 1, device_count, device_name)
    output.extend(debugger_cli_common.RichTextLines([device_row, ""]))

    # Add headers.
    base_command = "list_profile"
    attr_segs = {0: []}
    row = ""
    for col in range(profile_data.column_count()):
      column_name = profile_data.column_names()[col]
      sort_id = profile_data.column_sort_id(col)
      command = "%s -s %s" % (base_command, sort_id)
      if sort_by == sort_id and not sort_reverse:
        command += " -r"
      curr_row = ("{:<%d}" % column_widths[col]).format(column_name)
      prev_len = len(row)
      row += curr_row
      attr_segs[0].append(
          (prev_len, prev_len + len(column_name),
           [debugger_cli_common.MenuItem(None, command), "bold"]))

    output.extend(
        debugger_cli_common.RichTextLines([row], font_attr_segs=attr_segs))

    # Add data rows.
    for row in range(profile_data.row_count()):
      row_str = ""
      for col in range(profile_data.column_count()):
        row_str += ("{:<%d}" % column_widths[col]).format(
            profile_data.value(row, col))
      output.extend(debugger_cli_common.RichTextLines([row_str]))

    # Add stat totals.
    row_str = ""
    for col in range(len(device_total_row)):
      row_str += ("{:<%d}" % column_widths[col]).format(device_total_row[col])
    output.extend(debugger_cli_common.RichTextLines(""))
    output.extend(debugger_cli_common.RichTextLines(row_str))
    return output

  def _measure_list_profile_column_widths(self, profile_data):
    """Determine the maximum column widths for each data list.

    Args:
      profile_data: list of ProfileDatum objects.

    Returns:
      List of column widths in the same order as columns in data.
    """
    num_columns = len(profile_data.column_names())
    widths = [len(column_name) for column_name in profile_data.column_names()]
    for row in range(profile_data.row_count()):
      for col in range(num_columns):
        widths[col] = max(
            widths[col], len(str(profile_data.row_values(row)[col])) + 2)
    return widths

  def get_help(self, handler_name):
    return self._arg_parsers[handler_name].format_help()


def create_profiler_ui(graph,
                       run_metadata,
                       ui_type="curses",
                       on_ui_exit=None):
  """Create an instance of CursesUI based on a `tf.Graph` and `RunMetadata`.

  Args:
    graph: Python `Graph` object.
    run_metadata: A `RunMetadata` protobuf object.
    ui_type: (str) requested UI type, e.g., "curses", "readline".
    on_ui_exit: (`Callable`) the callback to be called when the UI exits.

  Returns:
    (base_ui.BaseUI) A BaseUI subtype object with a set of standard analyzer
      commands and tab-completions registered.
  """

  analyzer = ProfileAnalyzer(graph, run_metadata)

  cli = ui_factory.get_ui(ui_type, on_ui_exit=on_ui_exit)
  cli.register_command_handler(
      "list_profile",
      analyzer.list_profile,
      analyzer.get_help("list_profile"),
      prefix_aliases=["lp"])

  return cli
