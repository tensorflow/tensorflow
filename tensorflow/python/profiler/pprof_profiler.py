# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Profiler for TensorFlow models that outputs data in pprof format.

See https://github.com/google/pprof/blob/master/proto/profile.proto for pprof
profile format.
The following needs to be set for profiler to work:
  * trace_level needs to be set to FULL_TRACE
  * run_metadata object should be passed in to session.run call

Sample usage:
  options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.compat.v1.RunMetadata()

  with tf.compat.v1.Session as sess:
    ...
    sess.run(computation, run_metadata=run_metadata, options=options)
  pprof_profiler.profile(sess.graph, run_metadata, output_dir)


  The code above would output a pprof profile to separate output_dir/.*.pb.gz
  file for each device. These files can be passed to pprof for formatting.
  For e.g.:
     pprof -png --nodecount=100 --sample_index=1 output_dir/profile_output.pb.gz
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import namedtuple
import gzip
import os
import string
import sys
import time

from proto import profile_pb2


if sys.version_info < (3,):
  maketrans = string.maketrans
else:
  maketrans = str.maketrans


ProfileDatum = namedtuple('ProfileDatum', [
    'node_exec_stats', 'op_type', 'traceback'])


class StringTable(object):
  """Keeps track of strings to add to string_table in pprof proto."""

  def __init__(self):
    # Pprof requires first entry in string_table to be ''.
    self._string_table = ['']
    self._string_to_index = {'': 0}

  def index_of(self, value_str):
    """Get index of value_str in the string table.

    If value_str is not in the string table, we will add it at the end
    and then return the new index.
    Args:
      value_str: (string) Value to lookup/add in/to the string table.

    Returns:
      Index of value_str in the string table.
    """
    if value_str is None:
      value_str = ''
    if value_str in self._string_to_index:
      return self._string_to_index[value_str]
    index = len(self._string_table)
    self._string_table.append(value_str)
    self._string_to_index[value_str] = index
    return index

  def next_index(self):
    """Gets index that would be assigned to the next added string.

    Returns:
      Index of the next string if it was added.
    """
    return len(self._string_table)

  def string_table(self):
    """Returns a list of strings to store in pprof's string_table."""
    return self._string_table


class Functions(object):
  """Keeps track of `Function` protos for pprof profile."""

  def __init__(self, string_table):
    """Constructor.

    Args:
      string_table: A `StringTable` object.
    """
    self._string_table = string_table
    # Maps tuples in the form (file_path, function_name, start_line_number)
    # to `Function` protos.
    self._function_key_to_function = {}

  def index_of(self, file_path, function_name, function_start_line):
    """Returns index of the function, adding the function if needed.

    Args:
      file_path: (string) Path to file where the function is defined.
      function_name: (string) Function name.
      function_start_line: (integer) Start line number of function definition.

    Returns:
      Function index.
    """
    function_key = (file_path, function_name, function_start_line)
    if function_key in self._function_key_to_function:
      return self._function_key_to_function[function_key].id
    else:
      # Function indexes should start from 1
      function_index = len(self._function_key_to_function) + 1
      function = profile_pb2.Function()
      function.id = function_index
      function.name = self._string_table.index_of(function_name)
      function.filename = self._string_table.index_of(file_path)
      function.start_line = function_start_line
      self._function_key_to_function[function_key] = function
      return function_index

  def function_protos(self):
    """Returns list of `profile_pb2.Function` protos."""
    return self._function_key_to_function.values()


class Locations(object):
  """Keeps track of `Location` protos for pprof profile.

  `Locations` store information about function call locations.
  """

  def __init__(self, functions):
    """Constructor.

    Args:
      functions: A `Functions` object.
    """
    self._functions = functions
    # Maps tuples in the form (file_path, called_function_name, line_number)
    # to `Location` protos.
    self._location_key_to_location = {}

  def index_of(
      self, file_path, line_number, called_function_name, called_file_path,
      called_function_start_line):
    """Returns index of the location, adding the location if needed.

    Args:
      file_path: (string) Path to file that makes the call.
      line_number: (integer) Call line number.
      called_function_name: (string) Function name of the function called at
        `file_path` and `line_number`.
      called_file_path: (string) Path to file where the called function is
        defined.
      called_function_start_line: (integer) Start line number of called
        function definition in `called_file_path` file.

    Returns:
      Index of location.
    """
    location_key = (file_path, called_function_name, line_number)
    if location_key in self._location_key_to_location:
      location = self._location_key_to_location[location_key]
      return location.id
    else:
      # Location indexes should start from 1
      location_index = len(self._location_key_to_location) + 1
      location = profile_pb2.Location()
      location.id = location_index
      self._location_key_to_location[location_key] = location

      line = location.line.add()
      line.function_id = self._functions.index_of(
          called_file_path, called_function_name, called_function_start_line)
      line.line = line_number
      return location_index

  def location_protos(self):
    """Returns list of `profile_pb2.Location` protos."""
    return self._location_key_to_location.values()


class Samples(object):
  """Keeps track of `Sample` protos for pprof profile.

  Samples store the following statistics in order:
  count, all_time, op_time
  """

  def __init__(self, string_table):
    """Constructor.

    Args:
      string_table: A `StringTable` object.
    """
    self._string_table = string_table
    # TODO(annarev): figure out if location is unique for each node name.
    # If not, also key this dictionary based on location ids.
    self._node_name_to_sample = {}

  def add(self, datum, location_ids):
    """Adds a sample data point.

    Args:
      datum: `ProfileDatum` to add a sample for.
      location_ids: List of numberic location ids for this
        sample.
    """
    node_name = datum.node_exec_stats.node_name
    if node_name in self._node_name_to_sample:
      sample = self._node_name_to_sample[node_name]
      sample.location_id.extend(location_ids)
    else:
      sample = profile_pb2.Sample()
      # Sample stores 3 values: count, all_time, op_time
      sample.value.extend([0, 0, 0])

      label = sample.label.add()
      label.key = self._string_table.index_of('node_name')
      label.str = self._string_table.index_of(node_name)
      label = sample.label.add()
      label.key = self._string_table.index_of('op_type')
      label.str = self._string_table.index_of(datum.op_type)
      self._node_name_to_sample[node_name] = sample
    sample.value[0] += 1
    sample.value[1] += datum.node_exec_stats.all_end_rel_micros
    sample.value[2] += (
        datum.node_exec_stats.op_end_rel_micros -
        datum.node_exec_stats.op_start_rel_micros)

  def get_sample_protos(self):
    """Returns list of `Sample` protos for pprof profile."""
    return self._node_name_to_sample.values()


class PprofProfiler(object):
  """Creates profiles in pprof format."""

  def __init__(self, graph, run_metadata):
    """Constructor.

    Args:
      graph: A `Graph` instance.
      run_metadata: A list of `RunMetadata` objects.
    """
    self._graph = graph
    self._run_metadata = run_metadata
    self._string_table = StringTable()
    self._functions = Functions(self._string_table)
    self._locations = Locations(self._functions)

  def profile(self):
    """Generates pprof profiles.

    Returns:
      Dictionary mapping from device name to proto in `profile_pb2.Profile`
      format.
    """
    profiles = {}
    data_generator_func = self._get_profile_data_generator()
    for device_index, device_stats in enumerate(
        self._run_metadata.step_stats.dev_stats):
      # Create profile
      pprof_proto = self._get_pprof_proto(data_generator_func(device_stats))
      if not pprof_proto.sample:
        print(
            'Not enough data to create profile for device %s. Did you pass '
            'RunMetadata to session.run call?' % device_stats.device)
        continue
      # Add device name comment
      device_count = len(self._run_metadata.step_stats.dev_stats)
      device_description = (
          'Device %d of %d: %s' %
          (device_index + 1, device_count, device_stats.device))
      device_description_str_index = self._string_table.next_index()
      pprof_proto.string_table.append(device_description)
      pprof_proto.comment.append(device_description_str_index)
      profiles[device_stats.device] = pprof_proto
    return profiles

  def _get_pprof_proto(self, profile_datum_generator):
    """Returns profile data in pprof proto format.

    Args:
      profile_datum_generator: Generator outputting `ProfileDatum` objects.

    Returns:
      A proto in pprof format.
    """
    pprof_profile = profile_pb2.Profile()
    samples = Samples(self._string_table)

    for datum in profile_datum_generator:
      if not datum.traceback:
        continue

      stack_frame = datum.traceback[-1]
      after_apply_op = False
      location_ids = []

      # We add locations from stack trace in bottom-up order.
      for stack_frame_index in reversed(range(len(datum.traceback) - 1)):
        prev_stack_frame = stack_frame
        stack_frame = datum.traceback[stack_frame_index]

        # Call at current frame calls function at previous frame.
        prev_file_path = prev_stack_frame[0]
        prev_function = prev_stack_frame[2]
        prev_function_start_line = -1
        curr_file_path = stack_frame[0]
        curr_line_number = stack_frame[1]

        # Skip all calls up to apply_op since they are the same for all ops.
        if not after_apply_op:
          if prev_function == 'apply_op':
            after_apply_op = True
          continue
        location_index = self._locations.index_of(
            curr_file_path, curr_line_number,
            prev_function, prev_file_path, prev_function_start_line)
        location_ids.append(location_index)
      samples.add(datum, location_ids)

    sample_type_description = 'count'
    sample_type = pprof_profile.sample_type.add()
    sample_type.type = self._string_table.index_of(sample_type_description)
    sample_type.unit = self._string_table.index_of('count')
    sample_type_description = 'all_time'
    sample_type = pprof_profile.sample_type.add()
    sample_type.type = self._string_table.index_of(sample_type_description)
    sample_type.unit = self._string_table.index_of('nanoseconds')
    sample_type_description = 'op_time'
    sample_type = pprof_profile.sample_type.add()
    sample_type.type = self._string_table.index_of(sample_type_description)
    sample_type.unit = self._string_table.index_of('nanoseconds')

    pprof_profile.string_table.extend(self._string_table.string_table())
    pprof_profile.sample.extend(samples.get_sample_protos())
    pprof_profile.function.extend(self._functions.function_protos())
    pprof_profile.location.extend(self._locations.location_protos())
    return pprof_profile

  def _get_profile_data_generator(self):
    """Get function that generates `ProfileDatum` objects.

    Returns:
      A function that generates `ProfileDatum` objects.
    """
    node_to_traceback = defaultdict(list)
    node_to_op_type = defaultdict(str)
    for op in self._graph.get_operations():
      node_to_traceback[op.name] = op.traceback
      node_to_op_type[op.name] = op.type

    def profile_data_generator(device_step_stats):
      for node_stats in device_step_stats.node_stats:
        if node_stats.node_name == '_SOURCE' or node_stats.node_name == '_SINK':
          continue
        yield ProfileDatum(
            node_stats,
            node_to_op_type[node_stats.node_name],
            node_to_traceback[node_stats.node_name])

    return profile_data_generator


def get_profiles(graph, run_metadata):
  """Generate profiles in pprof format.

  See https://github.com/google/pprof/blob/master/proto/profile.proto
  for pprof proto format.

  Args:
    graph: A `Graph` object.
    run_metadata: A `RunMetadata` proto.

  Returns:
    A dictionary mapping from device name to pprof proto for that device.
  """
  return PprofProfiler(graph, run_metadata).profile()


def profile(graph, run_metadata, output_dir=None):
  """Generate profiles in pprof format.

  See https://github.com/google/pprof/blob/master/proto/profile.proto
  for pprof proto format.

  Args:
    graph: A `Graph` object.
    run_metadata: A `RunMetadata` proto.
    output_dir: (string) Directory to output pprof profile to.
      Profile files for each device will be stored in compressed
      serialized proto format. If output_dir is None, profile protos
      will be printed to stdout instead.

  Returns:
    List of output files created by this profile call.
    (Note: this list will be empty if output_dir is None)
  """
  profiles = get_profiles(graph, run_metadata)
  output_file_template = None
  if output_dir:
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    time_suffix = time.strftime('%Y%m%d%H%M%S')
    output_file_template = os.path.join(
        output_dir, '%s_' + time_suffix + '.pb.gz')

  profile_files = []
  for device, pprof_proto in profiles.items():
    if output_file_template is None:
      print('No output directory specified, printing to stdout instead.')
      print(pprof_proto)
    else:
      device_name = str(device).strip('/').translate(
          maketrans('/:', '__'))
      profile_file = output_file_template % device_name
      profile_files.append(profile_file)
      with gzip.open(profile_file, 'w') as output_file:
        print('Writing profile to %s...' % profile_file)
        output_file.write(pprof_proto.SerializeToString())
  return profile_files
