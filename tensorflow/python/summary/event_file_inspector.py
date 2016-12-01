# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Logic for TensorBoard inspector to help humans investigate event files.

Example usages:
tensorboard --inspect --event_file=myevents.out
tensorboard --inspect --event_file=myevents.out --tag=loss
tensorboard --inspect --logdir=mylogdir
tensorboard --inspect --logdir=mylogdir --tag=loss


This script runs over a logdir and creates an InspectionUnit for every
subdirectory with event files. If running over an event file, it creates only
one InspectionUnit. One block of output is printed to console for each
InspectionUnit.

The primary content of an InspectionUnit is the dict field_to_obs that maps
fields (e.g. "scalar", "histogram", "session_log:start", etc.) to a list of
Observations for the field. Observations correspond one-to-one with Events in an
event file but contain less information because they only store what is
necessary to generate the final console output.

The final output is rendered to console by applying some aggregating function
to the lists of Observations. Different functions are applied depending on the
type of field. For instance, for "scalar" fields, the inspector shows aggregate
statistics. For other fields like "session_log:start", all observed steps are
printed in order to aid debugging.


[1] Query a logdir or an event file for its logged tags and summary statistics
using --logdir or --event_file.

[[event_file]] contains these tags:
histograms
   binary/Sign/Activations
   binary/nn_tanh/act/Activations
   binary/nn_tanh/biases
   binary/nn_tanh/biases:gradient
   binary/nn_tanh/weights
   binary/nn_tanh/weights:gradient
images
   input_images/image/0
   input_images/image/1
   input_images/image/2
scalars
   Learning Rate
   Total Cost
   Total Cost (raw)

Debug output aggregated over all tags:
graph
   first_step           0
   last_step            0
   max_step             0
   min_step             0
   num_steps            1
   outoforder_steps     []
histograms
   first_step           491
   last_step            659823
   max_step             659823
   min_step             491
   num_steps            993
   outoforder_steps     []
images -
scalars
   first_step           0
   last_step            659823
   max_step             659823
   min_step             0
   num_steps            1985
   outoforder_steps     []
sessionlog:checkpoint
   first_step           7129
   last_step            657167
   max_step             657167
   min_step             7129
   num_steps            99
   outoforder_steps     []
sessionlog:start
   outoforder_steps     []
   steps                [0L]
sessionlog:stop -


[2] Drill down into a particular tag using --tag.

Debug output for binary/Sign/Activations:
histograms
   first_step           491
   last_step            659823
   max_step             659823
   min_step             491
   num_steps            993
   outoforder_steps     []
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os

from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.summary import event_accumulator
from tensorflow.python.summary import event_multiplexer
from tensorflow.python.summary.impl import event_file_loader

FLAGS = flags.FLAGS


# Map of field names within summary.proto to the user-facing names that this
# script outputs.
SUMMARY_TYPE_TO_FIELD = {'simple_value': 'scalars',
                         'histo': 'histograms',
                         'image': 'images',
                         'audio': 'audio'}
for summary_type in event_accumulator.SUMMARY_TYPES:
  if summary_type not in SUMMARY_TYPE_TO_FIELD:
    SUMMARY_TYPE_TO_FIELD[summary_type] = summary_type

# Types of summaries that we may want to query for by tag.
TAG_FIELDS = list(SUMMARY_TYPE_TO_FIELD.values())

# Summaries that we want to see every instance of.
LONG_FIELDS = ['sessionlog:start', 'sessionlog:stop']

# Summaries that we only want an abridged digest of, since they would
# take too much screen real estate otherwise.
SHORT_FIELDS = ['graph', 'sessionlog:checkpoint'] + TAG_FIELDS

# All summary types that we can inspect.
TRACKED_FIELDS = SHORT_FIELDS + LONG_FIELDS

# An `Observation` contains the data within each Event file that the inspector
# cares about. The inspector accumulates Observations as it processes events.
Observation = collections.namedtuple('Observation', ['step', 'wall_time',
                                                     'tag'])

# An InspectionUnit is created for each organizational structure in the event
# files visible in the final terminal output. For instance, one InspectionUnit
# is created for each subdirectory in logdir. When asked to inspect a single
# event file, there may only be one InspectionUnit.

# The InspectionUnit contains the `name` of the organizational unit that will be
# printed to console, a `generator` that yields `Event` protos, and a mapping
# from string fields to `Observations` that the inspector creates.
InspectionUnit = collections.namedtuple('InspectionUnit', ['name', 'generator',
                                                           'field_to_obs'])

PRINT_SEPARATOR = '=' * 70 + '\n'


def get_field_to_observations_map(generator, query_for_tag=''):
  """Return a field to `Observations` dict for the event generator.

  Args:
    generator: A generator over event protos.
    query_for_tag: A string that if specified, only create observations for
      events with this tag name.

  Returns:
    A dict mapping keys in `TRACKED_FIELDS` to an `Observation` list.
  """

  def increment(stat, event, tag=''):
    assert stat in TRACKED_FIELDS
    field_to_obs[stat].append(Observation(step=event.step,
                                          wall_time=event.wall_time,
                                          tag=tag)._asdict())

  field_to_obs = dict([(t, []) for t in TRACKED_FIELDS])

  for event in generator:
    ## Process the event
    if event.HasField('graph_def') and (not query_for_tag):
      increment('graph', event)
    if event.HasField('session_log') and (not query_for_tag):
      status = event.session_log.status
      if status == SessionLog.START:
        increment('sessionlog:start', event)
      elif status == SessionLog.STOP:
        increment('sessionlog:stop', event)
      elif status == SessionLog.CHECKPOINT:
        increment('sessionlog:checkpoint', event)
    elif event.HasField('summary'):
      for value in event.summary.value:
        if query_for_tag and value.tag != query_for_tag:
          continue

        for proto_name, display_name in SUMMARY_TYPE_TO_FIELD.items():
          if value.HasField(proto_name):
            increment(display_name, event, value.tag)
  return field_to_obs


def get_unique_tags(field_to_obs):
  """Returns a dictionary of tags that a user could query over.

  Args:
    field_to_obs: Dict that maps string field to `Observation` list.

  Returns:
    A dict that maps keys in `TAG_FIELDS` to a list of string tags present in
    the event files. If the dict does not have any observations of the type,
    maps to an empty list so that we can render this to console.
  """
  return {field: sorted(set([x.get('tag', '') for x in observations]))
          for field, observations in field_to_obs.items()
          if field in TAG_FIELDS}


def print_dict(d, show_missing=True):
  """Prints a shallow dict to console.

  Args:
    d: Dict to print.
    show_missing: Whether to show keys with empty values.
  """
  for k, v in sorted(d.items()):
    if (not v) and show_missing:
      # No instances of the key, so print missing symbol.
      print('{} -'.format(k))
    elif isinstance(v, list):
      # Value is a list, so print each item of the list.
      print(k)
      for item in v:
        print('   {}'.format(item))
    elif isinstance(v, dict):
      # Value is a dict, so print each (key, value) pair of the dict.
      print(k)
      for kk, vv in sorted(v.items()):
        print('   {:<20} {}'.format(kk, vv))


def get_dict_to_print(field_to_obs):
  """Transform the field-to-obs mapping into a printable dictionary.

  Args:
    field_to_obs: Dict that maps string field to `Observation` list.

  Returns:
    A dict with the keys and values to print to console.
  """

  def compressed_steps(steps):
    return {'num_steps': len(set(steps)),
            'min_step': min(steps),
            'max_step': max(steps),
            'last_step': steps[-1],
            'first_step': steps[0],
            'outoforder_steps': get_out_of_order(steps)}

  def full_steps(steps):
    return {'steps': steps, 'outoforder_steps': get_out_of_order(steps)}

  output = {}
  for field, observations in field_to_obs.items():
    if not observations:
      output[field] = None
      continue

    steps = [x['step'] for x in observations]
    if field in SHORT_FIELDS:
      output[field] = compressed_steps(steps)
    if field in LONG_FIELDS:
      output[field] = full_steps(steps)

  return output


def get_out_of_order(list_of_numbers):
  """Returns elements that break the monotonically non-decreasing trend.

  This is used to find instances of global step values that are "out-of-order",
  which may trigger TensorBoard event discarding logic.

  Args:
    list_of_numbers: A list of numbers.

  Returns:
    A list of tuples in which each tuple are two elements are adjacent, but the
    second element is lower than the first.
  """
  # TODO(cassandrax): Consider changing this to only check for out-of-order
  # steps within a particular tag.
  result = []
  for i in range(len(list_of_numbers)):
    if i == 0:
      continue
    if list_of_numbers[i] < list_of_numbers[i - 1]:
      result.append((list_of_numbers[i - 1], list_of_numbers[i]))
  return result


def generators_from_logdir(logdir):
  """Returns a list of event generators for subdirectories with event files.

  The number of generators returned should equal the number of directories
  within logdir that contain event files. If only logdir contains event files,
  returns a list of length one.

  Args:
    logdir: A log directory that contains event files.

  Returns:
    List of event generators for each subdirectory with event files.
  """
  subdirs = event_multiplexer.GetLogdirSubdirectories(logdir)
  generators = [itertools.chain(*[
      generator_from_event_file(os.path.join(subdir, f))
      for f in gfile.ListDirectory(subdir)
      if event_accumulator.IsTensorFlowEventsFile(os.path.join(subdir, f))
  ]) for subdir in subdirs]
  return generators


def generator_from_event_file(event_file):
  """Returns a generator that yields events from an event file."""
  return event_file_loader.EventFileLoader(event_file).Load()


def get_inspection_units(logdir='', event_file='', tag=''):
  """Returns a list of InspectionUnit objects given either logdir or event_file.

  If logdir is given, the number of InspectionUnits should equal the
  number of directories or subdirectories that contain event files.

  If event_file is given, the number of InspectionUnits should be 1.

  Args:
    logdir: A log directory that contains event files.
    event_file: Or, a particular event file path.
    tag: An optional tag name to query for.

  Returns:
    A list of InspectionUnit objects.
  """
  if logdir:
    subdirs = event_multiplexer.GetLogdirSubdirectories(logdir)
    inspection_units = []
    for subdir in subdirs:
      generator = itertools.chain(*[
          generator_from_event_file(os.path.join(subdir, f))
          for f in gfile.ListDirectory(subdir)
          if event_accumulator.IsTensorFlowEventsFile(os.path.join(subdir, f))
      ])
      inspection_units.append(InspectionUnit(
          name=subdir,
          generator=generator,
          field_to_obs=get_field_to_observations_map(generator, tag)))
    if inspection_units:
      print('Found event files in:\n{}\n'.format('\n'.join(
          [u.name for u in inspection_units])))
    elif event_accumulator.IsTensorFlowEventsFile(logdir):
      print(
          'It seems that {} may be an event file instead of a logdir. If this '
          'is the case, use --event_file instead of --logdir to pass '
          'it in.'.format(logdir))
    else:
      print('No event files found within logdir {}'.format(logdir))
    return inspection_units
  elif event_file:
    generator = generator_from_event_file(event_file)
    return [InspectionUnit(
        name=event_file,
        generator=generator,
        field_to_obs=get_field_to_observations_map(generator, tag))]


def inspect(logdir='', event_file='', tag=''):
  """Main function for inspector that prints out a digest of event files.

  Args:
    logdir: A log directory that contains event files.
    event_file: Or, a particular event file path.
    tag: An optional tag name to query for.

  Raises:
    ValueError: If neither logdir and event_file are given, or both are given.
  """
  if logdir and event_file:
    raise ValueError(
        'Must specify either --logdir or --event_file, but not both.')
  if not (logdir or event_file):
    raise ValueError('Must specify either --logdir or --event_file.')

  print(PRINT_SEPARATOR +
        'Processing event files... (this can take a few minutes)\n' +
        PRINT_SEPARATOR)
  inspection_units = get_inspection_units(logdir, event_file, tag)

  for unit in inspection_units:
    if tag:
      print('Event statistics for tag {} in {}:'.format(tag, unit.name))
    else:
      # If the user is not inspecting a particular tag, also print the list of
      # all available tags that they can query.
      print('These tags are in {}:'.format(unit.name))
      print_dict(get_unique_tags(unit.field_to_obs))
      print(PRINT_SEPARATOR)
      print('Event statistics for {}:'.format(unit.name))

    print_dict(get_dict_to_print(unit.field_to_obs), show_missing=(not tag))
    print(PRINT_SEPARATOR)


if __name__ == '__main__':
  app.run()
