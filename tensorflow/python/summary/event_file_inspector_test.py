# Copyright 2015 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from tensorflow.core.framework.summary_pb2 import HistogramProto
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.summary import event_file_inspector as efi
from tensorflow.python.training.summary_io import SummaryWriter


class EventFileInspectorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.logdir = os.path.join(self.get_temp_dir(), 'tfevents')
    self._MakeDirectoryIfNotExists(self.logdir)

  def tearDown(self):
    shutil.rmtree(self.logdir)

  def _MakeDirectoryIfNotExists(self, path):
    if not os.path.exists(path):
      os.mkdir(path)

  def _WriteScalarSummaries(self, data, subdirs=('',)):
    # Writes data to a tempfile in subdirs, and returns generator for the data.
    # If subdirs is given, writes data identically to all subdirectories.
    for subdir_ in subdirs:
      subdir = os.path.join(self.logdir, subdir_)
      self._MakeDirectoryIfNotExists(subdir)

      sw = SummaryWriter(subdir)
      for datum in data:
        summary = Summary()
        if 'simple_value' in datum:
          summary.value.add(tag=datum['tag'],
                            simple_value=datum['simple_value'])
          sw.add_summary(summary, global_step=datum['step'])
        elif 'histo' in datum:
          summary.value.add(tag=datum['tag'], histo=HistogramProto())
          sw.add_summary(summary, global_step=datum['step'])
        elif 'session_log' in datum:
          sw.add_session_log(datum['session_log'], global_step=datum['step'])
      sw.close()

  def testEmptyLogdir(self):
    # Nothing was wrriten to logdir
    units = efi.get_inspection_units(self.logdir)
    self.assertEqual([], units)

  def testGetAvailableTags(self):
    data = [{'tag': 'c', 'histo': 2, 'step': 10},
            {'tag': 'c', 'histo': 2, 'step': 11},
            {'tag': 'c', 'histo': 2, 'step': 9},
            {'tag': 'b', 'simple_value': 2, 'step': 20},
            {'tag': 'b', 'simple_value': 2, 'step': 15},
            {'tag': 'a', 'simple_value': 2, 'step': 3}]
    self._WriteScalarSummaries(data)
    units = efi.get_inspection_units(self.logdir)
    tags = efi.get_unique_tags(units[0].field_to_obs)
    self.assertEqual(['a', 'b'], tags['scalars'])
    self.assertEqual(['c'], tags['histograms'])

  def testInspectAll(self):
    data = [{'tag': 'c', 'histo': 2, 'step': 10},
            {'tag': 'c', 'histo': 2, 'step': 11},
            {'tag': 'c', 'histo': 2, 'step': 9},
            {'tag': 'b', 'simple_value': 2, 'step': 20},
            {'tag': 'b', 'simple_value': 2, 'step': 15},
            {'tag': 'a', 'simple_value': 2, 'step': 3}]
    self._WriteScalarSummaries(data)
    units = efi.get_inspection_units(self.logdir)
    printable = efi.get_dict_to_print(units[0].field_to_obs)
    self.assertEqual(printable['histograms']['max_step'], 11)
    self.assertEqual(printable['histograms']['min_step'], 9)
    self.assertEqual(printable['histograms']['num_steps'], 3)
    self.assertEqual(printable['histograms']['last_step'], 9)
    self.assertEqual(printable['histograms']['first_step'], 10)
    self.assertEqual(printable['histograms']['outoforder_steps'], [(11, 9)])

    self.assertEqual(printable['scalars']['max_step'], 20)
    self.assertEqual(printable['scalars']['min_step'], 3)
    self.assertEqual(printable['scalars']['num_steps'], 3)
    self.assertEqual(printable['scalars']['last_step'], 3)
    self.assertEqual(printable['scalars']['first_step'], 20)
    self.assertEqual(printable['scalars']['outoforder_steps'], [(20, 15),
                                                                (15, 3)])

  def testInspectTag(self):
    data = [{'tag': 'c', 'histo': 2, 'step': 10},
            {'tag': 'c', 'histo': 2, 'step': 11},
            {'tag': 'c', 'histo': 2, 'step': 9},
            {'tag': 'b', 'histo': 2, 'step': 20},
            {'tag': 'b', 'simple_value': 2, 'step': 15},
            {'tag': 'a', 'simple_value': 2, 'step': 3}]
    self._WriteScalarSummaries(data)
    units = efi.get_inspection_units(self.logdir, tag='c')
    printable = efi.get_dict_to_print(units[0].field_to_obs)
    self.assertEqual(printable['histograms']['max_step'], 11)
    self.assertEqual(printable['histograms']['min_step'], 9)
    self.assertEqual(printable['histograms']['num_steps'], 3)
    self.assertEqual(printable['histograms']['last_step'], 9)
    self.assertEqual(printable['histograms']['first_step'], 10)
    self.assertEqual(printable['histograms']['outoforder_steps'], [(11, 9)])
    self.assertEqual(printable['scalars'], None)

  def testSessionLogSummaries(self):
    data = [
        {'session_log': SessionLog(status=SessionLog.START), 'step': 0},
        {'session_log': SessionLog(status=SessionLog.CHECKPOINT), 'step': 1},
        {'session_log': SessionLog(status=SessionLog.CHECKPOINT), 'step': 2},
        {'session_log': SessionLog(status=SessionLog.CHECKPOINT), 'step': 3},
        {'session_log': SessionLog(status=SessionLog.STOP), 'step': 4},
        {'session_log': SessionLog(status=SessionLog.START), 'step': 5},
        {'session_log': SessionLog(status=SessionLog.STOP), 'step': 6},
    ]

    self._WriteScalarSummaries(data)
    units = efi.get_inspection_units(self.logdir)
    self.assertEqual(1, len(units))
    printable = efi.get_dict_to_print(units[0].field_to_obs)
    self.assertEqual(printable['sessionlog:start']['steps'], [0, 5])
    self.assertEqual(printable['sessionlog:stop']['steps'], [4, 6])
    self.assertEqual(printable['sessionlog:checkpoint']['num_steps'], 3)

  def testInspectAllWithNestedLogdirs(self):
    data = [{'tag': 'c', 'simple_value': 2, 'step': 10},
            {'tag': 'c', 'simple_value': 2, 'step': 11},
            {'tag': 'c', 'simple_value': 2, 'step': 9},
            {'tag': 'b', 'simple_value': 2, 'step': 20},
            {'tag': 'b', 'simple_value': 2, 'step': 15},
            {'tag': 'a', 'simple_value': 2, 'step': 3}]

    subdirs = ['eval', 'train']
    self._WriteScalarSummaries(data, subdirs=subdirs)
    units = efi.get_inspection_units(self.logdir)
    self.assertEqual(2, len(units))
    directory_names = [os.path.join(self.logdir, name) for name in subdirs]
    self.assertEqual(directory_names, sorted([unit.name for unit in units]))

    for unit in units:
      printable = efi.get_dict_to_print(unit.field_to_obs)['scalars']
      self.assertEqual(printable['max_step'], 20)
      self.assertEqual(printable['min_step'], 3)
      self.assertEqual(printable['num_steps'], 6)
      self.assertEqual(printable['last_step'], 3)
      self.assertEqual(printable['first_step'], 10)
      self.assertEqual(printable['outoforder_steps'], [(11, 9), (20, 15),
                                                       (15, 3)])

if __name__ == '__main__':
  googletest.main()
