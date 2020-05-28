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
"""Tests for V2 summary ops from summary_ops_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.core.util import event_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class SummaryOpsTest(test_util.TensorFlowTestCase):

  def tearDown(self):
    super(SummaryOpsTest, self).tearDown()
    summary_ops.trace_off()

  def keras_model(self, *args, **kwargs):
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir)
    with writer.as_default():
      summary_ops.keras_model(*args, **kwargs)
    writer.close()
    events = events_from_logdir(logdir)
    # The first event contains no summary values. The written content goes to
    # the second event.
    return events[1]

  @test_util.run_v2_only
  def testKerasModel(self):
    model = Sequential(
        [Dense(10, input_shape=(100,)),
         Activation('relu', name='my_relu')])
    event = self.keras_model(name='my_name', data=model, step=1)
    first_val = event.summary.value[0]
    self.assertEqual(model.to_json(), first_val.tensor.string_val[0].decode())

  @test_util.run_v2_only
  def testKerasModel_usesDefaultStep(self):
    model = Sequential(
        [Dense(10, input_shape=(100,)),
         Activation('relu', name='my_relu')])
    try:
      summary_ops.set_step(42)
      event = self.keras_model(name='my_name', data=model)
      self.assertEqual(42, event.step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  @test_util.run_v2_only
  def testKerasModel_subclass(self):

    class SimpleSubclass(Model):

      def __init__(self):
        super(SimpleSubclass, self).__init__(name='subclass')
        self.dense = Dense(10, input_shape=(100,))
        self.activation = Activation('relu', name='my_relu')

      def call(self, inputs):
        x = self.dense(inputs)
        return self.activation(x)

    model = SimpleSubclass()
    with test.mock.patch.object(logging, 'warn') as mock_log:
      self.assertFalse(
          summary_ops.keras_model(name='my_name', data=model, step=1))
      self.assertRegexpMatches(
          str(mock_log.call_args), 'Model failed to serialize as JSON.')

  @test_util.run_v2_only
  def testKerasModel_otherExceptions(self):
    model = Sequential()

    with test.mock.patch.object(model, 'to_json') as mock_to_json:
      with test.mock.patch.object(logging, 'warn') as mock_log:
        mock_to_json.side_effect = Exception('oops')
        self.assertFalse(
            summary_ops.keras_model(name='my_name', data=model, step=1))
        self.assertRegexpMatches(
            str(mock_log.call_args),
            'Model failed to serialize as JSON. Ignoring... oops')


def events_from_file(filepath):
  """Returns all events in a single event file.

  Args:
    filepath: Path to the event file.

  Returns:
    A list of all tf.Event protos in the event file.
  """
  records = list(tf_record.tf_record_iterator(filepath))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def events_from_logdir(logdir):
  """Returns all events in the single eventfile in logdir.

  Args:
    logdir: The directory in which the single event file is sought.

  Returns:
    A list of all tf.Event protos from the single event file.

  Raises:
    AssertionError: If logdir does not contain exactly one file.
  """
  assert gfile.Exists(logdir)
  files = gfile.ListDirectory(logdir)
  assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
  return events_from_file(os.path.join(logdir, files[0]))


if __name__ == '__main__':
  test.main()
