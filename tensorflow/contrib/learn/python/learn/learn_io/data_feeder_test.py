# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `DataFeeder`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

# pylint: disable=wildcard-import
from tensorflow.contrib.learn.python.learn.learn_io import *
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test

# pylint: enable=wildcard-import


class DataFeederTest(test.TestCase):
  # pylint: disable=undefined-variable
  """Tests for `DataFeeder`."""

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'base_dir')
    file_io.create_dir(self._base_dir)

  def tearDown(self):
    file_io.delete_recursively(self._base_dir)

  def _wrap_dict(self, data, prepend=''):
    return {prepend + '1': data, prepend + '2': data}

  def _assert_raises(self, input_data):
    with self.assertRaisesRegexp(TypeError, 'annot convert'):
      data_feeder.DataFeeder(input_data, None, n_classes=0, batch_size=1)

  def _assert_dtype(self, expected_np_dtype, expected_tf_dtype, input_data):
    feeder = data_feeder.DataFeeder(input_data, None, n_classes=0, batch_size=1)
    if isinstance(input_data, dict):
      for v in list(feeder.input_dtype.values()):
        self.assertEqual(expected_np_dtype, v)
    else:
      self.assertEqual(expected_np_dtype, feeder.input_dtype)
    with ops.Graph().as_default() as g, self.test_session(g):
      inp, _ = feeder.input_builder()
      if isinstance(inp, dict):
        for v in list(inp.values()):
          self.assertEqual(expected_tf_dtype, v.dtype)
      else:
        self.assertEqual(expected_tf_dtype, inp.dtype)

  def test_input_int8(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.int8)
    self._assert_dtype(np.int8, dtypes.int8, data)
    self._assert_dtype(np.int8, dtypes.int8, self._wrap_dict(data))

  def test_input_int16(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.int16)
    self._assert_dtype(np.int16, dtypes.int16, data)
    self._assert_dtype(np.int16, dtypes.int16, self._wrap_dict(data))

  def test_input_int32(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.int32)
    self._assert_dtype(np.int32, dtypes.int32, data)
    self._assert_dtype(np.int32, dtypes.int32, self._wrap_dict(data))

  def test_input_int64(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.int64)
    self._assert_dtype(np.int64, dtypes.int64, data)
    self._assert_dtype(np.int64, dtypes.int64, self._wrap_dict(data))

  def test_input_uint32(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.uint32)
    self._assert_dtype(np.uint32, dtypes.uint32, data)
    self._assert_dtype(np.uint32, dtypes.uint32, self._wrap_dict(data))

  def test_input_uint64(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.uint64)
    self._assert_dtype(np.uint64, dtypes.uint64, data)
    self._assert_dtype(np.uint64, dtypes.uint64, self._wrap_dict(data))

  def test_input_uint8(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.uint8)
    self._assert_dtype(np.uint8, dtypes.uint8, data)
    self._assert_dtype(np.uint8, dtypes.uint8, self._wrap_dict(data))

  def test_input_uint16(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.uint16)
    self._assert_dtype(np.uint16, dtypes.uint16, data)
    self._assert_dtype(np.uint16, dtypes.uint16, self._wrap_dict(data))

  def test_input_float16(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.float16)
    self._assert_dtype(np.float16, dtypes.float16, data)
    self._assert_dtype(np.float16, dtypes.float16, self._wrap_dict(data))

  def test_input_float32(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.float32)
    self._assert_dtype(np.float32, dtypes.float32, data)
    self._assert_dtype(np.float32, dtypes.float32, self._wrap_dict(data))

  def test_input_float64(self):
    data = np.matrix([[1, 2], [3, 4]], dtype=np.float64)
    self._assert_dtype(np.float64, dtypes.float64, data)
    self._assert_dtype(np.float64, dtypes.float64, self._wrap_dict(data))

  def test_input_bool(self):
    data = np.array([[False for _ in xrange(2)] for _ in xrange(2)])
    self._assert_dtype(np.bool, dtypes.bool, data)
    self._assert_dtype(np.bool, dtypes.bool, self._wrap_dict(data))

  def test_input_string(self):
    input_data = np.array([['str%d' % i for i in xrange(2)] for _ in xrange(2)])
    self._assert_dtype(input_data.dtype, dtypes.string, input_data)
    self._assert_dtype(input_data.dtype, dtypes.string,
                       self._wrap_dict(input_data))

  def _assertAllClose(self, src, dest, src_key_of=None, src_prop=None):

    def func(x):
      val = getattr(x, src_prop) if src_prop else x
      return val if src_key_of is None else src_key_of[val]

    if isinstance(src, dict):
      for k in list(src.keys()):
        self.assertAllClose(func(src[k]), dest)
    else:
      self.assertAllClose(func(src), dest)

  def test_unsupervised(self):

    def func(feeder):
      with self.test_session():
        inp, _ = feeder.input_builder()
        feed_dict_fn = feeder.get_feed_dict_fn()
        feed_dict = feed_dict_fn()
        self._assertAllClose(inp, [[1, 2]], feed_dict, 'name')

    data = np.matrix([[1, 2], [2, 3], [3, 4]])
    func(data_feeder.DataFeeder(data, None, n_classes=0, batch_size=1))
    func(
        data_feeder.DataFeeder(
            self._wrap_dict(data), None, n_classes=0, batch_size=1))

  def test_data_feeder_regression(self):

    def func(df):
      inp, out = df.input_builder()
      feed_dict_fn = df.get_feed_dict_fn()
      feed_dict = feed_dict_fn()
      self._assertAllClose(inp, [[3, 4], [1, 2]], feed_dict, 'name')
      self._assertAllClose(out, [2, 1], feed_dict, 'name')

    x = np.matrix([[1, 2], [3, 4]])
    y = np.array([1, 2])
    func(data_feeder.DataFeeder(x, y, n_classes=0, batch_size=3))
    func(
        data_feeder.DataFeeder(
            self._wrap_dict(x, 'in'),
            self._wrap_dict(y, 'out'),
            n_classes=self._wrap_dict(0, 'out'),
            batch_size=3))

  def test_epoch(self):

    def func(feeder):
      with self.test_session():
        feeder.input_builder()
        epoch = feeder.make_epoch_variable()
        feed_dict_fn = feeder.get_feed_dict_fn()
        # First input
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict[epoch.name], [0])
        # Second input
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict[epoch.name], [0])
        # Third input
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict[epoch.name], [0])
        # Back to the first input again, so new epoch.
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict[epoch.name], [1])

    data = np.matrix([[1, 2], [2, 3], [3, 4]])
    labels = np.array([0, 0, 1])
    func(data_feeder.DataFeeder(data, labels, n_classes=0, batch_size=1))
    func(
        data_feeder.DataFeeder(
            self._wrap_dict(data, 'in'),
            self._wrap_dict(labels, 'out'),
            n_classes=self._wrap_dict(0, 'out'),
            batch_size=1))

  def test_data_feeder_multioutput_regression(self):

    def func(df):
      inp, out = df.input_builder()
      feed_dict_fn = df.get_feed_dict_fn()
      feed_dict = feed_dict_fn()
      self._assertAllClose(inp, [[3, 4], [1, 2]], feed_dict, 'name')
      self._assertAllClose(out, [[3, 4], [1, 2]], feed_dict, 'name')

    x = np.matrix([[1, 2], [3, 4]])
    y = np.array([[1, 2], [3, 4]])
    func(data_feeder.DataFeeder(x, y, n_classes=0, batch_size=2))
    func(
        data_feeder.DataFeeder(
            self._wrap_dict(x, 'in'),
            self._wrap_dict(y, 'out'),
            n_classes=self._wrap_dict(0, 'out'),
            batch_size=2))

  def test_data_feeder_multioutput_classification(self):

    def func(df):
      inp, out = df.input_builder()
      feed_dict_fn = df.get_feed_dict_fn()
      feed_dict = feed_dict_fn()
      self._assertAllClose(inp, [[3, 4], [1, 2]], feed_dict, 'name')
      self._assertAllClose(
          out, [[[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]], feed_dict,
          'name')

    x = np.matrix([[1, 2], [3, 4]])
    y = np.array([[0, 1, 2], [2, 3, 4]])
    func(data_feeder.DataFeeder(x, y, n_classes=5, batch_size=2))
    func(
        data_feeder.DataFeeder(
            self._wrap_dict(x, 'in'),
            self._wrap_dict(y, 'out'),
            n_classes=self._wrap_dict(5, 'out'),
            batch_size=2))

  def test_streaming_data_feeder(self):

    def func(df):
      inp, out = df.input_builder()
      feed_dict_fn = df.get_feed_dict_fn()
      feed_dict = feed_dict_fn()
      self._assertAllClose(inp, [[[1, 2]], [[3, 4]]], feed_dict, 'name')
      self._assertAllClose(out, [[[1], [2]], [[2], [2]]], feed_dict, 'name')

    def x_iter(wrap_dict=False):
      yield np.array([[1, 2]]) if not wrap_dict else self._wrap_dict(
          np.array([[1, 2]]), 'in')
      yield np.array([[3, 4]]) if not wrap_dict else self._wrap_dict(
          np.array([[3, 4]]), 'in')

    def y_iter(wrap_dict=False):
      yield np.array([[1], [2]]) if not wrap_dict else self._wrap_dict(
          np.array([[1], [2]]), 'out')
      yield np.array([[2], [2]]) if not wrap_dict else self._wrap_dict(
          np.array([[2], [2]]), 'out')

    func(
        data_feeder.StreamingDataFeeder(
            x_iter(), y_iter(), n_classes=0, batch_size=2))
    func(
        data_feeder.StreamingDataFeeder(
            x_iter(True),
            y_iter(True),
            n_classes=self._wrap_dict(0, 'out'),
            batch_size=2))
    # Test non-full batches.
    func(
        data_feeder.StreamingDataFeeder(
            x_iter(), y_iter(), n_classes=0, batch_size=10))
    func(
        data_feeder.StreamingDataFeeder(
            x_iter(True),
            y_iter(True),
            n_classes=self._wrap_dict(0, 'out'),
            batch_size=10))

  def test_dask_data_feeder(self):
    if HAS_PANDAS and HAS_DASK:
      x = pd.DataFrame(
          dict(
              a=np.array([.1, .3, .4, .6, .2, .1, .6]),
              b=np.array([.7, .8, .1, .2, .5, .3, .9])))
      x = dd.from_pandas(x, npartitions=2)
      y = pd.DataFrame(dict(labels=np.array([1, 0, 2, 1, 0, 1, 2])))
      y = dd.from_pandas(y, npartitions=2)
      # TODO(ipolosukhin): Remove or restore this.
      # x = extract_dask_data(x)
      # y = extract_dask_labels(y)
      df = data_feeder.DaskDataFeeder(x, y, n_classes=2, batch_size=2)
      inp, out = df.input_builder()
      feed_dict_fn = df.get_feed_dict_fn()
      feed_dict = feed_dict_fn()
      self.assertAllClose(feed_dict[inp.name], [[0.40000001, 0.1],
                                                [0.60000002, 0.2]])
      self.assertAllClose(feed_dict[out.name], [[0., 0., 1.], [0., 1., 0.]])

  # TODO(rohanj): Fix this test by fixing data_feeder. Currently, h5py doesn't
  # support permutation based indexing lookups (More documentation at
  # http://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing)
  def DISABLED_test_hdf5_data_feeder(self):

    def func(df):
      inp, out = df.input_builder()
      feed_dict_fn = df.get_feed_dict_fn()
      feed_dict = feed_dict_fn()
      self._assertAllClose(inp, [[3, 4], [1, 2]], feed_dict, 'name')
      self.assertAllClose(out, [2, 1], feed_dict, 'name')

    try:
      import h5py  # pylint: disable=g-import-not-at-top
      x = np.matrix([[1, 2], [3, 4]])
      y = np.array([1, 2])
      file_path = os.path.join(self._base_dir, 'test_hdf5.h5')
      h5f = h5py.File(file_path, 'w')
      h5f.create_dataset('x', data=x)
      h5f.create_dataset('y', data=y)
      h5f.close()
      h5f = h5py.File(file_path, 'r')
      x = h5f['x']
      y = h5f['y']
      func(data_feeder.DataFeeder(x, y, n_classes=0, batch_size=3))
      func(
          data_feeder.DataFeeder(
              self._wrap_dict(x, 'in'),
              self._wrap_dict(y, 'out'),
              n_classes=self._wrap_dict(0, 'out'),
              batch_size=3))
    except ImportError:
      print("Skipped test for hdf5 since it's not installed.")


class SetupPredictDataFeederTest(DataFeederTest):
  """Tests for `DataFeeder.setup_predict_data_feeder`."""

  def test_iterable_data(self):
    # pylint: disable=undefined-variable

    def func(df):
      self._assertAllClose(six.next(df), [[1, 2], [3, 4]])
      self._assertAllClose(six.next(df), [[5, 6]])

    data = [[1, 2], [3, 4], [5, 6]]
    x = iter(data)
    x_dict = iter([self._wrap_dict(v) for v in iter(data)])
    func(data_feeder.setup_predict_data_feeder(x, batch_size=2))
    func(data_feeder.setup_predict_data_feeder(x_dict, batch_size=2))


if __name__ == '__main__':
  test.main()
