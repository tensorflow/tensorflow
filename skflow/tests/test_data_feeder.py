#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from struct import Struct
import numpy as np

import tensorflow as tf

from skflow import data_feeder
from skflow.io import *


class MockPlaceholder(object):

    def __init__(self, name):
        self.name = name


class DataFeederTest(tf.test.TestCase):

    def test_data_feeder_regression(self):
        X = np.matrix([[1, 2], [3, 4]])
        y = np.array([1, 2])
        df = data_feeder.DataFeeder(X, y, n_classes=0, batch_size=3)
        feed_dict_fn = df.get_feed_dict_fn(
            MockPlaceholder(name='input'),
            MockPlaceholder(name='output'))
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict['input'], [[1, 2], [3, 4], [1, 2]])
        self.assertAllClose(feed_dict['output'], [1, 2, 1])

    def test_data_feeder_multioutput_regression(self):
        X = np.matrix([[1, 2], [3, 4]])
        y = np.array([[1, 2], [3, 4]])
        df = data_feeder.DataFeeder(X, y, n_classes=0, batch_size=2)
        feed_dict_fn = df.get_feed_dict_fn(
            MockPlaceholder(name='input'),
            MockPlaceholder(name='output'))
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict['input'], [[1, 2], [3, 4]])
        self.assertAllClose(feed_dict['output'], [[1, 2], [3, 4]])

    def test_data_feeder_multioutput_classification(self):
        X = np.matrix([[1, 2], [3, 4]])
        y = np.array([[0, 1, 2], [2, 3, 4]])
        df = data_feeder.DataFeeder(X, y, n_classes=5, batch_size=2)
        feed_dict_fn = df.get_feed_dict_fn(
            MockPlaceholder(name='input'),
            MockPlaceholder(name='output'))
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict['input'], [[1, 2], [3, 4]])
        self.assertAllClose(feed_dict['output'], [[[1, 0, 0, 0, 0],
                                                   [0, 1, 0, 0, 0],
                                                   [0, 0, 1, 0, 0]],
                                                  [[0, 0, 1, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 1]]])

    def test_streaming_data_feeder(self):
        def X_iter():
            yield np.array([1, 2])
            yield np.array([3, 4])

        def y_iter():
            yield np.array([1])
            yield np.array([2])
        df = data_feeder.StreamingDataFeeder(X_iter(), y_iter(), n_classes=0,
                                             batch_size=2)
        feed_dict_fn = df.get_feed_dict_fn(
            MockPlaceholder(name='input'),
            MockPlaceholder(name='output'))
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict['input'], [[1, 2], [3, 4]])
        self.assertAllClose(feed_dict['output'], [1, 2])

    def test_dask_data_feeder(self):
        import pandas as pd
        import dask.dataframe as dd
        if HAS_PANDAS and HAS_DASK:
            X = pd.DataFrame(dict(a=list('aabbcc')))
            X = dd.from_pandas(X, npartitions=2)
            y = pd.DataFrame(dict(labels=list('010011')))
            y = dd.from_pandas(y, npartitions=2)
            X = extract_dask_data(X)
            y = extract_dask_labels(y)

            df = data_feeder.DaskDataFeeder(X, y, n_classes=2, batch_size=2)
            feed_dict_fn = df.get_feed_dict_fn(
                MockPlaceholder(name='input'),
                MockPlaceholder(name='output'))
            feed_dict = feed_dict_fn()
            self.assertListEqual(feed_dict['input'], [['a'], ['b']])
            self.assertListEqual(feed_dict['output'], [['0'], ['0']])

if __name__ == '__main__':
    tf.test.main()
