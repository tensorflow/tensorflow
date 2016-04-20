#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from struct import Struct
import numpy as np
import six

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.io import *


class DataFeederTest(tf.test.TestCase):

    def test_unsupervised(self):
        data = np.matrix([[1, 2], [2, 3], [3, 4]])
        feeder = data_feeder.DataFeeder(data, None, n_classes=0, batch_size=1)
        with self.test_session():
            inp, out = feeder.input_builder()
            feed_dict_fn = feeder.get_feed_dict_fn()
            feed_dict = feed_dict_fn()
            self.assertAllClose(feed_dict[inp.name], [[1, 2]])

    def test_data_feeder_regression(self):
        X = np.matrix([[1, 2], [3, 4]])
        y = np.array([1, 2])
        df = data_feeder.DataFeeder(X, y, n_classes=0, batch_size=3)
        inp, out = df.input_builder()
        feed_dict_fn = df.get_feed_dict_fn()
        feed_dict = feed_dict_fn()

        self.assertAllClose(feed_dict[inp.name], [[3, 4], [1, 2]])
        self.assertAllClose(feed_dict[out.name], [2, 1])

    def test_data_feeder_multioutput_regression(self):
        X = np.matrix([[1, 2], [3, 4]])
        y = np.array([[1, 2], [3, 4]])
        df = data_feeder.DataFeeder(X, y, n_classes=0, batch_size=2)
        inp, out = df.input_builder()
        feed_dict_fn = df.get_feed_dict_fn()
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict[inp.name], [[3, 4], [1, 2]])
        self.assertAllClose(feed_dict[out.name], [[3, 4], [1, 2]])

    def test_data_feeder_multioutput_classification(self):
        X = np.matrix([[1, 2], [3, 4]])
        y = np.array([[0, 1, 2], [2, 3, 4]])
        df = data_feeder.DataFeeder(X, y, n_classes=5, batch_size=2)
        inp, out = df.input_builder()
        feed_dict_fn = df.get_feed_dict_fn()
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict[inp.name], [[3, 4], [1, 2]])
        self.assertAllClose(feed_dict[out.name], [[[0, 0, 1, 0, 0],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 1]],
                                                  [[1, 0, 0, 0, 0],
                                                   [0, 1, 0, 0, 0],
                                                   [0, 0, 1, 0, 0]]])

    def test_streaming_data_feeder(self):
        def X_iter():
            yield np.array([1, 2])
            yield np.array([3, 4])

        def y_iter():
            yield np.array([1])
            yield np.array([2])
        df = data_feeder.StreamingDataFeeder(X_iter(), y_iter(), n_classes=0,
                                             batch_size=2)
        inp, out = df.input_builder()
        feed_dict_fn = df.get_feed_dict_fn()
        feed_dict = feed_dict_fn()
        self.assertAllClose(feed_dict[inp.name], [[1, 2], [3, 4]])
        self.assertAllClose(feed_dict[out.name], [1, 2])

    def test_dask_data_feeder(self):
        if HAS_PANDAS and HAS_DASK:
            X = pd.DataFrame(dict(a=np.array([.1, .3, .4, .6, .2, .1, .6]),
                                  b=np.array([.7, .8, .1, .2, .5, .3, .9])))
            X = dd.from_pandas(X, npartitions=2)
            y = pd.DataFrame(dict(labels=np.array([1, 0, 2, 1, 0, 1, 2])))
            y = dd.from_pandas(y, npartitions=2)
            # X = extract_dask_data(X)
            # y = extract_dask_labels(y)
            df = data_feeder.DaskDataFeeder(X, y, n_classes=2, batch_size=2)
            inp, out = df.input_builder()
            feed_dict_fn = df.get_feed_dict_fn()
            feed_dict = feed_dict_fn()
            self.assertAllClose(feed_dict[inp.name], [[ 0.40000001, 0.1],
                                                     [ 0.60000002, 0.2]])
            self.assertAllClose(feed_dict[out.name], [[ 0., 0., 1.],
                                                     [ 0., 1., 0.]])


class SetupPredictDataFeederTest(tf.test.TestCase):

    def test_iterable_data(self):
        X = iter([[1, 2], [3, 4], [5, 6]])
        df = data_feeder.setup_predict_data_feeder(X, batch_size=2)
        self.assertAllClose(six.next(df), [[1, 2], [3, 4]])
        self.assertAllClose(six.next(df), [[5, 6]])


if __name__ == '__main__':
    tf.test.main()
