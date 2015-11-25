"""Implementations of different data feeders to provide data for TF trainer."""

#  Copyright 2015 Google Inc. All Rights Reserved.
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

import itertools
import random
import numpy as np

from sklearn.utils import check_array


def _get_in_out_shape(x_shape, y_shape, n_classes, batch_size):
    """Returns shape for input and output of the data feeder."""
    input_shape = [batch_size] + list(x_shape[1:])
    y_shape = list(y_shape[1:]) if len(y_shape) > 1 else []
    # Skip first dimention if it is 1.
    if y_shape and y_shape[0] == 1:
        y_shape = y_shape[1:]
    if n_classes > 1:
        output_shape = [batch_size] + y_shape + [n_classes]
    else:
        output_shape = [batch_size] + y_shape
    return input_shape, output_shape


class DataFeeder(object):
    """Data feeder is an example class to sample data for TF trainer.

    Parameters:
        X: feature Nd numpy matrix of shape [n_samples, n_features, ...].
        y: target vector, either floats for regression or class id for
            classification. If matrix, will consider as a sequence
            of targets.
        n_classes: number of classes, 0 and 1 are considered regression.
        batch_size: mini batch size to accumulate.

    Attributes:
        X: input features.
        y: input target.
        n_classes: number of classes.
        batch_size: mini batch size to accumulate.
        input_shape: shape of the input.
        output_shape: shape of the output.
        input_dtype: dtype of input.
        output_dtype: dtype of output.
    """

    def __init__(self, X, y, n_classes, batch_size):
        self.X = check_array(X, ensure_2d=False,
                             allow_nd=True, dtype=[np.float32, np.int64])
        self.y = check_array(y, ensure_2d=False, dtype=np.float32)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.input_shape, self.output_shape = _get_in_out_shape(
            self.X.shape, self.y.shape, n_classes, batch_size)
        self.input_dtype, self.output_dtype = self.X.dtype, self.y.dtype

    def get_feed_dict_fn(self, input_placeholder, output_placeholder):
        """Returns a function, that will sample data and provide it to given
        placeholders.

        Args:
            input_placeholder: tf.Placeholder for input features mini batch.
            output_placeholder: tf.Placeholder for output targets.
        Returns:
            A function that when called samples a random subset of batch size
            from X and y.
        """
        def _feed_dict_fn():
            inp = np.zeros(self.input_shape, dtype=self.input_dtype)
            out = np.zeros(self.output_shape, dtype=self.output_dtype)
            for i in xrange(self.batch_size):
                sample = random.randint(0, self.X.shape[0] - 1)
                inp[i, :] = self.X[sample, :]
                if self.n_classes > 1:
                    if len(self.output_shape) == 2:
                        out.itemset((i, self.y[sample]), 1.0)
                    else:
                        for idx, value in enumerate(self.y[sample]):
                            out.itemset(tuple([i, idx, value]), 1.0)
                else:
                    out[i] = self.y[sample]
            return {input_placeholder.name: inp, output_placeholder.name: out}
        return _feed_dict_fn


class StreamingDataFeeder(object):
    """Data feeder for TF trainer that reads data from iterator.

    Streaming data feeder allows to read data as it comes it from disk or
    somewhere else. It's custom to have this iterators rotate infinetly over
    the dataset, to allow control of how much to learn on the trainer side.

    Parameters:
        X: iterator that returns for each element, returns features.
        y: iterator that returns for each element, returns 1 or many classes /
           regression values.
        n_classes: indicator of how many classes the target has.
        batch_size: Mini batch size to accumulate.
 
    Attributes:
        X: input features.
        y: input target.
        n_classes: number of classes.
        batch_size: mini batch size to accumulate.
        input_shape: shape of the input.
        output_shape: shape of the output.
        input_dtype: dtype of input.
        output_dtype: dtype of output.
    """

    def __init__(self, X, y, n_classes, batch_size):
        X_first_el = X.next()
        y_first_el = y.next()
        self.X = itertools.chain([X_first_el], X)
        self.y = itertools.chain([y_first_el], y)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.input_shape, self.output_shape = _get_in_out_shape(
            [1] + list(X_first_el.shape),
            [1] + list(y_first_el.shape), n_classes, batch_size)
        self.input_dtype = X_first_el.dtype
        # Convert float64 to float32, as all the parameters in the model are
        # floats32 and there is a lot of benefits in using it in NNs.
        if self.input_dtype == np.float64:
            self.input_dtype = np.float32
        # Output types are floats, due to both softmaxes and regression req.
        self.output_dtype = np.float32

    def get_feed_dict_fn(self, input_placeholder, output_placeholder):
        """Returns a function, that will sample data and provide it to given

        placeholders.

        Args:
            input_placeholder: tf.Placeholder for input features mini batch.
            output_placeholder: tf.Placeholder for output targets.
        Returns:
            A function that when called samples a random subset of batch size
            from X and y.
        """
        def _feed_dict_fn():
            inp = np.zeros(self.input_shape, dtype=self.input_dtype)
            out = np.zeros(self.output_shape, dtype=self.output_dtype)
            for i in xrange(self.batch_size):
                inp[i, :] = self.X.next()
                y = self.y.next()
                if self.n_classes > 1:
                    if len(self.output_shape) == 2:
                        out.itemset((i, y), 1.0)
                    else:
                        for idx, value in enumerate(y):
                            out.itemset(tuple([i, idx, value]), 1.0)
                else:
                    out[i] = y
            return {input_placeholder.name: inp, output_placeholder.name: out}
        return _feed_dict_fn

