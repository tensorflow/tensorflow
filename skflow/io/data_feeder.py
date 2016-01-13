"""Implementations of different data feeders to provide data for TF trainer."""

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

from __future__ import division, print_function, absolute_import

import itertools
import six
from six.moves import xrange   # pylint: disable=redefined-builtin

import numpy as np
from sklearn.utils import check_array

from skflow.io import HAS_PANDAS, extract_pandas_data, extract_pandas_matrix, extract_pandas_labels
from skflow.io import HAS_DASK, extract_dask_data, extract_dask_labels


def _get_in_out_shape(x_shape, y_shape, n_classes, batch_size):
    """Returns shape for input and output of the data feeder."""
    x_shape = list(x_shape[1:]) if len(x_shape) > 1 else [1]
    input_shape = [batch_size] + x_shape
    y_shape = list(y_shape[1:]) if len(y_shape) > 1 else []
    # Skip first dimention if it is 1.
    if y_shape and y_shape[0] == 1:
        y_shape = y_shape[1:]
    if n_classes > 1:
        output_shape = [batch_size] + y_shape + [n_classes]
    else:
        output_shape = [batch_size] + y_shape
    return input_shape, output_shape


def _data_type_filter(X, y):
    """Filter data types into acceptable format"""
    if HAS_PANDAS:
        X = extract_pandas_data(X)
        y = extract_pandas_labels(y)
    if HAS_DASK:
        X = extract_dask_data(X)
        y = extract_dask_labels(y)
    return X, y


def _is_iterable(X):
    return hasattr(X, 'next') or hasattr(X, '__next__')


def setup_train_data_feeder(X, y, n_classes, batch_size):
    """Create data feeder, to sample inputs from dataset.
    If X and y are iterators, use StreamingDataFeeder.

    Args:
        X: numpy, pandas or Dask matrix or iterable.
        y: numpy, pandas or Dask array or iterable.
        n_classes: number of classes.
        batch_size: size to split data into parts.

    Returns:
        DataFeeder object that returns training data.
    """
    X, y = _data_type_filter(X, y)
    if HAS_DASK:
        import dask.dataframe as dd
        if isinstance(X, dd.Series) and isinstance(y, dd.Series):
            data_feeder_cls = DaskDataFeeder
        else:
            data_feeder_cls = DataFeeder
    else:
        data_feeder_cls = DataFeeder

    if _is_iterable(X):
        if not _is_iterable(y):
            raise ValueError("Both X and y should be iterators for "
                             "streaming learning to work.")
        data_feeder_cls = StreamingDataFeeder
    return data_feeder_cls(X, y, n_classes, batch_size)


def _batch_data(X, batch_size):
    chunk = []
    for data in X:
        chunk.append(data)
        if batch_size > 0 and len(chunk) > batch_size:
            yield np.matrix(chunk)
    yield np.matrix(chunk)


def setup_predict_data_feeder(X, batch_size=-1):
    """Returns an iterable for feeding into predict step.

    Args:
        X: numpy, pandas, Dask array or iterable.
        batch_size: Size of batches to split data into.
            If negative, returns one batch of full size.

    Returns:
        List or iterator of parts of data to predict on.
    """
    if HAS_PANDAS:
        X = extract_pandas_data(X)
    if HAS_DASK:
        X = extract_dask_data(X)
    if _is_iterable(X):
        return _batch_data(X, batch_size)
    if len(X.shape) == 1:
        X = np.reshape(X, (-1, 1))
    return [X]


def setup_processor_data_feeder(X):
    """Sets up processor iterable.

    Args:
        X: numpy, pandas or iterable.

    Returns:
        Iterable of data to process.
    """
    if HAS_PANDAS:
        X = extract_pandas_matrix(X)
    return X


class DataFeeder(object):
    """Data feeder is an example class to sample data for TF trainer.

    Parameters:
        X: feature Nd numpy matrix of shape [n_samples, n_features, ...].
        y: target vector, either floats for regression or class id for
            classification. If matrix, will consider as a sequence
            of targets.
        n_classes: number of classes, 0 and 1 are considered regression.
        batch_size: mini batch size to accumulate.
        random_state: numpy RandomState object to reproduce sampling.

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

    def __init__(self, X, y, n_classes, batch_size, random_state=None):
        x_dtype = np.int64 if X.dtype == np.int64 else np.float32
        self.X = check_array(X, ensure_2d=False,
                             allow_nd=True, dtype=x_dtype)
        self.y = check_array(y, ensure_2d=False, dtype=np.float32)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.input_shape, self.output_shape = _get_in_out_shape(
            self.X.shape, self.y.shape, n_classes, batch_size)
        self.input_dtype, self.output_dtype = self.X.dtype, self.y.dtype
        self.random_state = np.random.RandomState(42) if random_state is None else random_state
        self.indices = self.random_state.permutation(self.X.shape[0])
        self.offset = 0
        self.epoch = 0

    def get_feed_params_fn(self):
        """Returns a function, that will return a dict with data feed params while training.
        Returns:
            A function, that will return a dict with data feed params while training.
        """
        def _feed_params_fn():
            return {
                'epoch': self.epoch,
                'offset': self.offset
            }
        return _feed_params_fn

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
            # take random indices
            batch_indices = self.indices[self.offset: self.offset+self.batch_size]

            # assign input features from random indices
            inp = [self.X[batch_indices]] if len(self.X.shape) == 1 else self.X[batch_indices]

            # assign labels from random indices
            self.output_shape[0] = batch_indices.shape[0]
            out = np.zeros(self.output_shape, dtype=self.output_dtype)
            for i in xrange(out.shape[0]):
                sample = batch_indices[i]
                if self.n_classes > 1:
                    if len(self.output_shape) == 2:
                        out.itemset((i, self.y[sample]), 1.0)
                    else:
                        for idx, value in enumerate(self.y[sample]):
                            out.itemset(tuple([i, idx, value]), 1.0)
                else:
                    out[i] = self.y[sample]

            # move offset and reset it if necessary
            self.offset += self.batch_size
            if self.offset >= self.X.shape[0]:
                self.indices = self.random_state.permutation(self.X.shape[0])
                self.offset = 0
                self.epoch += 1

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
        X_first_el = six.next(X)
        y_first_el = six.next(y)
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

    def get_feed_params_fn(self):
        """Returns a function, that will return a dict with data feed params while training.
        Returns:
            A function, that will return a dict with data feed params while training.
        """
        def _feed_params_fn():
            return {}
        return _feed_params_fn

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
                inp[i, :] = six.next(self.X)
                y = six.next(self.y)
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


class DaskDataFeeder(object):
    """Data feeder for TF trainer that reads data from dask.Series.

    Numpy arrays can be serialized to disk and it's possible to do random seeks into them.
    DaskDataFeeder will remove requirement to have full dataset in the memory and still do
    random seeks for sampling of batches.

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
    def __init__(self, X, y, n_classes, batch_size, random_state=None):
        import dask.dataframe as dd
        # TODO: check X and y dtypes in dask_io like pandas
        self.X = X
        self.y = y
        # save column names
        self.X_columns = list(X.columns)
        self.y_columns = list(y.columns)
        # combine into a data frame
        self.df = dd.multi.concat([X, y], axis=1)

        self.n_classes = n_classes
        X_shape = tuple([X.count().compute()])
        y_shape = tuple([y.count().compute()])
        self.sample_fraction = batch_size/float(list(X_shape)[0])
        self.input_shape, self.output_shape = _get_in_out_shape(
            X_shape, y_shape, n_classes, batch_size)
        self.input_dtype, self.output_dtype = self.X.dtype, self.y.dtype
        if random_state is None:
            self.random_state = np.random.RandomState(42)
        else:
            self.random_state = random_state

    def get_feed_params_fn(self):
        """Returns a function, that will return a dict with data feed params while training.
        Returns:
            A function, that will return a dict with data feed params while training.
        """
        def _feed_params_fn():
            return {}
        return _feed_params_fn

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
            # TODO: option for with/without replacement (dev version of dask)
            sample = self.df.random(self.sample_fraction,
                                    random_state=self.random_state)
            inp = sample[self.X_columns]
            out = sample[self.y_columns]
            return {input_placeholder.name: inp, output_placeholder.name: out}
        return _feed_dict_fn
