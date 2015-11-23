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

import random
import numpy as np

from sklearn.utils import check_array


class DataFeeder(object):
    """Data feeder is an example class to sample data for TF trainer.

    Parameters:
        X: feature Nd numpy matrix of shape [n_samples, n_features, ...].
        y: target vector, either floats for regression or class id for
            classification.
        n_classes: number of classes, 0 and 1 are considered regression.
        batch_size: mini batch size to accumulate.
    """

    def __init__(self, X, y, n_classes, batch_size):
        self.X = check_array(X, dtype=np.float32, ensure_2d=False,
                             allow_nd=True)
        self.y = check_array(y, ensure_2d=False, dtype=None)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self._input_shape = [batch_size] + list(X.shape[1:])
        self._output_shape = [batch_size, n_classes] if n_classes > 1 else [batch_size]

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
            inp = np.zeros(self._input_shape)
            out = np.zeros(self._output_shape)
            for i in xrange(self.batch_size):
                sample = random.randint(0, self.X.shape[0] - 1)
                inp[i, :] = self.X[sample, :]
                if self.n_classes > 1:
                    out[i, self.y[sample]] = 1.0
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
    """

    def __init__(self, X, y, n_classes, batch_size):
        pass

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
            pass
        return _feed_dict_fn

