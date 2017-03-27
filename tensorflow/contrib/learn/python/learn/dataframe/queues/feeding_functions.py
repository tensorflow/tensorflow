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
"""Helper functions for enqueuing data from arrays and pandas `DataFrame`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# pylint: disable=unused-import
from tensorflow.python.estimator.inputs.queues.feeding_functions import _ArrayFeedFn
from tensorflow.python.estimator.inputs.queues.feeding_functions import _enqueue_data as enqueue_data
from tensorflow.python.estimator.inputs.queues.feeding_functions import _OrderedDictNumpyFeedFn
from tensorflow.python.estimator.inputs.queues.feeding_functions import _PandasFeedFn
from tensorflow.python.estimator.inputs.queues.feeding_functions import errors
# pylint: enable=unused-import


class _GeneratorFeedFn(object):
    """Creates feed dictionaries from `Generator` of `dicts` of numpy arrays."""

    def __init__(self,
                 placeholders,
                 generator,
                 batch_size,
                 random_start=False,
                 seed=None,
                 num_epochs=None):
        first_sample = next(generator())
        if len(placeholders) != len(first_sample):
            raise ValueError("Expected {} placeholders; got {}.".format(
                len(first_sample), len(placeholders)))
        self._col_placeholders = placeholders
        self._generator_function = generator
        self._iterator = generator()
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._epoch = 0
        random.seed(seed)

    def __call__(self):
        if self._num_epochs and self._epoch >= self._num_epochs:
            raise errors.OutOfRangeError(None, None,
                                         "Already emitted %s epochs." % self._epoch)
        list_dict = {}
        list_dict_size = 0
        while list_dict_size < self._batch_size:
            try:
                data_row = next(self._iterator)
            except StopIteration:
                self._epoch += 1
                self._iterator = self._generator_function()
                data_row = next(self._iterator)
            for index, key in enumerate(sorted(data_row.keys())):
                list_dict.setdefault(
                    self._col_placeholders[index], list()).append(data_row[key])
            list_dict_size += 1
        feed_dict = {key: np.asarray(item) for key, item
                         in list(list_dict.items())}
        return feed_dict
