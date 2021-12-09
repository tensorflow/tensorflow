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
# LINT.IfChange
"""Utils for managing different mode strings used by Keras and Estimator models.
"""

import collections


class KerasModeKeys:
    """Standard names for model modes.

    The following standard keys are defined:

    * `TRAIN`: training/fitting mode.
    * `TEST`: testing/evaluation mode.
    * `PREDICT`: prediction/inference mode.
    """

    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"


# TODO(kathywu): Remove copy in Estimator after nightlies
class EstimatorModeKeys:
    """Standard names for Estimator model modes.

    The following standard keys are defined:

    * `TRAIN`: training/fitting mode.
    * `EVAL`: testing/evaluation mode.
    * `PREDICT`: predication/inference mode.
    """

    TRAIN = "train"
    EVAL = "eval"
    PREDICT = "infer"


def is_predict(mode):
    return mode in [KerasModeKeys.PREDICT, EstimatorModeKeys.PREDICT]


def is_eval(mode):
    return mode in [KerasModeKeys.TEST, EstimatorModeKeys.EVAL]


def is_train(mode):
    return mode in [KerasModeKeys.TRAIN, EstimatorModeKeys.TRAIN]


class ModeKeyMap(collections.abc.Mapping):
    """Map using ModeKeys as keys.

    This class creates an immutable mapping from modes to values. For example,
    SavedModel export of Keras and Estimator models use this to map modes to their
    corresponding MetaGraph tags/SignatureDef keys.

    Since this class uses modes, rather than strings, as keys, both "predict"
    (Keras's PREDICT ModeKey) and "infer" (Estimator's PREDICT ModeKey) map to the
    same value.
    """

    def __init__(self, **kwargs):
        self._internal_dict = {}
        self._keys = []
        for key in kwargs:
            self._keys.append(key)
            dict_key = self._get_internal_key(key)
            if dict_key in self._internal_dict:
                raise ValueError(
                    "Error creating ModeKeyMap. Multiple keys/values found for {} mode.".format(
                        dict_key
                    )
                )
            self._internal_dict[dict_key] = kwargs[key]

    def _get_internal_key(self, key):
        """Return keys used for the internal dictionary."""
        if is_train(key):
            return KerasModeKeys.TRAIN
        if is_eval(key):
            return KerasModeKeys.TEST
        if is_predict(key):
            return KerasModeKeys.PREDICT
        raise ValueError("Invalid mode key: {}.".format(key))

    def __getitem__(self, key):
        return self._internal_dict[self._get_internal_key(key)]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


# LINT.ThenChange(//tensorflow/python/saved_model/model_utils/mode_keys.py)
