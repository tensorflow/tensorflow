# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A base class to provide a model and corresponding input data for testing."""


class ModelAndInput:
    """Base class to provide model and its corresponding inputs."""

    def get_model(self):
        """Returns a compiled keras model object, together with output name.

        Returns:
          model: a keras model object
          output_name: a string for the name of the output layer
        """
        raise NotImplementedError("must be implemented in descendants")

    def get_data(self):
        """Returns data for training and predicting.

        Returns:
          x_train: data used for training
          y_train: label used for training
          x_predict: data used for predicting
        """
        raise NotImplementedError("must be implemented in descendants")

    def get_batch_size(self):
        """Returns the batch_size used by the model."""
        raise NotImplementedError("must be implemented in descendants")
