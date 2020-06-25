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
"""Tests for keras.layers.preprocessing.normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.platform import test


@combinations.generate(
    combinations.combine(
        distribution=strategy_combinations.all_strategies,
        mode=["eager", "graph"]))
class ImagePreprocessingDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution(self, distribution):
    # TODO(b/159738418): large image input causes OOM in ubuntu multi gpu.
    np_images = np.random.random((32, 32, 32, 3)).astype(np.float32)
    image_dataset = dataset_ops.Dataset.from_tensor_slices(np_images).batch(
        16, drop_remainder=True)

    with distribution.scope():
      input_data = keras.Input(shape=(32, 32, 3), dtype=dtypes.float32)
      image_preprocessor = keras.Sequential([
          image_preprocessing.Resizing(height=256, width=256),
          image_preprocessing.RandomCrop(height=224, width=224),
          image_preprocessing.RandomTranslation(.1, .1),
          image_preprocessing.RandomRotation(.2),
          image_preprocessing.RandomFlip(),
          image_preprocessing.RandomZoom(.2, .2)])
      preprocessed_image = image_preprocessor(input_data)
      flatten_layer = keras.layers.Flatten(data_format="channels_last")
      output = flatten_layer(preprocessed_image)
      cls_layer = keras.layers.Dense(units=1, activation="sigmoid")
      output = cls_layer(output)
      model = keras.Model(inputs=input_data, outputs=output)
    model.compile(loss="binary_crossentropy")
    _ = model.predict(image_dataset)


if __name__ == "__main__":
  test.main()
