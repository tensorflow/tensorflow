# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.ops import math_ops


class DistributeKplTestUtils:
  """Utils for test of tf.distribute + KPL."""
  FEATURE_VOCAB = [
      "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong",
      "wonder_woman"
  ]
  LABEL_VOCAB = ["yes", "no"]

  def define_kpls_for_training(self, use_adapt):
    """Function that defines KPL used for unit tests of tf.distribute.

    Args:
      use_adapt: if adapt will be called. False means there will be precomputed
        statistics.

    Returns:
      feature_mapper: a simple keras model with one keras StringLookup layer
      which maps feature to index.
      label_mapper: similar to feature_mapper, but maps label to index.

    """
    if use_adapt:
      feature_lookup_layer = (
          string_lookup.StringLookup(
              num_oov_indices=1))
      feature_lookup_layer.adapt(self.FEATURE_VOCAB)
      label_lookup_layer = (
          string_lookup.StringLookup(
              num_oov_indices=0, mask_token=None))
      label_lookup_layer.adapt(self.LABEL_VOCAB)
    else:
      feature_lookup_layer = (
          string_lookup.StringLookup(
              vocabulary=self.FEATURE_VOCAB, num_oov_indices=1))
      label_lookup_layer = (
          string_lookup.StringLookup(
              vocabulary=self.LABEL_VOCAB, num_oov_indices=0, mask_token=None))

    raw_feature_input = keras.layers.Input(
        shape=(3,), dtype=dtypes.string, name="feature", ragged=True)
    feature_id_input = feature_lookup_layer(raw_feature_input)
    feature_mapper = keras.Model({"features": raw_feature_input},
                                 feature_id_input)

    raw_label_input = keras.layers.Input(
        shape=(1,), dtype=dtypes.string, name="label")
    label_id_input = label_lookup_layer(raw_label_input)
    label_mapper = keras.Model({"label": raw_label_input}, label_id_input)

    return feature_mapper, label_mapper

  def dataset_fn(self, feature_mapper, label_mapper):
    """Function that generates dataset for test of tf.distribute + KPL.

    Args:
      feature_mapper: a simple keras model with one keras StringLookup layer
        which maps feature to index.
      label_mapper: similar to feature_mapper, but maps label to index.

    Returns:
      Generated dataset for test of tf.distribute + KPL.

    """

    def feature_and_label_gen():
      # Generator of dataset.
      while True:
        features = random.sample(self.FEATURE_VOCAB, 3)
        label = ["yes"] if self.FEATURE_VOCAB[0] in features else ["no"]
        yield {"features": features, "label": label}

    raw_dataset = dataset_ops.Dataset.from_generator(
        feature_and_label_gen,
        output_signature={
            "features": tensor_spec.TensorSpec([3], dtypes.string),
            "label": tensor_spec.TensorSpec([1], dtypes.string)
        }).shuffle(100).batch(32)

    train_dataset = raw_dataset.map(lambda x: (  # pylint: disable=g-long-lambda
        {
            "features": feature_mapper(x["features"])
        }, label_mapper(x["label"])))
    return train_dataset

  def define_model(self):
    """A simple model for test of tf.distribute + KPL."""
    # Create the model. The input needs to be compatible with KPLs.
    model_input = keras.layers.Input(
        shape=(3,), dtype=dtypes.int64, name="model_input")

    # input_dim includes a mask token and an oov token.
    emb_output = keras.layers.Embedding(
        input_dim=len(self.FEATURE_VOCAB) + 2, output_dim=20)(
            model_input)
    emb_output = math_ops.reduce_mean(emb_output, axis=1)
    dense_output = keras.layers.Dense(
        units=1, activation="sigmoid")(
            emb_output)
    model = keras.Model({"features": model_input}, dense_output)
    return model
