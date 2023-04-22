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
"""Test related utilities for KPL + tf.distribute."""

import random
import tempfile

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class DistributeKplTestUtils(test.TestCase):
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

  def define_reverse_lookup_layer(self):
    """Create string reverse lookup layer for serving."""

    label_inverse_lookup_layer = string_lookup.StringLookup(
        num_oov_indices=0,
        mask_token=None,
        vocabulary=self.LABEL_VOCAB,
        invert=True)
    return label_inverse_lookup_layer

  def create_serving_signature(self, model, feature_mapper,
                               label_inverse_lookup_layer):
    """Create serving signature for the given model."""

    @def_function.function
    def serve_fn(raw_features):
      raw_features = array_ops.expand_dims(raw_features, axis=0)
      transformed_features = model.feature_mapper(raw_features)
      outputs = model(transformed_features)
      outputs = array_ops.squeeze(outputs, axis=0)
      outputs = math_ops.cast(math_ops.greater(outputs, 0.5), dtypes.int64)
      decoded_outputs = model.label_inverse_lookup_layer(outputs)
      return array_ops.squeeze(decoded_outputs, axis=0)

    model.feature_mapper = feature_mapper
    model.label_inverse_lookup_layer = label_inverse_lookup_layer
    # serving does NOT have batch dimension
    return serve_fn.get_concrete_function(
        tensor_spec.TensorSpec(
            shape=(3), dtype=dtypes.string, name="example"))

  def test_save_load_serving_model(self, model, feature_mapper,
                                   label_inverse_lookup_layer):
    """Test save/load/serving model."""

    serving_fn = self.create_serving_signature(model, feature_mapper,
                                               label_inverse_lookup_layer)

    saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    model.save(saved_model_dir, save_format="tf",
               signatures={"serving_default": serving_fn})

    # Test the saved_model.
    loaded_serving_fn = keras.saving.save.load_model(
        saved_model_dir).signatures["serving_default"]

    # check the result w/ and w/o avenger.
    prediction0 = loaded_serving_fn(
        constant_op.constant(["avenger", "ironman", "avenger"]))["output_0"]
    self.assertIn(prediction0.numpy().decode("UTF-8"), ("yes", "no"))

    prediction1 = loaded_serving_fn(
        constant_op.constant(["ironman", "ironman", "unkonwn"]))["output_0"]
    self.assertIn(prediction1.numpy().decode("UTF-8"), ("yes", "no"))
