# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Base Class for TPU Embeddings Mid level APIs."""

import functools
from typing import Any, Dict, Iterable, Optional, Union, Text

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import nest


class TPUEmbeddingBase(tracking.AutoTrackable):
  """The TPUEmbedding Base class.

  This class only contains the basic logic to check the feature config and table
  config for the tpu embedding mid level APIs.
  """

  def __init__(
      self,
      feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer] = None):  # pylint:disable=protected-access
    """Creates the TPUEmbeddingBase object."""
    self._feature_config = feature_config
    self._output_shapes = []
    for feature in nest.flatten(feature_config):
      self._output_shapes.append(feature.output_shape)
    # Set table order here to the order of the first occurrence of the table in
    # a feature provided by the user. The order of this struct must be fixed
    # to provide the user with deterministic behavior over multiple
    # instantiations.
    self._table_config = []
    for feature in nest.flatten(feature_config):
      if feature.table not in self._table_config:
        self._table_config.append(feature.table)

    # Ensure tables have unique names. Also error check the optimizer as we
    # specifically don't do that in the TableConfig class to allow high level
    # APIs that are built on this to use strings/other classes to represent
    # optimizers (before they are passed to this class).
    table_names = []
    for i, table in enumerate(self._table_config):
      if table.optimizer is None:
        # TODO(bfontain) Should we allow some sort of optimizer merging here?
        table.optimizer = optimizer
      if (table.optimizer is not None and
          not isinstance(table.optimizer, tpu_embedding_v2_utils._Optimizer)):  # pylint: disable=protected-access
        raise ValueError("{} is an unsupported optimizer class. Please pass an "
                         "instance of one of the optimizer classes under "
                         "tf.tpu.experimental.embedding.".format(
                             type(table.optimizer)))
      if table.name is None:
        table.name = "table_{}".format(i)
      if table.name in table_names:
        raise ValueError("Tables must have a unique name. "
                         f"Multiple tables with name {table.name} found.")
      table_names.append(table.name)

    self._built = False

  @property
  def embedding_tables(self):
    """Returns a dict of embedding tables, keyed by `TableConfig`."""
    raise NotImplementedError

  def _create_variables(self, table: tpu_embedding_v2_utils.TableConfig,
                        trainable: bool) -> Dict[Text, tf_variables.Variable]:
    """Create all variables including table variables and slot variables."""
    variable_shape = (table.vocabulary_size, table.dim)

    def getter(name, shape, dtype, initializer, trainable):
      del shape
      # _add_variable_with_custom_getter clears the shape sometimes, so we
      # take the global shape from outside the getter.
      initial_value = functools.partial(
          initializer, variable_shape, dtype=dtype)
      return tf_variables.Variable(
          name=name,
          initial_value=initial_value,
          shape=variable_shape,
          dtype=dtype,
          trainable=trainable)

    def variable_creator(name, initializer, trainable=True):
      # Use add_variable_with_custom_getter here so that we take advantage of
      # the checkpoint loading to allow restore before the variables get
      # created which avoids double initialization.
      return self._add_variable_with_custom_getter(
          name=name,
          initializer=initializer,
          shape=variable_shape,
          dtype=dtypes.float32,
          getter=getter,
          trainable=trainable)

    parameters = variable_creator(
        table.name, table.initializer, trainable=trainable)

    def slot_creator(name, initializer):
      return variable_creator(table.name + "/" + name, initializer, False)

    if table.optimizer is not None:
      slot_vars = table.optimizer._create_slots(parameters, slot_creator)  # pylint: disable=protected-access
    else:
      slot_vars = {}
    slot_vars["parameters"] = parameters
    return slot_vars

  def _create_variables_and_slots(self):
    """Create variables and slots variables for TPU embeddings."""
    raise NotImplementedError

  def build(self):
    """Create variables and slots variables for TPU embeddings."""
    if self._built:
      return
    self._variables = self._create_variables_and_slots()
    self._built = True

  def __call__(self, features: Any, weights: Optional[Any] = None) -> Any:
    """Call the mid level api to do embedding lookup."""
    if not self._built:
      self.build()
    return self.embedding_lookup(features, weights)

  def embedding_lookup(self,
                       features: Any,
                       weights: Optional[Any] = None) -> Any:
    """Lookup the embedding table using the input features."""
    raise NotImplementedError
