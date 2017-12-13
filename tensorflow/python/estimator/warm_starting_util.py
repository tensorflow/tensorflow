# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities to warm-start TF.Learn Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six

from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver


class _WarmStartSettings(
    collections.namedtuple("_WarmStartSettings", [
        "ckpt_to_initialize_from",
        "col_to_prev_vocab",
        "col_to_prev_tensor",
        "exclude_columns",
    ])):
  """Settings for warm-starting input layer in models.

  Attributes:
    ckpt_to_initialize_from: [Required] A string specifying the directory with
      checkpoint file(s) or path to checkpoint from which to warm-start the
      model parameters.
    col_to_prev_vocab: [Optional] Dict of `FeatureColumn` to vocabularies used
      for the `FeatureColumn` in `ckpt_to_initialize_from`.  Vocabularies can
      be represented either by a string (path to vocabulary), or tuple of
      (string, int), representing (path of the vocabulary, vocab_size) if only
      `vocab_size` entries of the old vocabulary were used in the checkpoint. If
      the dict is not explicitly provided, the vocabularies are assumed to be
      same between previous and present checkpoints.
    col_to_prev_tensor: [Optional] Dict of `FeatureColumn` to name of the
      variable (corresponding to the `FeatureColumn`) in
      `ckpt_to_initialize_from`. If not explicitly provided, the name of the
      variable is assumed to be same between previous and present checkpoints.
    exclude_columns: [Optional] List of `FeatureColumn`s that should not be
      warm-started from provided checkpoint.

  Example Uses:

  # Feature columns defining transformations on inputs.
  sc_vocab_file = tf.feature_column.categorical_column_with_vocabulary_file(
      "sc_vocab_file", "new_vocab.txt", vocab_size=100)
  sc_vocab_list = tf.feature_column.cateogorical_column_with_vocabulary_list(
      "sc_vocab_list", vocabulary_list=["a", "b"])

  # Warm-start all weights. The parameters corresponding to "sc_vocab_file" have
  # the same name and same vocab as current checkpoint. The parameters
  # corresponding to "sc_vocab_list" have the same name.
  ws = _WarmStartSettings(ckpt_to_initialize_from="/tmp")

  # Warm-start all weights but the parameters corresponding to "sc_vocab_file"
  # have a different vocab from the one used in current checkpoint.
  ws = _WarmStartSettings(ckpt_to_initialize_from="/tmp",
                          col_to_prev_vocab={sc_vocab_file: "old_vocab.txt"})

  # Warm-start all weights but the parameters corresponding to "sc_vocab_file"
  # have a different vocab from the one used in current checkpoint, and only
  # 100 of those entries were used.
  ws = _WarmStartSettings(ckpt_to_initialize_from="/tmp",
                          col_to_prev_vocab={sc_vocab_file:
                                             ("old_vocab.txt", 100)})

  # Warm-start all weights but the parameters corresponding to "sc_vocab_file"
  # have a different vocab from the one used in current checkpoint and the
  # parameters corresponding to "sc_vocab_list" have a different name from the
  # current checkpoint.
  ws = _WarmStartSettings(ckpt_to_initialize_from="/tmp",
                          col_to_prev_vocab={sc_vocab_file: "old_vocab.txt"},
                          col_to_prev_tensor={sc_vocab_list: "old_tensor_name"})

  # Warm-start all weights except those corrresponding to "sc_vocab_file".
  ws = _WarmStartSettings(ckpt_to_initialize_from="/tmp",
                          exclude_columns=[sc_vocab_file])
  """

  def __new__(cls,
              ckpt_to_initialize_from,
              col_to_prev_vocab=None,
              col_to_prev_tensor=None,
              exclude_columns=None):
    if not ckpt_to_initialize_from:
      raise ValueError(
          "`ckpt_to_initialize_from` MUST be set in _WarmStartSettings")
    return super(_WarmStartSettings, cls).__new__(
        cls,
        ckpt_to_initialize_from,
        col_to_prev_vocab or {},
        col_to_prev_tensor or {},
        exclude_columns or [],)


def _is_variable(x):
  return (isinstance(x, variables.Variable) or
          isinstance(x, resource_variable_ops.ResourceVariable))


def _infer_var_name(var):
  """Returns name of the `var`.

  Args:
    var: A list. The list can contain either of the following:
      (i) A single `Variable`
      (ii) A single `ResourceVariable`
      (iii) Multiple `Variable` objects which must be slices of the same larger
        variable.
      (iv) A single `PartitionedVariable`

  Returns:
    Name of the `var`
  """
  name_to_var_dict = saver.BaseSaverBuilder.OpListToDict(var)
  if len(name_to_var_dict) > 1:
    raise TypeError("`var` passed as arg violates the constraints.")
  return list(name_to_var_dict.keys())[0]


def _warmstart_var(var, prev_ckpt, prev_tensor_name=None):
  """Warm-starts given variable from `prev_tensor_name` tensor in `prev_ckpt`.

  Args:
    var: Current graph's variable that needs to be warm-started (initialized).
      Can be either of the following:
      (i) `Variable`
      (ii) `ResourceVariable`
      (iii) `PartitionedVariable`
      (iv) list of `Variable` and/or `PartitionedVariable`: The list may
        contain one or more variables that has been sharded.  For example:
        [Variable('a/part_0'), Variable('b/part_0'), Variable('a/part_1'),
         PartitionedVariable([Variable('c/part_0'), Variable('c/part_1')])]
        where we have three whole Variables represented ('a', 'b', and 'c').
    prev_ckpt: A string specifying the directory with checkpoint file(s) or path
      to checkpoint. The given checkpoint must have tensor with name
      `prev_tensor_name` (if not None) or tensor with name same as given `var`.
    prev_tensor_name: Name of the tensor to lookup in provided `prev_ckpt`. If
      None, we lookup tensor with same name as given `var`.

  Raises:
    ValueError: If prev_tensor_name is not None, but the given var represents
      more than one Variable.
    TypeError: If var is not one of the allowed types.
  """
  if _is_variable(var):
    current_var_name = _infer_var_name([var])
  elif isinstance(var, variables.PartitionedVariable):
    current_var_name = _infer_var_name([var])
    var = var._get_variable_list()  # pylint: disable=protected-access
  elif (isinstance(var, list) and all(
      _is_variable(v) or isinstance(v, variables.PartitionedVariable)
      for v in var)):
    # Convert length-1 lists of vars to single tf.Variables.  This ensures that
    # checkpoint_utils.init_from_checkpoint() doesn't incorrectly assume
    # slice info is present.
    if len(var) == 1:
      current_var_name = _infer_var_name(var)
      var = var[0]
    else:
      # If we have multiple elements in var, we cannot assume they all
      # represent the same Variable.
      name_to_var_dict = saver.BaseSaverBuilder.OpListToDict(
          var, convert_variable_to_tensor=False)
      if prev_tensor_name:
        # Providing a prev_tensor_name is only viable if var representes a
        # single Variable.
        if len(name_to_var_dict) > 1:
          raise ValueError("var represented more than one Variable, but "
                           "prev_tensor_name was provided.")
        checkpoint_utils.init_from_checkpoint(prev_ckpt, {
            prev_tensor_name: var
        })
      else:
        # OpListToDict gives us roughly what we need, but
        # the values in the dict may be PartitionedVariables (which
        # init_from_checkpoint does not expect) that we need to convert to
        # lists.
        name_to_var_dict_fixed = {}
        for name, var in six.iteritems(name_to_var_dict):
          if isinstance(var, variables.PartitionedVariable):
            name_to_var_dict_fixed[name] = var._get_variable_list()  # pylint: disable=protected-access
          else:
            name_to_var_dict_fixed[name] = var
        checkpoint_utils.init_from_checkpoint(prev_ckpt, name_to_var_dict_fixed)
      return
  else:
    raise TypeError(
        "var MUST be one of the following: a Variable, PartitionedVariable, or "
        "list of Variable's and/or PartitionedVariable's, but is {}".format(
            type(var)))
  if not prev_tensor_name:
    # Assume tensor name remains the same.
    prev_tensor_name = current_var_name
  checkpoint_utils.init_from_checkpoint(prev_ckpt, {prev_tensor_name: var})


# pylint: disable=protected-access
# Accesses protected members of tf.Variable to reset the variable's internal
# state.
def _warmstart_var_with_vocab(var,
                              current_vocab_path,
                              current_vocab_size,
                              prev_ckpt,
                              prev_vocab_path,
                              previous_vocab_size=-1,
                              current_oov_buckets=0,
                              prev_tensor_name=None,
                              initializer=None):
  """Warm-starts given variable from `prev_tensor_name` tensor in `prev_ckpt`.

  Use this method when the `var` is backed by vocabulary. This method stitches
  the given `var` such that values corresponding to individual features in the
  vocabulary remain consistent irrespective of changing order of the features
  between old and new vocabularies.

  Args:
    var: Current graph's variable that needs to be warm-started (initialized).
      Can be either of the following:
      (i) `Variable`
      (ii) `ResourceVariable`
      (iii) list of `Variable`: The list must contain slices of the same larger
        variable.
      (iv) `PartitionedVariable`
    current_vocab_path: Path to the vocab file used for the given `var`.
    current_vocab_size: An `int` specifying the number of entries in the current
      vocab.
    prev_ckpt: A string specifying the directory with checkpoint file(s) or path
      to checkpoint. The given checkpoint must have tensor with name
      `prev_tensor_name` (if not None) or tensor with name same as given `var`.
    prev_vocab_path: Path to the vocab file used for the tensor in `prev_ckpt`.
    previous_vocab_size: If provided, will constrain previous vocab to the first
      `previous_vocab_size` entries.  -1 means use the entire previous vocab.
    current_oov_buckets: An `int` specifying the number of out-of-vocabulary
      buckets used for given `var`.
    prev_tensor_name: Name of the tensor to lookup in provided `prev_ckpt`. If
      None, we lookup tensor with same name as given `var`.
    initializer: Variable initializer to be used for missing entries.  If None,
      missing entries will be zero-initialized.

  Raises:
    ValueError: If required args are not provided.
  """
  if not (current_vocab_path and current_vocab_size and prev_ckpt and
          prev_vocab_path):
    raise ValueError("Invalid args: Must provide all of [current_vocab_path, "
                     "current_vocab_size, prev_ckpt, prev_vocab_path}.")
  if _is_variable(var):
    var = [var]
  elif isinstance(var, list) and all(_is_variable(v) for v in var):
    var = var
  elif isinstance(var, variables.PartitionedVariable):
    var = var._get_variable_list()
  else:
    raise TypeError(
        "var MUST be one of the following: a Variable, list of Variable or "
        "PartitionedVariable, but is {}".format(type(var)))

  if not prev_tensor_name:
    # Assume tensor name remains the same.
    prev_tensor_name = _infer_var_name(var)

  for v in var:
    v_shape = v.get_shape().as_list()
    slice_info = v._get_save_slice_info()
    partition_info = None
    if slice_info:
      partition_info = variable_scope._PartitionInfo(
          full_shape=slice_info.full_shape,
          var_offset=slice_info.var_offset)

    # TODO(vihanjain): Support _WarmstartSettings where class vocabularies need
    # remapping too.
    init = checkpoint_ops._load_and_remap_matrix_initializer(
        ckpt_path=saver.latest_checkpoint(prev_ckpt),
        old_tensor_name=prev_tensor_name,
        new_row_vocab_size=current_vocab_size,
        new_col_vocab_size=v_shape[1],
        old_row_vocab_size=previous_vocab_size,
        old_row_vocab_file=prev_vocab_path,
        new_row_vocab_file=current_vocab_path,
        old_col_vocab_file=None,
        new_col_vocab_file=None,
        num_row_oov_buckets=current_oov_buckets,
        num_col_oov_buckets=0,
        initializer=initializer)
    new_init_val = ops.convert_to_tensor(
        init(shape=v_shape, partition_info=partition_info))
    v._initializer_op = state_ops.assign(v, new_init_val)
# pylint: enable=protected-access


def _warmstart_input_layer(cols_to_vars, warmstart_settings):
  """Warm-starts input layer of a model using given settings.

  Args:
    cols_to_vars: Dict of feature columns to corresponding graph variables.
    warmstart_settings: An object of `_WarmStartSettings`.

    Typical usage example:

    ```python
    tfcl = tf.contrib.layers
    # Define features and transformations.
    sc_vocab_list = tf.feature_column.categorical_column_with_vocabulary_list(
        "sc_vocab_list", vocabulary_list=["a", "b"])
    sc_vocab_file = tf.feature_column.categorical_column_with_vocabulary_file(
        "sc_vocab_file", "new_vocab.txt", vocab_size=100)
    cross = tf.feature_column.crossed_column(
      [sc_vocab_list, sc_vocab_file], hash_bucket_size=5000)

    all_cols = set(sc_vocab_list, sc_vocab_file, cross)
    batch_features = tf.parse_example(
        serialized=serialized_examples,
        features=tf.contrib.layers.create_feature_spec_for_parsing(all_cols))

    cols_to_vars = {}
    tf.feature_column.linear_model(
        features=batch_features,
        feature_columns=all_cols,
        units=1,
        cols_to_vars=cols_to_vars)

    # Warm-start entire input layer.
    ws_settings = _WarmStartSettings(
        "/tmp/prev_model_dir",
        col_to_prev_vocab={sc_vocab_file: "old_vocab.txt"})
    _warmstart_input_layer(cols_to_vars, ws_settings)
    # Warm-start bias too.
    _warmstart_var(cols_to_vars['bias'], ws_settings.ckpt_to_initialize_from)
    ```

    The above example effectively warm-starts full linear model.

  Raises:
    ValueError: If a column in cols_to_vars has an entry in
      warmstart_settings.cols_to_prev_vocab, but is not an instance of
      _VocabularyFileCategoricalColumn or _EmbeddingColumn.
  """
  for col, var in six.iteritems(cols_to_vars):
    if not isinstance(col, feature_column._FeatureColumn):  # pylint: disable=protected-access
      raise TypeError(
          "Keys in dict `cols_to_vars` must be of type FeatureColumn. Found "
          "key of type: {}".format(type(col)))
    if col in warmstart_settings.exclude_columns:
      logging.info("Skipping warm-starting column: {}".format(col.name))
      continue

    prev_tensor_name = warmstart_settings.col_to_prev_tensor.get(col)
    # pylint: disable=protected-access
    is_sparse_vocab_column = isinstance(
        col, feature_column._VocabularyFileCategoricalColumn)
    is_embedding_vocab_column = (
        isinstance(col, feature_column._EmbeddingColumn) and
        isinstance(col.categorical_column,
                   feature_column._VocabularyFileCategoricalColumn))
    if is_sparse_vocab_column or is_embedding_vocab_column:
      # pylint: enable=protected-access
      initializer = None
      if is_embedding_vocab_column:
        initializer = col.initializer
        vocabulary_file = col.categorical_column.vocabulary_file
        vocabulary_size = col.categorical_column.vocabulary_size
        num_oov_buckets = col.categorical_column.num_oov_buckets
      else:
        vocabulary_file = col.vocabulary_file
        vocabulary_size = col.vocabulary_size
        num_oov_buckets = col.num_oov_buckets
      prev_vocab = warmstart_settings.col_to_prev_vocab.get(
          col, vocabulary_file)
      if isinstance(prev_vocab, str):
        prev_vocab_path = prev_vocab
        previous_vocab_size = -1
        logging.info(
            "Warm-starting column: {}; prev_vocab: {}; "
            "prev_tensor: {}".format(col.name, prev_vocab_path,
                                     (prev_tensor_name or "Unchanged")))
      elif isinstance(prev_vocab, tuple):
        prev_vocab_path = prev_vocab[0]
        previous_vocab_size = prev_vocab[1]
        logging.info("Warm-starting column: {}; prev_vocab: {} (first {} "
                     "entries); prev_tensor: {}".format(
                         col.name, prev_vocab_path, previous_vocab_size,
                         (prev_tensor_name or "Unchanged")))

      _warmstart_var_with_vocab(
          var,
          current_vocab_path=vocabulary_file,
          current_vocab_size=vocabulary_size,
          prev_ckpt=warmstart_settings.ckpt_to_initialize_from,
          prev_vocab_path=prev_vocab_path,
          previous_vocab_size=previous_vocab_size,
          current_oov_buckets=num_oov_buckets,
          prev_tensor_name=prev_tensor_name,
          initializer=initializer)
    else:
      if col in warmstart_settings.col_to_prev_vocab:
        raise ValueError("Vocabulary provided for column %s which is not a "
                         "_VocabularyFileCategoricalColumn or _EmbeddingColumn")
      logging.info("Warm-starting column: {}; prev_tensor: {}".format(
          col.name, prev_tensor_name or "Unchanged"))
      _warmstart_var(var, warmstart_settings.ckpt_to_initialize_from,
                     prev_tensor_name)
