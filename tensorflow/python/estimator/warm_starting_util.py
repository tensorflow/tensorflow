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

from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver
from tensorflow.python.util.tf_export import tf_export


@tf_export("estimator.VocabInfo")
class VocabInfo(
    collections.namedtuple("VocabInfo", [
        "new_vocab",
        "new_vocab_size",
        "num_oov_buckets",
        "old_vocab",
        "old_vocab_size",
        "backup_initializer",
    ])):
  """Vocabulary information for WarmStartSettings.

  See @{tf.estimator.WarmStartSettings$WarmStartSettings} for examples of using
  VocabInfo to warm-start.

  Attributes:
    new_vocab: [Required] A path to the new vocabulary file (used with the
      model to be trained).
    new_vocab_size: [Required] An integer indicating how many entries of the new
      vocabulary will used in training.
    num_oov_buckets: [Required] An integer indicating how many OOV buckets are
      associated with the vocabulary.
    old_vocab: [Required] A path to the old vocabulary file (used with the
      checkpoint to be warm-started from).
    old_vocab_size: [Optional] An integer indicating how many entries of the old
      vocabulary were used in the creation of the checkpoint. If not provided,
      the entire old vocabulary will be used.
    backup_initializer: [Optional] A variable initializer used for variables
      corresponding to new vocabulary entries and OOV. If not provided, these
      entries will be zero-initialized.
  """

  def __new__(cls,
              new_vocab,
              new_vocab_size,
              num_oov_buckets,
              old_vocab,
              old_vocab_size=-1,
              backup_initializer=None):
    return super(VocabInfo, cls).__new__(
        cls,
        new_vocab,
        new_vocab_size,
        num_oov_buckets,
        old_vocab,
        old_vocab_size,
        backup_initializer,
    )


@tf_export("estimator.WarmStartSettings")
class WarmStartSettings(
    collections.namedtuple("WarmStartSettings", [
        "ckpt_to_initialize_from",
        "vars_to_warm_start",
        "var_name_to_vocab_info",
        "var_name_to_prev_var_name",
    ])):
  """Settings for warm-starting in Estimators.

  Example Use with canned `DNNEstimator`:

  ```
  emb_vocab_file = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_file(
          "sc_vocab_file", "new_vocab.txt", vocab_size=100),
      dimension=8)
  emb_vocab_list = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          "sc_vocab_list", vocabulary_list=["a", "b"]),
      dimension=8)
  estimator = tf.estimator.DNNClassifier(
    hidden_units=[128, 64], feature_columns=[emb_vocab_file, emb_vocab_list],
    warm_start_from=ws)
  ```

  where `ws` could be defined as:

  Warm-start all weights in the model (input layer and hidden weights).
  Either the directory or a specific checkpoint can be provided (in the case
  of the former, the latest checkpoint will be used):

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp")
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp/model-1000")
  ```

  Warm-start only the embeddings (input layer):

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp",
                         vars_to_warm_start=".*input_layer.*")
  ```

  Warm-start all weights but the embedding parameters corresponding to
  `sc_vocab_file` have a different vocab from the one used in the current
  model:

  ```
  vocab_info = ws_util.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt"
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start only `sc_vocab_file` embeddings (and no other variables), which
  have a different vocab from the one used in the current model:

  ```
  vocab_info = ws_util.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt"
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      vars_to_warm_start=None,
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start all weights but the parameters corresponding to `sc_vocab_file`
  have a different vocab from the one used in current checkpoint, and only
  100 of those entries were used:

  ```
  vocab_info = ws_util.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt",
      old_vocab_size=100
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start all weights but the parameters corresponding to `sc_vocab_file`
  have a different vocab from the one used in current checkpoint and the
  parameters corresponding to `sc_vocab_list` have a different name from the
  current checkpoint:

  ```
  vocab_info = ws_util.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt",
      old_vocab_size=100
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      },
      var_name_to_prev_var_name={
          "input_layer/sc_vocab_list_embedding/embedding_weights":
              "old_tensor_name"
      })
  ```

  Attributes:
    ckpt_to_initialize_from: [Required] A string specifying the directory with
      checkpoint file(s) or path to checkpoint from which to warm-start the
      model parameters.
    vars_to_warm_start: [Optional] A regular expression that captures which
      variables to warm-start (see tf.get_collection).  Defaults to `'.*'`,
      which warm-starts all variables.  If `None` is explicitly given, only
      variables specified in `var_name_to_vocab_info` will be warm-started.
    var_name_to_vocab_info: [Optional] Dict of variable names (strings) to
      VocabInfo. The variable names should be "full" variables, not the names
      of the partitions.  If not explicitly provided, the variable is assumed to
      have no vocabulary.
    var_name_to_prev_var_name: [Optional] Dict of variable names (strings) to
      name of the previously-trained variable in `ckpt_to_initialize_from`. If
      not explicitly provided, the name of the variable is assumed to be same
      between previous checkpoint and current model.
  """

  def __new__(cls,
              ckpt_to_initialize_from,
              vars_to_warm_start=".*",
              var_name_to_vocab_info=None,
              var_name_to_prev_var_name=None):
    if not ckpt_to_initialize_from:
      raise ValueError(
          "`ckpt_to_initialize_from` MUST be set in WarmStartSettings")
    return super(WarmStartSettings, cls).__new__(
        cls,
        ckpt_to_initialize_from,
        vars_to_warm_start,
        var_name_to_vocab_info or {},
        var_name_to_prev_var_name or {},
    )


def _is_variable(x):
  return (isinstance(x, variables_lib.Variable) or
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
    raise TypeError("`var` = %s passed as arg violates the constraints.  "
                    "name_to_var_dict = %s" % (var, name_to_var_dict))
  return list(name_to_var_dict.keys())[0]


def _warm_start_var(var, prev_ckpt, prev_tensor_name=None):
  """Warm-starts given variable from `prev_tensor_name` tensor in `prev_ckpt`.

  Args:
    var: Current graph's variable that needs to be warm-started (initialized).
      Can be either of the following:
      (i) `Variable`
      (ii) `ResourceVariable`
      (iii) list of `Variable`: The list must contain slices of the same larger
        variable.
      (iv) `PartitionedVariable`
    prev_ckpt: A string specifying the directory with checkpoint file(s) or path
      to checkpoint. The given checkpoint must have tensor with name
      `prev_tensor_name` (if not None) or tensor with name same as given `var`.
    prev_tensor_name: Name of the tensor to lookup in provided `prev_ckpt`. If
      None, we lookup tensor with same name as given `var`.
  """
  if _is_variable(var):
    current_var_name = _infer_var_name([var])
  elif isinstance(var, list) and all(_is_variable(v) for v in var):
    current_var_name = _infer_var_name(var)
  elif isinstance(var, variables_lib.PartitionedVariable):
    current_var_name = _infer_var_name([var])
    var = var._get_variable_list()  # pylint: disable=protected-access
  else:
    raise TypeError(
        "var MUST be one of the following: a Variable, list of Variable or "
        "PartitionedVariable, but is {}".format(type(var)))
  if not prev_tensor_name:
    # Assume tensor name remains the same.
    prev_tensor_name = current_var_name
  checkpoint_utils.init_from_checkpoint(prev_ckpt, {prev_tensor_name: var})


# pylint: disable=protected-access
# Accesses protected members of tf.Variable to reset the variable's internal
# state.
def _warm_start_var_with_vocab(var,
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
  elif isinstance(var, variables_lib.PartitionedVariable):
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

    # TODO(eddz): Support WarmStartSettings where class vocabularies need
    # remapping too.
    init = checkpoint_ops._load_and_remap_matrix_initializer(
        ckpt_path=checkpoint_utils._get_checkpoint_filename(prev_ckpt),
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


def _warm_start(warm_start_settings):
  """Warm-starts a model using the given settings.

  If you are using a tf.estimator.Estimator, this will automatically be called
  during training.

  Args:
    warm_start_settings: An object of `WarmStartSettings`.
  Raises:
    ValueError: If the WarmStartSettings contains prev_var_name or VocabInfo
      configuration for variable names that are not used.  This is to ensure
      a stronger check for variable configuration than relying on users to
      examine the logs.
  """
  logging.info("Warm-starting from: %s",
               (warm_start_settings.ckpt_to_initialize_from,))
  # We have to deal with partitioned variables, since get_collection flattens
  # out the list.
  grouped_variables = {}
  # Both warm_start_settings.vars_to_warm_start = '.*' and
  # warm_start_settings.vars_to_warm_start = None will match everything here.
  for v in ops.get_collection(
      # TODO(eddz): Allow for different collections here (to support
      # warm-starting accumulators).
      ops.GraphKeys.TRAINABLE_VARIABLES,
      scope=warm_start_settings.vars_to_warm_start):
    if not isinstance(v, list):
      var_name = _infer_var_name([v])
    else:
      var_name = _infer_var_name(v)
    grouped_variables.setdefault(var_name, []).append(v)

  # Keep track of which var_names in var_name_to_prev_var_name and
  # var_name_to_vocab_info have been used.  Err on the safer side by throwing an
  # exception if any are unused by the end of the loop.  It is easy to misname
  # a variable during this configuration, in which case without this check, we
  # would fail to warm-start silently.
  prev_var_name_used = set()
  vocab_info_used = set()

  for var_name, variable in six.iteritems(grouped_variables):
    prev_var_name = warm_start_settings.var_name_to_prev_var_name.get(var_name)
    if prev_var_name:
      prev_var_name_used.add(var_name)
    vocab_info = warm_start_settings.var_name_to_vocab_info.get(var_name)
    if vocab_info:
      vocab_info_used.add(var_name)
      logging.info(
          "Warm-starting variable: {}; current_vocab: {} current_vocab_size: {}"
          " prev_vocab: {} prev_vocab_size: {} current_oov: {} prev_tensor: {}"
          " initializer: {}".format(
              var_name,
              vocab_info.new_vocab,
              vocab_info.new_vocab_size,
              vocab_info.old_vocab,
              (vocab_info.old_vocab_size if vocab_info.old_vocab_size > 0
               else "All"),
              vocab_info.num_oov_buckets,
              prev_var_name or "Unchanged",
              vocab_info.backup_initializer or "zero-initialized"))
      _warm_start_var_with_vocab(
          variable,
          current_vocab_path=vocab_info.new_vocab,
          current_vocab_size=vocab_info.new_vocab_size,
          prev_ckpt=warm_start_settings.ckpt_to_initialize_from,
          prev_vocab_path=vocab_info.old_vocab,
          previous_vocab_size=vocab_info.old_vocab_size,
          current_oov_buckets=vocab_info.num_oov_buckets,
          prev_tensor_name=prev_var_name,
          initializer=vocab_info.backup_initializer)
    else:
      # For the special value of warm_start_settings.vars_to_warm_start = None,
      # we only warm-start variables with explicitly specified vocabularies.
      if warm_start_settings.vars_to_warm_start:
        logging.info("Warm-starting variable: {}; prev_var_name: {}".format(
            var_name, prev_var_name or "Unchanged"))
        # Because we use a default empty list in grouped_variables, single
        # unpartitioned variables will be lists here, which we rectify in order
        # for init_from_checkpoint logic to work correctly.
        if len(variable) == 1:
          variable = variable[0]
        _warm_start_var(variable, warm_start_settings.ckpt_to_initialize_from,
                        prev_var_name)

  prev_var_name_not_used = set(
      warm_start_settings.var_name_to_prev_var_name.keys()) - prev_var_name_used
  vocab_info_not_used = set(
      warm_start_settings.var_name_to_vocab_info.keys()) - vocab_info_used

  if prev_var_name_not_used:
    raise ValueError(
        "You provided the following variables in "
        "warm_start_settings.var_name_to_prev_var_name that were not used: "
        "{0}.  Perhaps you misspelled them?  Here is the list of viable "
        "variable names: {1}".format(prev_var_name_not_used,
                                     grouped_variables.keys()))
  if vocab_info_not_used:
    raise ValueError(
        "You provided the following variables in "
        "warm_start_settings.var_name_to_vocab_info that were not used: {0}. "
        " Perhaps you misspelled them?  Here is the list of viable variable "
        "names: {1}".format(vocab_info_not_used, grouped_variables.keys()))


def _get_default_warm_start_settings(warm_start_from):
  """Returns default WarmStartSettings.

  Args:
    warm_start_from: Either a string representing the filepath of a checkpoint
      to initialize from, or an instance of WarmStartSettings.

  Returns:
    Either None or an instance of WarmStartSettings.

  Raises:
    ValueError: If warm_start_from is not None but is neither a string nor an
      instance of WarmStartSettings.
  """
  if warm_start_from is None:
    return None
  if isinstance(warm_start_from, six.string_types):
    return WarmStartSettings(ckpt_to_initialize_from=warm_start_from)
  elif isinstance(warm_start_from, WarmStartSettings):
    return warm_start_from
  else:
    raise ValueError("warm_start_from must be a string or a WarmStartSettings")
