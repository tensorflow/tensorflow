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

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.VocabInfo"])
class VocabInfo(
    collections.namedtuple("VocabInfo", [
        "new_vocab",
        "new_vocab_size",
        "num_oov_buckets",
        "old_vocab",
        "old_vocab_size",
        "backup_initializer",
        "axis",
    ])):
  """Vocabulary information for warm-starting.

  See `tf.estimator.WarmStartSettings` for examples of using
  VocabInfo to warm-start.

  Args:
    new_vocab: [Required] A path to the new vocabulary file (used with the model
      to be trained).
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
    axis: [Optional] Denotes what axis the vocabulary corresponds to.  The
      default, 0, corresponds to the most common use case (embeddings or
      linear weights for binary classification / regression).  An axis of 1
      could be used for warm-starting output layers with class vocabularies.

  Returns:
    A `VocabInfo` which represents the vocabulary information for warm-starting.

  Raises:
    ValueError: `axis` is neither 0 or 1.

      Example Usage:
```python
      embeddings_vocab_info = tf.VocabInfo(
          new_vocab='embeddings_vocab',
          new_vocab_size=100,
          num_oov_buckets=1,
          old_vocab='pretrained_embeddings_vocab',
          old_vocab_size=10000,
          backup_initializer=tf.compat.v1.truncated_normal_initializer(
              mean=0.0, stddev=(1 / math.sqrt(embedding_dim))),
          axis=0)

      softmax_output_layer_kernel_vocab_info = tf.VocabInfo(
          new_vocab='class_vocab',
          new_vocab_size=5,
          num_oov_buckets=0,  # No OOV for classes.
          old_vocab='old_class_vocab',
          old_vocab_size=8,
          backup_initializer=tf.compat.v1.glorot_uniform_initializer(),
          axis=1)

      softmax_output_layer_bias_vocab_info = tf.VocabInfo(
          new_vocab='class_vocab',
          new_vocab_size=5,
          num_oov_buckets=0,  # No OOV for classes.
          old_vocab='old_class_vocab',
          old_vocab_size=8,
          backup_initializer=tf.compat.v1.zeros_initializer(),
          axis=0)

      #Currently, only axis=0 and axis=1 are supported.
  ```
  """

  def __new__(cls,
              new_vocab,
              new_vocab_size,
              num_oov_buckets,
              old_vocab,
              old_vocab_size=-1,
              backup_initializer=None,
              axis=0):
    if axis != 0 and axis != 1:
      raise ValueError("The only supported values for the axis argument are 0 "
                       "and 1.  Provided axis: {}".format(axis))

    return super(VocabInfo, cls).__new__(
        cls,
        new_vocab,
        new_vocab_size,
        num_oov_buckets,
        old_vocab,
        old_vocab_size,
        backup_initializer,
        axis,
    )


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
  name_to_var_dict = saveable_object_util.op_list_to_dict(var)
  if len(name_to_var_dict) > 1:
    raise TypeError("`var` = %s passed as arg violates the constraints.  "
                    "name_to_var_dict = %s" % (var, name_to_var_dict))
  return list(name_to_var_dict.keys())[0]


def _get_var_info(var, prev_tensor_name=None):
  """Helper method for standarizing Variable and naming.

  Args:
    var: Current graph's variable that needs to be warm-started (initialized).
      Can be either of the following: (i) `Variable` (ii) `ResourceVariable`
      (iii) list of `Variable`: The list must contain slices of the same larger
        variable. (iv) `PartitionedVariable`
    prev_tensor_name: Name of the tensor to lookup in provided `prev_ckpt`. If
      None, we lookup tensor with same name as given `var`.

  Returns:
    A tuple of the Tensor name and var.
  """
  if checkpoint_utils._is_variable(var):  # pylint: disable=protected-access
    current_var_name = _infer_var_name([var])
  elif (isinstance(var, list) and
        all(checkpoint_utils._is_variable(v) for v in var)):  # pylint: disable=protected-access
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

  return prev_tensor_name, var


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
                               initializer=None,
                               axis=0):
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
    axis: Axis of the variable that the provided vocabulary corresponds to.

  Raises:
    ValueError: If required args are not provided.
  """
  if not (current_vocab_path and current_vocab_size and prev_ckpt and
          prev_vocab_path):
    raise ValueError("Invalid args: Must provide all of [current_vocab_path, "
                     "current_vocab_size, prev_ckpt, prev_vocab_path}.")
  if checkpoint_utils._is_variable(var):
    var = [var]
  elif (isinstance(var, list) and
        all(checkpoint_utils._is_variable(v) for v in var)):
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

  total_v_first_axis = sum(v.get_shape().as_list()[0] for v in var)
  for v in var:
    v_shape = v.get_shape().as_list()
    slice_info = v._get_save_slice_info()
    partition_info = None
    if slice_info:
      partition_info = variable_scope._PartitionInfo(
          full_shape=slice_info.full_shape, var_offset=slice_info.var_offset)

    if axis == 0:
      new_row_vocab_size = current_vocab_size
      new_col_vocab_size = v_shape[1]
      old_row_vocab_size = previous_vocab_size
      old_row_vocab_file = prev_vocab_path
      new_row_vocab_file = current_vocab_path
      old_col_vocab_file = None
      new_col_vocab_file = None
      num_row_oov_buckets = current_oov_buckets
      num_col_oov_buckets = 0
    elif axis == 1:
      # Note that we must compute this value across all partitions, whereas
      # in the axis = 0 case, we can simply use v_shape[1] because we don't
      # allow partitioning across axis = 1.
      new_row_vocab_size = total_v_first_axis
      new_col_vocab_size = current_vocab_size
      old_row_vocab_size = -1
      old_row_vocab_file = None
      new_row_vocab_file = None
      old_col_vocab_file = prev_vocab_path
      new_col_vocab_file = current_vocab_path
      num_row_oov_buckets = 0
      num_col_oov_buckets = current_oov_buckets
    else:
      raise ValueError("The only supported values for the axis argument are 0 "
                       "and 1.  Provided axis: {}".format(axis))

    init = checkpoint_ops._load_and_remap_matrix_initializer(
        ckpt_path=checkpoint_utils._get_checkpoint_filename(prev_ckpt),
        old_tensor_name=prev_tensor_name,
        new_row_vocab_size=new_row_vocab_size,
        new_col_vocab_size=new_col_vocab_size,
        old_row_vocab_size=old_row_vocab_size,
        old_row_vocab_file=old_row_vocab_file,
        new_row_vocab_file=new_row_vocab_file,
        old_col_vocab_file=old_col_vocab_file,
        new_col_vocab_file=new_col_vocab_file,
        num_row_oov_buckets=num_row_oov_buckets,
        num_col_oov_buckets=num_col_oov_buckets,
        initializer=initializer)
    new_init_val = ops.convert_to_tensor(
        init(shape=v_shape, partition_info=partition_info))
    v._initializer_op = state_ops.assign(v, new_init_val)


# pylint: enable=protected-access


def _get_grouped_variables(vars_to_warm_start):
  """Collects and groups (possibly partitioned) variables into a dictionary.

  The variables can be provided explicitly through vars_to_warm_start, or they
  are retrieved from collections (see below).

  Args:
    vars_to_warm_start: One of the following:

      - A regular expression (string) that captures which variables to
        warm-start (see tf.compat.v1.get_collection).  This expression will
        only consider variables in the TRAINABLE_VARIABLES collection.
      - A list of strings, each representing a full variable name to warm-start.
        These will consider variables in GLOBAL_VARIABLES collection.
      - A list of Variables to warm-start.
      - `None`, in which case all variables in TRAINABLE_VARIABLES will be used.
  Returns:
    A dictionary mapping variable names (strings) to lists of Variables.
  Raises:
    ValueError: If vars_to_warm_start is not a string, `None`, a list of
      `Variables`, or a list of strings.
  """
  if isinstance(vars_to_warm_start, str) or vars_to_warm_start is None:
    # Both vars_to_warm_start = '.*' and vars_to_warm_start = None will match
    # everything (in TRAINABLE_VARIABLES) here.
    logging.info("Warm-starting variables only in TRAINABLE_VARIABLES.")
    list_of_vars = ops.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES, scope=vars_to_warm_start)
  elif isinstance(vars_to_warm_start, list):
    if all(isinstance(v, str) for v in vars_to_warm_start):
      list_of_vars = []
      for v in vars_to_warm_start:
        list_of_vars += ops.get_collection(
            ops.GraphKeys.GLOBAL_VARIABLES, scope=v)
    elif all(checkpoint_utils._is_variable(v) for v in vars_to_warm_start):  # pylint: disable=protected-access
      list_of_vars = vars_to_warm_start
    else:
      raise ValueError("If `vars_to_warm_start` is a list, it must be all "
                       "`Variable` or all `str`.  Given types are {}".format(
                           [type(v) for v in vars_to_warm_start]))
  else:
    raise ValueError("`vars_to_warm_start must be a `list` or `str`.  Given "
                     "type is {}".format(type(vars_to_warm_start)))
  # We have to deal with partitioned variables, since get_collection flattens
  # out the list.
  grouped_variables = {}
  for v in list_of_vars:
    if not isinstance(v, list):
      var_name = _infer_var_name([v])
    else:
      var_name = _infer_var_name(v)
    grouped_variables.setdefault(var_name, []).append(v)

  return grouped_variables


def _get_object_checkpoint_renames(path, variable_names):
  """Returns a dictionary mapping variable names to checkpoint keys.

  The warm-starting utility expects variable names to match with the variable
  names in the checkpoint. For object-based checkpoints, the variable names
  and names in the checkpoint are different. Thus, for object-based checkpoints,
  this function is used to obtain the map from variable names to checkpoint
  keys.

  Args:
    path: path to checkpoint directory or file.
    variable_names: list of variable names to load from the checkpoint.

  Returns:
    If the checkpoint is object-based, this function returns a map from variable
    names to their corresponding checkpoint keys.
    If the checkpoint is name-based, this returns an empty dict.

  Raises:
    ValueError: If the object-based checkpoint is missing variables.
  """
  fname = checkpoint_utils._get_checkpoint_filename(path)  # pylint: disable=protected-access
  try:
    names_to_keys = saver_lib.object_graph_key_mapping(fname)
  except errors.NotFoundError:
    # If an error is raised from `object_graph_key_mapping`, then the
    # checkpoint is name-based. There are no renames, so return an empty dict.
    return {}

  missing_names = set(variable_names) - set(names_to_keys.keys())
  if missing_names:
    raise ValueError(
        "Attempting to warm-start from an object-based checkpoint, but found "
        "that the checkpoint did not contain values for all variables. The "
        "following variables were missing: {}"
        .format(missing_names))
  return {name: names_to_keys[name] for name in variable_names}


@tf_export(v1=["train.warm_start"])
def warm_start(ckpt_to_initialize_from,
               vars_to_warm_start=".*",
               var_name_to_vocab_info=None,
               var_name_to_prev_var_name=None):
  """Warm-starts a model using the given settings.

  If you are using a tf.estimator.Estimator, this will automatically be called
  during training.

  Args:
    ckpt_to_initialize_from: [Required] A string specifying the directory with
      checkpoint file(s) or path to checkpoint from which to warm-start the
      model parameters.
    vars_to_warm_start: [Optional] One of the following:

      - A regular expression (string) that captures which variables to
        warm-start (see tf.compat.v1.get_collection).  This expression will only
        consider variables in the TRAINABLE_VARIABLES collection -- if you need
        to warm-start non_TRAINABLE vars (such as optimizer accumulators or
        batch norm statistics), please use the below option.
      - A list of strings, each a regex scope provided to
        tf.compat.v1.get_collection with GLOBAL_VARIABLES (please see
        tf.compat.v1.get_collection).  For backwards compatibility reasons,
        this is separate from the single-string argument type.
      - A list of Variables to warm-start.  If you do not have access to the
        `Variable` objects at the call site, please use the above option.
      - `None`, in which case only TRAINABLE variables specified in
        `var_name_to_vocab_info` will be warm-started.

      Defaults to `'.*'`, which warm-starts all variables in the
      TRAINABLE_VARIABLES collection.  Note that this excludes variables such
      as accumulators and moving statistics from batch norm.
    var_name_to_vocab_info: [Optional] Dict of variable names (strings) to
      `tf.estimator.VocabInfo`. The variable names should be "full" variables,
      not the names of the partitions.  If not explicitly provided, the variable
      is assumed to have no (changes to) vocabulary.
    var_name_to_prev_var_name: [Optional] Dict of variable names (strings) to
      name of the previously-trained variable in `ckpt_to_initialize_from`. If
      not explicitly provided, the name of the variable is assumed to be same
      between previous checkpoint and current model.  Note that this has no
      effect on the set of variables that is warm-started, and only controls
      name mapping (use `vars_to_warm_start` for controlling what variables to
      warm-start).

  Raises:
    ValueError: If the WarmStartSettings contains prev_var_name or VocabInfo
      configuration for variable names that are not used.  This is to ensure
      a stronger check for variable configuration than relying on users to
      examine the logs.
  """
  logging.info("Warm-starting from: {}".format(ckpt_to_initialize_from))
  grouped_variables = _get_grouped_variables(vars_to_warm_start)

  if var_name_to_vocab_info is None:
    var_name_to_vocab_info = {}

  if not var_name_to_prev_var_name:
    # Detect whether the checkpoint is object-based, in which case the
    # var_name_to_prev_var_name dictionary should map variable names to
    # checkpoint keys. If the user has specified var_name_to_prev_var_name, we
    # do not override it.
    var_name_to_prev_var_name = _get_object_checkpoint_renames(
        ckpt_to_initialize_from, grouped_variables.keys())

  warmstarted_count = 0

  # Keep track of which var_names in var_name_to_prev_var_name and
  # var_name_to_vocab_info have been used.  Err on the safer side by throwing an
  # exception if any are unused by the end of the loop.  It is easy to misname
  # a variable during this configuration, in which case without this check, we
  # would fail to warm-start silently.
  prev_var_name_used = set()
  vocab_info_used = set()

  # Group the vocabless vars into one call to init_from_checkpoint.
  vocabless_vars = {}
  for var_name, variable in six.iteritems(grouped_variables):
    prev_var_name = var_name_to_prev_var_name.get(var_name)
    if prev_var_name:
      prev_var_name_used.add(var_name)
    vocab_info = var_name_to_vocab_info.get(var_name)
    if vocab_info:
      vocab_info_used.add(var_name)
      warmstarted_count += 1
      logging.debug(
          "Warm-starting variable: {}; current_vocab: {} current_vocab_size: {}"
          " prev_vocab: {} prev_vocab_size: {} current_oov: {} prev_tensor: {}"
          " initializer: {}".format(
              var_name, vocab_info.new_vocab, vocab_info.new_vocab_size,
              vocab_info.old_vocab, (vocab_info.old_vocab_size if
                                     vocab_info.old_vocab_size > 0 else "All"),
              vocab_info.num_oov_buckets, prev_var_name or "Unchanged",
              vocab_info.backup_initializer or "zero-initialized"))
      _warm_start_var_with_vocab(
          variable,
          current_vocab_path=vocab_info.new_vocab,
          current_vocab_size=vocab_info.new_vocab_size,
          prev_ckpt=ckpt_to_initialize_from,
          prev_vocab_path=vocab_info.old_vocab,
          previous_vocab_size=vocab_info.old_vocab_size,
          current_oov_buckets=vocab_info.num_oov_buckets,
          prev_tensor_name=prev_var_name,
          initializer=vocab_info.backup_initializer,
          axis=vocab_info.axis)
    else:
      # For the special value of vars_to_warm_start = None,
      # we only warm-start variables with explicitly specified vocabularies.
      if vars_to_warm_start:
        warmstarted_count += 1
        logging.debug("Warm-starting variable: {}; prev_var_name: {}".format(
            var_name, prev_var_name or "Unchanged"))
        # Because we use a default empty list in grouped_variables, single
        # unpartitioned variables will be lists here, which we rectify in order
        # for init_from_checkpoint logic to work correctly.
        if len(variable) == 1:
          variable = variable[0]
        prev_tensor_name, var = _get_var_info(variable, prev_var_name)
        vocabless_vars[prev_tensor_name] = var

  checkpoint_utils.init_from_checkpoint(ckpt_to_initialize_from, vocabless_vars)
  prev_var_name_not_used = set(
      var_name_to_prev_var_name.keys()) - prev_var_name_used
  vocab_info_not_used = set(var_name_to_vocab_info.keys()) - vocab_info_used

  logging.info("Warm-started %d variables.", warmstarted_count)

  if prev_var_name_not_used:
    raise ValueError(
        "You provided the following variables in "
        "var_name_to_prev_var_name that were not used: "
        "{0}.  Perhaps you misspelled them?  Here is the list of viable "
        "variable names: {1}".format(prev_var_name_not_used,
                                     grouped_variables.keys()))
  if vocab_info_not_used:
    raise ValueError(
        "You provided the following variables in "
        "var_name_to_vocab_info that were not used: {0}. "
        " Perhaps you misspelled them?  Here is the list of viable variable "
        "names: {1}".format(vocab_info_not_used, grouped_variables.keys()))
