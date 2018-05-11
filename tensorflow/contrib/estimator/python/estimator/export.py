# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Wrapper for methods to export train/eval graphs from Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import model_fn as model_fn_lib


def export_saved_model_for_mode(
    estimator, export_dir_base, input_receiver_fn,
    assets_extra=None,
    as_text=False,
    checkpoint_path=None,
    strip_default_attrs=False,
    mode=model_fn_lib.ModeKeys.PREDICT):
  # pylint: disable=line-too-long
  """Exports a single train/eval/predict graph as a SavedModel.

  For a detailed guide, see
  @{$saved_model#using_savedmodel_with_estimators$Using SavedModel with Estimators}.

  Sample usage:
  ```python
  classifier = tf.estimator.LinearClassifier(
      feature_columns=[age, language])
  classifier.train(input_fn=input_fn, steps=1000)

  feature_spec = {
      'age': tf.placeholder(dtype=tf.int64),
      'language': array_ops.placeholder(dtype=tf.string)
  }
  label_spec = tf.placeholder(dtype=dtypes.int64)

  train_rcvr_fn = tf.contrib.estimator.build_raw_supervised_input_receiver_fn(
      feature_spec, label_spec)

  export_dir = tf.contrib.estimator.export_saved_model_for_mode(
      classifier,
      export_dir_base='my_model/',
      input_receiver_fn=train_rcvr_fn,
      mode=model_fn_lib.ModeKeys.TRAIN)

  # export_dir is a timestamped directory with the SavedModel, which
  # can be used for serving, analysis with TFMA, or directly loaded in.
  with ops.Graph().as_default() as graph:
    with session.Session(graph=graph) as sess:
      loader.load(sess, [tag_constants.TRAINING], export_dir)
      ...
  ```

  This method takes an input_receiver_fn and mode. For the mode passed in,
  this method builds a new graph by calling the input_receiver_fn to obtain
  feature and label `Tensor`s. Next, this method calls the `Estimator`'s
  model_fn in the passed mode to generate the model graph based on
  those features and labels, and restores the given checkpoint
  (or, lacking that, the most recent checkpoint) into the graph.
  Finally, it creates a timestamped export directory below the
  export_dir_base, and writes a `SavedModel` into it containing
  the `MetaGraphDef` for the given mode and its associated signatures.

  For prediction, the exported `MetaGraphDef` will provide one `SignatureDef`
  for each element of the export_outputs dict returned from the model_fn,
  named using the same keys.  One of these keys is always
  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY, indicating which
  signature will be served when a serving request does not specify one.
  For each signature, the outputs are provided by the corresponding
  `ExportOutput`s, and the inputs are always the input receivers provided by
  the serving_input_receiver_fn.

  For training and evaluation, the train_op is stored in an extra collection,
  and loss, metrics, and predictions are included in a SignatureDef for the
  mode in question.

  Extra assets may be written into the SavedModel via the assets_extra
  argument.  This should be a dict, where each key gives a destination path
  (including the filename) relative to the assets.extra directory.  The
  corresponding value gives the full path of the source file to be copied.
  For example, the simple case of copying a single file without renaming it
  is specified as `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.

  Args:
    estimator: an instance of tf.estimator.Estimator
    export_dir_base: A string containing a directory in which to create
      timestamped subdirectories containing exported SavedModels.
    input_receiver_fn: a function that takes no argument and
      returns the appropriate subclass of `InputReceiver`.
    assets_extra: A dict specifying how to populate the assets.extra directory
      within the exported SavedModel, or `None` if no extra assets are needed.
    as_text: whether to write the SavedModel proto in text format.
    checkpoint_path: The checkpoint path to export.  If `None` (the default),
      the most recent checkpoint found within the model directory is chosen.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
      removed from the NodeDefs. For a detailed guide, see
      [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
    mode: tf.estimator.ModeKeys value indicating with mode will be exported.

  Returns:
    The string path to the exported directory.

  Raises:
    ValueError: if input_receiver_fn is None, no export_outputs
      are provided, or no checkpoint can be found.
  """
  # pylint: enable=line-too-long

  # pylint: disable=protected-access
  return estimator._export_saved_model_for_mode(
      export_dir_base, input_receiver_fn,
      assets_extra=assets_extra,
      as_text=as_text,
      checkpoint_path=checkpoint_path,
      strip_default_attrs=strip_default_attrs,
      mode=mode)
  # pylint: enable=protected-access


def export_all_saved_models(
    estimator, export_dir_base, input_receiver_fn_map,
    assets_extra=None,
    as_text=False,
    checkpoint_path=None,
    strip_default_attrs=False):
  # pylint: disable=line-too-long
  """Exports requested train/eval/predict graphs as separate SavedModels.

  This is a wrapper around export_saved_model_for_mode that accepts
  multiple modes simultaneously and creates directories for each under
  export_dir_base. See `Estimator.export_saved_model_for_mode` for
  further details as to how the export works for each mode.

  Sample usage:
  ```python
  classifier = tf.estimator.LinearClassifier(
      feature_columns=[age, language])
  classifier.train(input_fn=input_fn)

  feature_spec = {
      'age': tf.placeholder(dtype=tf.int64),
      'language': array_ops.placeholder(dtype=tf.string)
  }
  label_spec = tf.placeholder(dtype=dtypes.int64)

  train_rcvr_fn = tf.contrib.estimator.build_raw_supervised_input_receiver_fn(
      feature_spec, label_spec)

  serve_rcvr_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      feature_spec)

  rcvr_fn_map = {
      model_fn_lib.ModeKeys.TRAIN: train_rcvr_fn,
      model_fn_lib.ModeKeys.PREDICT: serve_rcvr_fn,
  }

  export_dirs = tf.contrib.estimator.export_all_saved_models(
      classifier,
      export_dir_base='my_model/',
      input_receiver_fn_map=rcvr_fn_map)

  # export_dirs is a dict of directories with SavedModels, which
  # can be used for serving, analysis with TFMA, or directly loaded in.
  with ops.Graph().as_default() as graph:
    with session.Session(graph=graph) as sess:
      loader.load(sess, [tag_constants.TRAINING],
                  export_dirs[tf.estimator.ModeKeys.TRAIN])
      ...
  ```

  Args:
    estimator: an instance of tf.estimator.Estimator
    export_dir_base: A string containing a directory in which to create
      timestamped subdirectories containing exported SavedModels.
    input_receiver_fn_map: dict of tf.estimator.ModeKeys to input_receiver_fn
      mappings, where the input_receiver_fn is a function that takes no
      argument and returns the appropriate subclass of `InputReceiver`.
    assets_extra: A dict specifying how to populate the assets.extra directory
      within the exported SavedModel, or `None` if no extra assets are needed.
    as_text: whether to write the SavedModel proto in text format.
    checkpoint_path: The checkpoint path to export.  If `None` (the default),
      the most recent checkpoint found within the model directory is chosen.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
      removed from the NodeDefs. For a detailed guide, see
      [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

  Returns:
    A dict of tf.estimator.ModeKeys value to string path for each exported
    directory.

  Raises:
    ValueError: if any input_receiver_fn is None, no export_outputs
      are provided, or no checkpoint can be found.
  """
  # pylint: enable=line-too-long

  # pylint: disable=protected-access
  return estimator._export_all_saved_models(
      export_dir_base, input_receiver_fn_map,
      assets_extra=assets_extra,
      as_text=as_text,
      checkpoint_path=checkpoint_path,
      strip_default_attrs=strip_default_attrs)
  # pylint: enable=protected-access
