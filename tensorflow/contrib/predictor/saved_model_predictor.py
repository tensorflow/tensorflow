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

"""A `Predictor` constructed from a `SavedModel`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from tensorflow.contrib.predictor import predictor
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants


DEFAULT_TAGS = 'serve'

_DEFAULT_INPUT_ALTERNATIVE_FORMAT = 'default_input_alternative:{}'


def get_meta_graph_def(saved_model_dir, tags):
  """Gets `MetaGraphDef` from a directory containing a `SavedModel`.

  Returns the `MetaGraphDef` for the given tag-set and SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel.
    tags: Comma separated list of tags used to identify the correct
      `MetaGraphDef`.

  Raises:
    ValueError: An error when the given tags cannot be found.

  Returns:
    A `MetaGraphDef` corresponding to the given tags.
  """
  saved_model = reader.read_saved_model(saved_model_dir)
  set_of_tags = set([tag.strip() for tag in tags.split(',')])
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
      return meta_graph_def
  raise ValueError('Could not find MetaGraphDef with tags {}'.format(tags))


def _get_signature_def(signature_def_key, export_dir, tags):
  """Construct a `SignatureDef` proto."""
  signature_def_key = (
      signature_def_key or
      signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

  metagraph_def = get_meta_graph_def(export_dir, tags)

  try:
    signature_def = metagraph_def.signature_def[signature_def_key]
  except KeyError as e:
    formatted_key = _DEFAULT_INPUT_ALTERNATIVE_FORMAT.format(
        signature_def_key)
    try:
      signature_def = metagraph_def.signature_def[formatted_key]
    except KeyError:
      raise ValueError(
          'Got signature_def_key "{}". Available signatures are {}. '
          'Original error:\n{}'.format(
              signature_def_key, list(metagraph_def.signature_def), e))
    logging.warning('Could not find signature def "%s". '
                    'Using "%s" instead', signature_def_key, formatted_key)
  return signature_def


def _check_signature_arguments(signature_def_key,
                               signature_def,
                               input_names,
                               output_names):
  """Validates signature arguments for `SavedModelPredictor`."""
  signature_def_key_specified = signature_def_key is not None
  signature_def_specified = signature_def is not None
  input_names_specified = input_names is not None
  output_names_specified = output_names is not None
  if input_names_specified != output_names_specified:
    raise ValueError(
        'input_names and output_names must both be specified or both be '
        'unspecified.'
    )

  if (signature_def_key_specified + signature_def_specified +
      input_names_specified > 1):
    raise ValueError(
        'You must specify at most one of signature_def_key OR signature_def OR'
        '(input_names AND output_names).'
    )


class SavedModelPredictor(predictor.Predictor):
  """A `Predictor` constructed from a `SavedModel`."""

  def __init__(self,
               export_dir,
               signature_def_key=None,
               signature_def=None,
               input_names=None,
               output_names=None,
               tags=None,
               graph=None,
               config=None):
    """Initialize a `CoreEstimatorPredictor`.

    Args:
      export_dir: a path to a directory containing a `SavedModel`.
      signature_def_key: Optional string specifying the signature to use. If
        `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used. Only one of
        `signature_def_key` and `signature_def` should be specified.
      signature_def: A `SignatureDef` proto specifying the inputs and outputs
        for prediction. Only one of `signature_def_key` and `signature_def`
        should be specified.
      input_names: A dictionary mapping strings to `Tensor`s in the `SavedModel`
        that represent the input. The keys can be any string of the user's
        choosing.
      output_names: A dictionary mapping strings to `Tensor`s in the
        `SavedModel` that represent the output. The keys can be any string of
        the user's choosing.
      tags: Optional. Comma separated list of tags that will be used to retrieve
        the correct `SignatureDef`. Defaults to `DEFAULT_TAGS`.
      graph: Optional. The Tensorflow `graph` in which prediction should be
        done.
      config: `ConfigProto` proto used to configure the session.
    Raises:
      ValueError: If more than one of signature_def_key OR signature_def OR
        (input_names AND output_names) is specified.
    """
    _check_signature_arguments(
        signature_def_key, signature_def, input_names, output_names)
    tags = tags or DEFAULT_TAGS
    self._graph = graph or ops.Graph()

    with self._graph.as_default():
      self._session = session.Session(config=config)
      loader.load(self._session, tags.split(','), export_dir)

    if input_names is None:
      if signature_def is None:
        signature_def = _get_signature_def(signature_def_key, export_dir, tags)
      input_names = {k: v.name for k, v in signature_def.inputs.items()}
      output_names = {k: v.name for k, v in signature_def.outputs.items()}

    self._feed_tensors = {k: self._graph.get_tensor_by_name(v)
                          for k, v in input_names.items()}
    self._fetch_tensors = {k: self._graph.get_tensor_by_name(v)
                           for k, v in output_names.items()}
