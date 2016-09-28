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
"""Format tensors (ndarrays) for screen display and navigation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.debug.cli import debugger_cli_common

_NUMPY_OMISSION = "...,"
_NUMPY_DEFAULT_EDGE_ITEMS = 3

BEGIN_INDICES_KEY = "i0"
OMITTED_INDICES_KEY = "omitted"


def format_tensor(
    tensor, tensor_name, include_metadata=False, np_printoptions=None):
  """Generate a RichTextLines object showing a tensor in formatted style.

  Args:
    tensor: The tensor to be displayed, as a numpy ndarray or other
      appropriate format (e.g., None representing uninitialized tensors).
    tensor_name: Name of the tensor, as a str. If set to None, will suppress
      the tensor name line in the return value.
    include_metadata: Whether metadata such as dtype and shape are to be
      included in the formatted text.
    np_printoptions: A dictionary of keyword arguments that are passed to a
      call of np.set_printoptions() to set the text format for display numpy
      ndarrays.

  Returns:
    A RichTextLines object. Its annotation field has line-by-line markups to
    indicate which indices in the array the first element of each line
    corresponds to.
  """
  lines = []

  if tensor_name is not None:
    lines.append("Tensor \"%s\":" % tensor_name)

  if not isinstance(tensor, np.ndarray):
    # If tensor is not a np.ndarray, return simple text-line representation of
    # the object without annotations.
    if lines:
      lines.append("")
    lines.extend(repr(tensor).split("\n"))
    return debugger_cli_common.RichTextLines(lines)

  if include_metadata:
    lines.append("  dtype: %s" % str(tensor.dtype))
    lines.append("  shape: %s" % str(tensor.shape))

  if lines:
    lines.append("")
  hlines = len(lines)

  # Apply custom string formatting options for numpy ndarray.
  if np_printoptions is not None:
    np.set_printoptions(**np_printoptions)

  array_lines = repr(tensor).split("\n")
  lines.extend(array_lines)

  if tensor.dtype.type is not np.string_:
    # Parse array lines to get beginning indices for each line.

    # TODO(cais): Currently, we do not annotate string-type tensors due to
    #   difficulty in escaping sequences. Address this issue.
    annotations = _annotate_ndarray_lines(
        array_lines, tensor, np_printoptions=np_printoptions, offset=hlines)

  return debugger_cli_common.RichTextLines(lines, annotations=annotations)


def _annotate_ndarray_lines(
    array_lines, tensor, np_printoptions=None, offset=0):
  """Generate annotations for line-by-line begin indices of tensor text.

  Parse the numpy-generated text representation of a numpy ndarray to
  determine the indices of the first element of each text line (if any
  element is present in the line).

  For example, given the following multi-line ndarray text representation:
      ["array([[ 0.    ,  0.0625,  0.125 ,  0.1875],",
       "       [ 0.25  ,  0.3125,  0.375 ,  0.4375],",
       "       [ 0.5   ,  0.5625,  0.625 ,  0.6875],",
       "       [ 0.75  ,  0.8125,  0.875 ,  0.9375]])"]
  the generate annotation will be:
      {0: {BEGIN_INDICES_KEY: [0, 0]},
       1: {BEGIN_INDICES_KEY: [1, 0]},
       2: {BEGIN_INDICES_KEY: [2, 0]},
       3: {BEGIN_INDICES_KEY: [3, 0]}}

  Args:
    array_lines: Text lines representing the tensor, as a list of str.
    tensor: The tensor being formatted as string.
    np_printoptions: A dictionary of keyword arguments that are passed to a
      call of np.set_printoptions().
    offset: Line number offset applied to the line indices in the returned
      annotation.

  Returns:
    An annotation as a dict.
  """

  if np_printoptions and "edgeitems" in np_printoptions:
    edge_items = np_printoptions["edgeitems"]
  else:
    edge_items = _NUMPY_DEFAULT_EDGE_ITEMS

  annotations = {}

  # Put metadata about the tensor in the annotations["tensor_metadata"].
  annotations["tensor_metadata"] = {
      "dtype": tensor.dtype, "shape": tensor.shape}

  dims = np.shape(tensor)
  ndims = len(dims)
  curr_indices = [0] * len(dims)
  curr_dim = 0
  for i in xrange(len(array_lines)):
    line = array_lines[i].strip()

    if not line:
      # Skip empty lines, which can appear for >= 3D arrays.
      continue

    if line == _NUMPY_OMISSION:
      annotations[offset + i] = {OMITTED_INDICES_KEY: copy.copy(curr_indices)}
      curr_indices[curr_dim - 1] = dims[curr_dim - 1] - edge_items
    else:
      num_lbrackets = line.count("[")  # TODO(cais): String array escaping.
      num_rbrackets = line.count("]")

      curr_dim += num_lbrackets - num_rbrackets

      annotations[offset + i] = {BEGIN_INDICES_KEY: copy.copy(curr_indices)}
      if num_rbrackets == 0:
        line_content = line[line.rfind("[") + 1:]
        num_elements = line_content.count(",")
        curr_indices[curr_dim - 1] += num_elements
      else:
        if curr_dim > 0:
          curr_indices[curr_dim - 1] += 1
          for k in xrange(curr_dim, ndims):
            curr_indices[k] = 0

  return annotations


def locate_tensor_element(formatted, indices):
  """Locate a tensor element in formatted text lines, given element indices.

  Given a RichTextLines object representing a tensor and indices of the sought
  element, return the row number at which the element is located (if exists).

  TODO(cais): Return column number as well.

  Args:
    formatted: A RichTextLines object containing formatted text lines
      representing the tensor.
    indices: Indices of the sought element, as a list of int.

  Returns:
    1) A boolean indicating whether the element falls into an omitted line.
    2) Row index.

  Raises:
    AttributeError: If:
      Input argument "formatted" does not have the required annotations.
    ValueError: If:
      1) Indices do not match the dimensions of the tensor, or
      2) Indices exceed sizes of the tensor, or
      3) Indices contain negative value(s).
  """

  # Check that tensor_metadata is available.
  if "tensor_metadata" not in formatted.annotations:
    raise AttributeError("tensor_metadata is not available in annotations.")

  # Check "indices" match tensor dimensions.
  dims = formatted.annotations["tensor_metadata"]["shape"]
  if len(indices) != len(dims):
    raise ValueError(
        "Dimensions mismatch: requested: %d; actual: %d" %
        (len(indices), len(dims)))

  # Check "indices" is within size limits.
  for req_idx, siz in zip(indices, dims):
    if req_idx >= siz:
      raise ValueError("Indices exceed tensor dimensions.")
    if req_idx < 0:
      raise ValueError("Indices contain negative value(s).")

  lines = formatted.lines
  annot = formatted.annotations
  prev_r = 0
  prev_indices = [0] * len(dims)
  for r in xrange(len(lines)):
    if r not in annot:
      continue

    if BEGIN_INDICES_KEY in annot[r]:
      if indices >= prev_indices and indices < annot[r][BEGIN_INDICES_KEY]:
        return OMITTED_INDICES_KEY in annot[prev_r], prev_r
      else:
        prev_r = r
        prev_indices = annot[r][BEGIN_INDICES_KEY]
    elif OMITTED_INDICES_KEY in annot[r]:
      if indices >= prev_indices and indices < annot[r][OMITTED_INDICES_KEY]:
        return OMITTED_INDICES_KEY in annot[prev_r], prev_r
      else:
        prev_r = r
        prev_indices = annot[r][OMITTED_INDICES_KEY]

  return OMITTED_INDICES_KEY in annot[prev_r], prev_r
