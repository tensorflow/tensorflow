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
# ========================================================================
"""Utilities to handle tensor tracer parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import os.path
import re

from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

TRACE_MODE_PART_TENSOR = 'part-tensor'
TRACE_MODE_FULL_TENSOR = 'full-tensor'
TRACE_MODE_FULL_TENSOR_SUMMARY = 'full_tensor_summary'

TRACE_MODE_NAN_INF = 'nan-inf'
TRACE_MODE_NORM = 'norm'
TRACE_MODE_MAX_ABS = 'max-abs'
TRACE_MODE_SUMMARY = 'summary'
# summary mode to collects a finite set of signatures for each traced tensor,
# (such as norm, max, min, mean) and dumps it using tb summaries.

# Full tensor mode dumps the whole tensor values for the traced tensors without
# any processing on them; using tb summaries.

_SUBMODE_BRIEF = 'brief'
_SUBMODE_DETAILED = 'detailed'

_FLAG_SINGLE_QUOTE_PAT = re.compile(r"\s*--([^=]+)='([^']*)'")
_FLAG_DOUBLE_QUOTE_PAT = re.compile(r'\s*--([^=]+)="([^"]*)"')
_FLAG_NO_QUOTE_PAT = re.compile(r'\s*--([^=]+)=(\S*)')
_FLAG_NO_EQUAL_PAT = re.compile(r'\s*--([^=]+)\s*')

FLAGS_ENV_VAR = 'TENSOR_TRACER_FLAGS'
FLAG_NAME_ENABLE = 'enable'
FLAG_NAME_TRACE_MODE = 'trace_mode'
FLAG_NAME_TRACE_SCALAR_OPS = 'trace_scalar'
FLAG_NAME_SUBMODE = 'submode'
FLAG_NAME_EXCLUDED_OPNAMES = 'excluded_opnames'
FLAG_NAME_EXCLUDED_OPTYPES = 'excluded_optypes'
FLAG_NAME_INCLUDED_OPNAMES = 'included_opnames'
FLAG_NAME_INCLUDED_OPTYPES = 'included_optypes'
FLAG_NAME_TRACE_LEVEL = 'trace_level'
FLAG_NAME_TRACE_DIR = 'trace_dir'
FLAG_NAME_REPORT_FILE = 'report_file'
FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR = 'use_test_undeclared_outputs_dir'
FLAG_NAME_OP_RANGE = 'op_range'
# Folder to dump the pre (before tensor tracer updates) and post graphs (after
# tensor tracer updates).
FLAG_NAME_DUMP_BEFORE_AFTER_GRAPHS = 'dump_graphs'
FLAG_NAME_SUMMARY_SIGNATURES = 'signatures'
FLAG_NAME_SUMMARY_PER_CORE = 'collect_summary_per_core'
FLAG_NAME_TEMP_CACHE_VAR = 'use_temp_cache'
FLAG_NAME_INSPECT_TRACE = 'inspect_trace'
FLAG_NAME_FINGERPRINT_DIR = 'use_fingerprint_subdirectory'
FLAG_FLUSH_SUMMARY = 'flush_summaries'

# Flag used in v2 only.
FLAG_SUMMARY_MODE_TYPE = 'summary_mode'
UI_MODE = 'ui'
TEXT_MODE = 'text'

_OP_RANGE_PAT = re.compile(r'(\d+):(\d+)')
_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR = 'TEST_UNDECLARED_OUTPUTS_DIR'

_TT_DEFAULT_TRACE_LEVEL = 3
_TT_PREFIX = 'tensor_tracer'

_TT_NORM = 'norm'
_TT_MAX = 'max'
_TT_MAX_ABS = 'max-abs'
_TT_MIN = 'min'
_TT_MEAN = 'mean'
_TT_VAR = 'var'
_TT_SIZE = 'size'

TT_SUMMARY_NORM = '%s_%s' % (_TT_PREFIX, _TT_NORM)
TT_SUMMARY_MAX = '%s_%s' % (_TT_PREFIX, _TT_MAX)
TT_SUMMARY_MAX_ABS = '%s_%s' % (_TT_PREFIX, _TT_MAX_ABS)
TT_SUMMARY_MIN = '%s_%s' % (_TT_PREFIX, _TT_MIN)
TT_SUMMARY_MEAN = '%s_%s' % (_TT_PREFIX, _TT_MEAN)
TT_SUMMARY_VAR = '%s_%s' % (_TT_PREFIX, _TT_VAR)
TT_SUMMARY_SIZE = '%s_%s' % (_TT_PREFIX, _TT_SIZE)

TT_SUMMARY_SIGNATURES = (TT_SUMMARY_NORM, TT_SUMMARY_MAX, TT_SUMMARY_MIN,
                         TT_SUMMARY_MEAN, TT_SUMMARY_VAR, TT_SUMMARY_SIZE,
                         TT_SUMMARY_MAX_ABS)


class TTParameters(object):
  """A class that handles the parameters of Tensor Tracer."""

  def __init__(self, env=None):
    if env:
      self._env = env
    else:
      self._env = os.environ
    self._validate_flag_names()
    self.trace_mode = self._get_trace_mode()
    self.submode = self._get_submode()
    self.trace_dir = self._get_trace_dir()
    self.report_file_path = self._get_report_filepath()
    self.op_range = self._get_op_range()
    self.excluded_opname_re_list = self._flag_value_to_re_list(
        FLAG_NAME_EXCLUDED_OPNAMES)
    self.excluded_optype_re_list = self._flag_value_to_re_list(
        FLAG_NAME_EXCLUDED_OPTYPES)

    self.included_opname_re_list = self._flag_value_to_re_list(
        FLAG_NAME_INCLUDED_OPNAMES)
    self.included_optype_re_list = self._flag_value_to_re_list(
        FLAG_NAME_INCLUDED_OPTYPES)

    self.trace_scalar_ops = self.is_flag_on(FLAG_NAME_TRACE_SCALAR_OPS)
    self.use_compact_trace = self.trace_mode in (TRACE_MODE_NAN_INF,
                                                 TRACE_MODE_NORM,
                                                 TRACE_MODE_MAX_ABS,
                                                 TRACE_MODE_SUMMARY)
    self.use_temp_cache_var = self.is_flag_on(FLAG_NAME_TEMP_CACHE_VAR)
    self.inspect_trace = self.is_flag_on(FLAG_NAME_INSPECT_TRACE)
    self.use_fingerprint_subdir = self.is_flag_on(FLAG_NAME_FINGERPRINT_DIR)

    _, self.graph_dump_path = self.get_flag_value(
        FLAG_NAME_DUMP_BEFORE_AFTER_GRAPHS)
    self.trace_level = self._get_flag_int_value(FLAG_NAME_TRACE_LEVEL,
                                                _TT_DEFAULT_TRACE_LEVEL)
    self.summary_signatures = self._get_summary_signatures()
    self.collect_summary_per_core = self.is_flag_on(FLAG_NAME_SUMMARY_PER_CORE)
    self.flush_summaries_with_outside_compile = self.is_flag_on(
        FLAG_FLUSH_SUMMARY)
    self.summary_mode = self._get_summary_mode()
    self._check_flag_errors()

  def _check_flag_errors(self):
    if self.trace_mode in (TRACE_MODE_SUMMARY, TRACE_MODE_FULL_TENSOR_SUMMARY):
      if not self.trace_dir:
        raise ValueError('trace_dir must be explicitly provided in '
                         'TENSOR_TRACER_FLAGS when summary mode is used.')

  def _get_report_filepath(self):
    """Sets the path of the output report file."""

    found, report_file_path = self.get_flag_value(FLAG_NAME_REPORT_FILE)
    if found and report_file_path and self.use_test_undeclared_outputs_dir():
      if os.path.isabs(report_file_path):
        raise ValueError('If use_test_undeclared_outputs_dir is set,'
                         'report_file_path cannot be an absolute path (%s)'
                         %report_file_path)
      outputs_dir = self._env.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
      report_file_path = os.path.join(outputs_dir, report_file_path)
    return report_file_path

  def _get_op_range(self):
    """Sets the index range of the Ops that we will consider tracing."""
    found, op_range = self.get_flag_value(FLAG_NAME_OP_RANGE)
    if not found or not op_range:
      op_range = (-1, -1)  # this means including all ops.
      return op_range
    match = _OP_RANGE_PAT.match(op_range)
    if not match:
      op_range = (-1, -1)  # this means including all ops.
      return op_range
    op_range = (int(match.group(1)), int(match.group(2)))
    return op_range

  def _get_trace_dir(self):
    found, trace_dir = self.get_flag_value(FLAG_NAME_TRACE_DIR)
    if found and trace_dir and self.use_test_undeclared_outputs_dir():
      raise ValueError(
          'Cannot not use --%s and --%s at the same time' %
          (FLAG_NAME_TRACE_DIR, FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR))
    if self.use_test_undeclared_outputs_dir():
      trace_dir = self._env.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
    return trace_dir

  def _get_trace_mode(self):
    """Checks if the given trace mode is valid."""

    found, trace_mode = self.get_flag_value(FLAG_NAME_TRACE_MODE)
    if not found or not trace_mode:
      trace_mode = TRACE_MODE_NORM
    valid_trace_modes = [
        TRACE_MODE_NAN_INF, TRACE_MODE_PART_TENSOR, TRACE_MODE_FULL_TENSOR,
        TRACE_MODE_NORM, TRACE_MODE_MAX_ABS,
        TRACE_MODE_SUMMARY, TRACE_MODE_FULL_TENSOR_SUMMARY
    ]
    if trace_mode not in valid_trace_modes:
      raise ValueError('Invalid trace mode "%s" given to the Tensor_Tracer.'
                       'Valid trace modes are: %s'%(trace_mode,
                                                    valid_trace_modes))
    return trace_mode

  def is_brief_mode(self):
    return self.submode == _SUBMODE_BRIEF

  def _get_submode(self):
    """Checks if the given submode is valid."""

    found, submode = self.get_flag_value(FLAG_NAME_SUBMODE)
    if not found or not submode:
      submode = _SUBMODE_DETAILED
    if not submode:
      return
    valid_submodes = [_SUBMODE_DETAILED, _SUBMODE_BRIEF]
    if submode not in valid_submodes:
      raise ValueError('Invalid submode "%s" given to the Tensor_Tracer.'
                       'Valid submodes are: %s'%(submode,
                                                 valid_submodes))
    return submode

  @staticmethod
  def match_next_flag(flags, pos):
    """Returns the match for the next TensorTracer flag.

    Args:
       flags: a string that contains the flags.
       pos: where in flags to start the search.

    Returns:
       A pair where the first element is the regular-expression
       match found and the second element indicates if the match
       has a value.
    """

    match = _FLAG_DOUBLE_QUOTE_PAT.match(flags, pos)
    if match:
      return match, True
    match = _FLAG_SINGLE_QUOTE_PAT.match(flags, pos)
    if match:
      return match, True
    match = _FLAG_NO_QUOTE_PAT.match(flags, pos)
    if match:
      return match, True
    match = _FLAG_NO_EQUAL_PAT.match(flags, pos)
    if match:
      # The flag is found but is not given a value.
      return match, False
    # The flag is not found.
    return None, False

  def _validate_flag_names(self):
    """Validates if the TensorTrace flags passed are valid."""
    valid_flag_names = [
        FLAG_NAME_ENABLE, FLAG_NAME_TRACE_MODE,
        FLAG_NAME_TRACE_SCALAR_OPS,
        FLAG_NAME_SUBMODE, FLAG_NAME_EXCLUDED_OPNAMES,
        FLAG_NAME_EXCLUDED_OPTYPES, FLAG_NAME_INCLUDED_OPNAMES,
        FLAG_NAME_INCLUDED_OPTYPES, FLAG_NAME_TRACE_DIR,
        FLAG_NAME_REPORT_FILE,
        FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR,
        FLAG_NAME_OP_RANGE,
        FLAG_NAME_DUMP_BEFORE_AFTER_GRAPHS, FLAG_NAME_TRACE_LEVEL,
        FLAG_NAME_SUMMARY_SIGNATURES, FLAG_NAME_SUMMARY_PER_CORE,
        FLAG_NAME_TEMP_CACHE_VAR, FLAG_NAME_FINGERPRINT_DIR,
        FLAG_NAME_INSPECT_TRACE, FLAG_FLUSH_SUMMARY, FLAG_SUMMARY_MODE_TYPE
    ]
    tensor_tracer_flags = self._env.get(FLAGS_ENV_VAR)
    if not tensor_tracer_flags:
      return
    pos = 0
    while True:
      match, _ = TTParameters.match_next_flag(tensor_tracer_flags, pos)
      if not match:
        break
      flag_name = match.group(1)
      if flag_name not in valid_flag_names:
        raise ValueError(
            'The flag name "%s" passed via the environment variable "%s" '
            'is invalid. Valid flag names are:'
            '\n%s' % (flag_name, FLAGS_ENV_VAR, valid_flag_names))
      pos = match.end()

  def _get_summary_signatures(self):
    """Verifies and returns the summary signatures.

    Returns:
      A dictionary of the signature identifiers {signature: index} that will be
      computed when trace_mode is summary.
    """
    signatures = self._flag_value_as_list(FLAG_NAME_SUMMARY_SIGNATURES)

    tt_signatures = []
    for signature in signatures:
      signature_with_prefix = '%s_%s' % (_TT_PREFIX, signature)
      if signature in TT_SUMMARY_SIGNATURES:
        tt_signatures.append(signature)
      elif signature_with_prefix in TT_SUMMARY_SIGNATURES:
        tt_signatures.append(signature_with_prefix)
      else:
        logging.warning('Unknown signature:%s. Supported signatures: %s' % (
            signature, TT_SUMMARY_SIGNATURES))
    if not tt_signatures:
      # Default case collects norm and max only.
      return {TT_SUMMARY_MAX_ABS: 0, TT_SUMMARY_NORM: 1}
    else:
      return {signature: idx for idx, signature in enumerate(tt_signatures)}

  def get_signature_to_agg_fn_map(self):
    """Returns a map that contains the aggregate function for each signature."""
    return {TRACE_MODE_NORM: linalg_ops.norm,
            TRACE_MODE_MAX_ABS: math_ops.reduce_max,
            TRACE_MODE_NAN_INF: math_ops.reduce_max,
            TT_SUMMARY_NORM: linalg_ops.norm,
            TT_SUMMARY_MAX: math_ops.reduce_max,
            TT_SUMMARY_MAX_ABS:
                lambda t, axis=0: math_ops.reduce_max(math_ops.abs(t),  # pylint: disable=g-long-lambda
                                                      axis=axis),
            TT_SUMMARY_MIN: math_ops.reduce_min,
            TT_SUMMARY_MEAN: math_ops.reduce_mean,
            TT_SUMMARY_VAR: math_ops.reduce_max,  # Simply reduce max variance.
            TT_SUMMARY_SIZE: math_ops.reduce_sum}

  def _flag_value_as_list(self, wanted_flag_name):
    """Returns the string list of a TensorTracer flag.

    Args:
      wanted_flag_name: the name of the flag we are looking for.

    Returns:
      The list value of the flag.
    """
    string_value_list = []
    found, flag_value = self.get_flag_value(wanted_flag_name)

    if found:
      string_value_list = flag_value.split(',')
    return string_value_list

  def _flag_value_as_int_list(self, wanted_flag_name):
    """Returns the integer list of a TensorTracer flag.

    Args:
      wanted_flag_name: the name of the flag we are looking for.

    Returns:
      the value of the flag.
    Raises:
      RuntimeError: If supposedly deadcode is reached.
    """
    int_list = []
    found, flag_value = self.get_flag_value(wanted_flag_name)

    if found and flag_value:
      try:
        integer_values = flag_value.split(',')
        int_list = [int(int_val) for int_val in integer_values]
      except ValueError:
        logging.warning('Cannot convert %s to int for flag %s', int_list,
                        wanted_flag_name)
    return int_list

  def _get_flag_int_value(self, wanted_flag_name, default_value):
    """Returns the int value of a TensorTracer flag.

    Args:
      wanted_flag_name: the name of the flag we are looking for.
      default_value: the default value for the flag, if not provided.
    Returns:
      the value of the flag.
    Raises:
      RuntimeError: If supposedly deadcode is reached.
    """
    flag_int_value = default_value
    found, flag_value = self.get_flag_value(wanted_flag_name)

    if found:
      try:
        flag_int_value = int(flag_value)
      except ValueError:
        logging.warning('Cannot convert %s to int for flag %s' % (
            flag_int_value, wanted_flag_name))
    return flag_int_value

  def get_flag_value(self, wanted_flag_name):
    """Returns the value of a TensorTracer flags.

    Args:
      wanted_flag_name: the name of the flag we are looking for.

    Returns:
      A pair where the first element indicates if the flag is
      found and the second element is the value of the flag.

    Raises:
      RuntimeError: If supposedly deadcode is reached.
    """

    tensor_tracer_flags = self._env.get(FLAGS_ENV_VAR)
    if not tensor_tracer_flags:
      return False, None
    pos = 0
    while True:
      match, has_value = TTParameters.match_next_flag(
          tensor_tracer_flags, pos)
      if not match:
        return False, None
      flag_name = match.group(1)
      if has_value:
        flag_value = match.group(2)
      else:
        flag_value = None
      if flag_name == wanted_flag_name:
        return True, flag_value
      pos = match.end()
    raise RuntimeError('Should not reach here.')

  def _flag_value_to_re_list(self, flag_name):
    """Converts list of strings to compiled RE."""

    re_list = []
    found, flag_value = self.get_flag_value(flag_name)
    if not found or not flag_value:
      return re_list
    list_of_values = flag_value.split(',')
    for v in list_of_values:
      r = re.compile(v)
      re_list.append(r)
    return re_list

  def is_flag_on(self, flag_name):
    """Returns True if the given flag is on."""

    found, flag_value = self.get_flag_value(flag_name)
    if not found:
      return False
    if flag_value is None:
      return True
    # Depends on the flag value.
    flag_value = flag_value.lower()
    enabled = flag_value in ['1', 't', 'true', 'y', 'yes']
    return enabled

  def is_enabled(self):
    """Returns True if TensorTracer is enabled."""

    if self.is_flag_on(FLAG_NAME_ENABLE):
      logging.debug('Tensor Tracer is enabled with flags %s.',
                    self._env.get(FLAGS_ENV_VAR))
      return True
    else:
      return False

  def use_test_undeclared_outputs_dir(self):
    """Decides the output directory of the report and trace files.

    Args:
       None.

    Returns:
       True if the output files should be written to the
       test-undeclared-outputs-directory defined via an
       env variable.
    """

    return self.is_flag_on(FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR)

  def _get_summary_mode(self):
    """Returns the summary mode after checking if it is valid."""

    found, summary_mode = self.get_flag_value(FLAG_SUMMARY_MODE_TYPE)
    if not found:
      summary_mode = UI_MODE

    valid_summary_modes = [UI_MODE, TEXT_MODE]
    if summary_mode not in valid_summary_modes:
      raise ValueError('Invalid summary mode "%s" given to the Tensor_Tracer.'
                       'Valid submodes are: %s'%(summary_mode,
                                                 valid_summary_modes))
    return summary_mode
