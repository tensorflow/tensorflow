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

TRACE_MODE_NAN_INF = 'nan-inf'
TRACE_MODE_PART_TENSOR = 'part-tensor'
TRACE_MODE_FULL_TENSOR = 'full-tensor'
TRACE_MODE_FULL_IF_NAN = 'trace-back-if-nan'
TRACE_MODE_NORM = 'norm'
TRACE_MODE_MAX_ABS = 'max-abs'
TRACE_MODE_SUMMARY = 'summary'
# summary mode to collects a finite set of signatures for each traced tensor,
# (such as norm, max, min, mean) and dumps it using tb summaries.
TRACE_MODE_FULL_TENSOR_SUMMARY = 'full_tensor_summary'
# Full tensor mode dumps the whole tensor values for the traced tensors without
# any processing on them; using tb summaries.
_FLAG_NAME_TRACE_STACK_SIZE = 'trace_stack_size'
_SUBMODE_BRIEF = 'brief'
_SUBMODE_DETAILED = 'detailed'
_FLAGS_ENV_VAR = 'TENSOR_TRACER_FLAGS'
_FLAG_SINGLE_QUOTE_PAT = re.compile(r"\s*--([^=]+)='([^']*)'")
_FLAG_DOUBLE_QUOTE_PAT = re.compile(r'\s*--([^=]+)="([^"]*)"')
_FLAG_NO_QUOTE_PAT = re.compile(r'\s*--([^=]+)=(\S*)')
_FLAG_NO_EQUAL_PAT = re.compile(r'\s*--([^=]+)\s*')
_FLAG_NAME_ENABLE = 'enable'
_FLAG_NAME_TRACE_MODE = 'trace_mode'
_FLAG_NAME_USE_COMPACT_TRACE = 'compact_trace'
_FLAG_NAME_TRACE_SCALAR_OPS = 'trace_scalar'
_FLAG_NAME_TRACE_BEFORE_OPS = 'trace_before_included_ops'
_FLAG_NAME_TRACE_AFTER_OPS = 'trace_after_included_ops'
_FLAG_NAME_SUBMODE = 'submode'
_FLAG_NAME_INCLUDE_LESS_INTERESTING_OPS = 'include_less_interesting_ops'
_FLAG_NAME_EXCLUDED_OPNAMES = 'excluded_opnames'
_FLAG_NAME_EXCLUDED_OPTYPES = 'excluded_optypes'
_FLAG_NAME_INCLUDED_OPNAMES = 'included_opnames'
_FLAG_NAME_INCLUDED_OPTYPES = 'included_optypes'
_FLAG_NAME_INCLUDED_CORES = 'included_cores'
_FLAG_NAME_TRACE_LEVEL = 'trace_level'
_FLAG_NAME_TRACE_DIR = 'trace_dir'
_FLAG_NAME_REPORT_FILE = 'report_file'
_FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR = 'use_test_undeclared_outputs_dir'
_FLAG_NAME_OP_RANGE = 'op_range'
# Folder to dump the pre (before tensor tracer updates) and post graphs (after
# tensor tracer updates).
_FLAG_DUMP_BEFORE_AFTER_GRAPHS = 'dump_graphs'
_OP_RANGE_PAT = re.compile(r'(\d+):(\d+)')
_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR = 'TEST_UNDECLARED_OUTPUTS_DIR'
_FLAG_SUMMARY_SIGNATURES = 'signatures'
_FLAG_NAME_SUMMARY_PER_CORE = 'collect_summary_per_core'

_TT_DEFAULT_TRACE_LEVEL = 3
_TT_PREFIX = 'tensor_tracer'

_TT_NORM = 'norm'
_TT_MAX = 'max'
_TT_MIN = 'min'
_TT_MEAN = 'mean'
_TT_VAR = 'var'
_TT_SIZE = 'size'

TT_SUMMARY_NORM = '%s_%s' % (_TT_PREFIX, _TT_NORM)
TT_SUMMARY_MAX = '%s_%s' % (_TT_PREFIX, _TT_MAX)
TT_SUMMARY_MIN = '%s_%s' % (_TT_PREFIX, _TT_MIN)
TT_SUMMARY_MEAN = '%s_%s' % (_TT_PREFIX, _TT_MEAN)
TT_SUMMARY_VAR = '%s_%s' % (_TT_PREFIX, _TT_VAR)
TT_SUMMARY_SIZE = '%s_%s' % (_TT_PREFIX, _TT_SIZE)

TT_SUMMARY_SIGNATURES = (TT_SUMMARY_NORM, TT_SUMMARY_MAX, TT_SUMMARY_MIN,
                         TT_SUMMARY_MEAN, TT_SUMMARY_VAR, TT_SUMMARY_SIZE)

_TT_DEFAULT_TRACE_LEVEL = 3


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
        _FLAG_NAME_EXCLUDED_OPNAMES)
    self.excluded_optype_re_list = self._flag_value_to_re_list(
        _FLAG_NAME_EXCLUDED_OPTYPES)

    self.included_opname_re_list = self._flag_value_to_re_list(
        _FLAG_NAME_INCLUDED_OPNAMES)
    self.included_optype_re_list = self._flag_value_to_re_list(
        _FLAG_NAME_INCLUDED_OPTYPES)

    self.is_conditional_trace = self._is_conditional_trace_mode()
    self.trace_scalar_ops = self.is_flag_on(_FLAG_NAME_TRACE_SCALAR_OPS)
    self.use_compact_trace = self.is_flag_on(_FLAG_NAME_USE_COMPACT_TRACE)

    # _trace_ops_before_included and _trace_ops_after_included denotes to depth
    # of tracing relative to the ops given in --included_opnames or
    # --included_optypes
    # For example, in the below graph
    #                op1 --> op2 --> op3 --> op4 --> op5
    # If --included_opnames=op3 then only op3 will be traced.
    # If also --trace_before_included_ops=2 (_trace_ops_before_included), then
    # op1 and op2 will be traced as they are at most 2 hops apart from an
    # included op. Similarly, if --trace_after_included_ops=2, then op4 and op5
    # will also be traced.
    self.trace_ops_before_included = self._get_flag_int_value(
        _FLAG_NAME_TRACE_BEFORE_OPS, 0)
    self.trace_ops_after_included = self._get_flag_int_value(
        _FLAG_NAME_TRACE_AFTER_OPS, 0)
    self.trace_stack_size = self._get_flag_int_value(
        _FLAG_NAME_TRACE_STACK_SIZE, 1)
    _, self.graph_dump_path = self.get_flag_value(
        _FLAG_DUMP_BEFORE_AFTER_GRAPHS)
    self.included_cores = self._flag_value_as_int_list(
        _FLAG_NAME_INCLUDED_CORES)
    self.include_less_interesting_ops = self.is_flag_on(
        _FLAG_NAME_INCLUDE_LESS_INTERESTING_OPS)
    self.trace_level = self._get_flag_int_value(
        _FLAG_NAME_TRACE_LEVEL, _TT_DEFAULT_TRACE_LEVEL)
    self.summary_signatures = self._get_summary_signatures()
    self.collect_summary_per_core = self.is_flag_on(_FLAG_NAME_SUMMARY_PER_CORE)

  def _is_conditional_trace_mode(self):
    return self.trace_mode == TRACE_MODE_FULL_IF_NAN

  def _get_report_filepath(self):
    """Sets the path of the output report file."""

    found, report_file_path = self.get_flag_value(
        _FLAG_NAME_REPORT_FILE)
    if found and report_file_path \
       and self.use_test_undeclared_outputs_dir():
      if os.path.isabs(report_file_path):
        raise ValueError('If use_test_undeclared_outputs_dir is set,'
                         'report_file_path cannot be an absolute path (%s)'
                         %report_file_path)
      outputs_dir = self._env.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
      report_file_path = os.path.join(outputs_dir, report_file_path)
    return report_file_path

  def _get_op_range(self):
    """Sets the index range of the Ops that we will consider tracing."""
    found, op_range = self.get_flag_value(_FLAG_NAME_OP_RANGE)
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
    found, trace_dir = self.get_flag_value(_FLAG_NAME_TRACE_DIR)
    if found and trace_dir \
       and self.use_test_undeclared_outputs_dir():
      raise ValueError('Cannot not use --%s and --%s at the same time'
                       %(_FLAG_NAME_TRACE_DIR,
                         _FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR))
    if self.use_test_undeclared_outputs_dir():
      trace_dir = self._env.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
    return trace_dir

  def _get_trace_mode(self):
    """Checks if the given trace mode is valid."""

    found, trace_mode = self.get_flag_value(_FLAG_NAME_TRACE_MODE)
    if not found or not trace_mode:
      trace_mode = TRACE_MODE_NORM
    valid_trace_modes = [
        TRACE_MODE_NAN_INF, TRACE_MODE_PART_TENSOR, TRACE_MODE_FULL_TENSOR,
        TRACE_MODE_NORM, TRACE_MODE_MAX_ABS, TRACE_MODE_FULL_IF_NAN,
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

    found, submode = self.get_flag_value(_FLAG_NAME_SUBMODE)
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
        _FLAG_NAME_ENABLE, _FLAG_NAME_TRACE_MODE, _FLAG_NAME_USE_COMPACT_TRACE,
        _FLAG_NAME_TRACE_SCALAR_OPS, _FLAG_NAME_TRACE_BEFORE_OPS,
        _FLAG_NAME_TRACE_AFTER_OPS, _FLAG_NAME_TRACE_STACK_SIZE,
        _FLAG_NAME_SUBMODE, _FLAG_NAME_EXCLUDED_OPNAMES,
        _FLAG_NAME_EXCLUDED_OPTYPES, _FLAG_NAME_INCLUDED_OPNAMES,
        _FLAG_NAME_INCLUDED_OPTYPES, _FLAG_NAME_TRACE_DIR,
        _FLAG_NAME_INCLUDED_CORES, _FLAG_NAME_REPORT_FILE,
        _FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR,
        _FLAG_NAME_INCLUDE_LESS_INTERESTING_OPS, _FLAG_NAME_OP_RANGE,
        _FLAG_DUMP_BEFORE_AFTER_GRAPHS, _FLAG_NAME_TRACE_LEVEL,
        _FLAG_SUMMARY_SIGNATURES, _FLAG_NAME_SUMMARY_PER_CORE
    ]
    tensor_tracer_flags = self._env.get(_FLAGS_ENV_VAR)
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
            '\n%s'%(flag_name, _FLAGS_ENV_VAR, valid_flag_names))
      pos = match.end()

  def _get_summary_signatures(self):
    """Verifies and returns the summary signatures.

    Returns:
      A dictionary of the signature identifiers {signature: index} that will be
      computed when trace_mode is summary.
    """
    signatures = self._flag_value_as_list(_FLAG_SUMMARY_SIGNATURES)

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
      return {TT_SUMMARY_MAX: 0, TT_SUMMARY_NORM: 1}
    else:
      return {signature: idx for idx, signature in enumerate(tt_signatures)}

  def get_signature_to_agg_fn_map(self):
    """Returns a map that contains the aggragate function for each signature."""
    return {TT_SUMMARY_NORM: linalg_ops.norm,
            TT_SUMMARY_MAX: math_ops.reduce_max,
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

    if found:
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

    tensor_tracer_flags = self._env.get(_FLAGS_ENV_VAR)
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

    if self.is_flag_on(_FLAG_NAME_ENABLE):
      logging.info('Tensor Tracer is enabled with flags %s.' %
                   self._env.get(_FLAGS_ENV_VAR))
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

    return self.is_flag_on(_FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR)
