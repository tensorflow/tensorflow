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
"""Code transformation exceptions."""

import collections

from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.util import traceback_utils


class FrameInfo(
    collections.namedtuple('FrameInfo',
                           ('filename', 'lineno', 'function_name', 'code',
                            'is_converted', 'is_allowlisted'))):

  __slots__ = ()


def _stack_trace_inside_mapped_code(tb, source_map, converter_filename):
  """Summarizes inner traceback frames up to the call to a given function.

  This functions locates the innermost (i.e. most recent) frame that corresponds
  to code that can be mapped by source_map originated from, and returns a
  translated stack trace ending at that frame. If no such frame is found, the
  entire stack trace is summarized.

  For example, the following code:

    def f():
      for i in tf.range(1):
        z = y + i  # z only defined here

  Would generate this traceback:

    <converted code>
        ag__.for_stmt(...)
    <for_stmt>
        return _known_len_tf_for_stmt(iter_, extra_test, body, init_state)
    <_known_len_tf_for_stmt>
        _disallow_undefs_into_loop(*init_state)
    <_disallow_undefs_into_loop>
        raise ...

  Which is then processed into:

    <f>
        for i in tf.range(1):
    <for_stmt>
        return _known_len_tf_for_stmt(iter_, extra_test, body, init_state)
    <_known_len_tf_for_stmt>
        _disallow_undefs_into_loop(*init_state)
    <_disallow_undefs_into_loop>
        raise ...

  Args:
    tb: traceback.FrameSummary, The traceback corresponding to an error.
      Typically, the output of traceback.Summary.extract(capture_locals=True).
    source_map: Dict[LineLocation, OriginInfo], a source map as created by
      origin_info.create_source_map.
    converter_filename: str, the file path of the converted module. Call frames
      corresponding to this module are elided and their preceding frames are
      marked as allowlisted. Note that frames enclosing converted code are
      dropped using a different mechanism.

  Returns:
    List[FrameInfo]
  """
  result_frames = []
  for filename, line_number, function_name, text in reversed(tb):

    loc = origin_info.LineLocation(filename=filename, lineno=line_number)
    if loc in source_map:
      origin = source_map[loc]
      fi = FrameInfo(
          filename=origin.loc.filename,
          lineno=origin.loc.lineno,
          function_name=origin.function_name,
          code=origin.source_code_line,
          is_converted=True,
          is_allowlisted=False)
      result_frames.append(fi)
      break

    if filename == converter_filename:
      if result_frames:
        prev = result_frames[-1]
        assert not prev.is_converted  # See the if above.
        fi = FrameInfo(
            filename=prev.filename,
            lineno=prev.lineno,
            function_name=prev.function_name,
            code=prev.code,
            is_converted=False,
            is_allowlisted=True)
        result_frames[-1] = fi
      continue

    fi = FrameInfo(
        filename=filename,
        lineno=line_number,
        function_name=function_name,
        code=text,
        is_converted=False,
        is_allowlisted=False)
    result_frames.append(fi)

  return tuple(result_frames)


KNOWN_STRING_CONSTRUCTOR_ERRORS = (
    AssertionError,
    AttributeError,
    NameError,
    NotImplementedError,
    RuntimeError,
    StopIteration,
    TypeError,
    UnboundLocalError,
    ValueError,
)


# KeyError escapes newlines in strings. We create a special subclass
# that doesn't do that. Overriding the name for display purposes; hopefully
# that won't create too many surprises.
class MultilineMessageKeyError(KeyError):

  def __init__(self, message, original_key):
    super(MultilineMessageKeyError, self).__init__(original_key)
    self.__message = message

  def __str__(self):
    return self.__message

MultilineMessageKeyError.__name__ = KeyError.__name__


class ErrorMetadataBase(object):
  """Container objects attached to exceptions raised in user code.

  This metadata allows re-raising exceptions that occur in generated code, with
  a custom error message that includes a stack trace relative to user-readable
  code from which the generated code originated.
  """

  __slots__ = ('translated_stack', 'cause_message')

  def __init__(self, callsite_tb, cause_metadata, cause_message, source_map,
               converter_filename):
    translated_stack = _stack_trace_inside_mapped_code(
        callsite_tb, source_map, converter_filename)

    if cause_metadata is None:
      self.translated_stack = translated_stack
      self.cause_message = cause_message
    else:
      # Daisy chain the translated stacks.
      self.translated_stack = (
          cause_metadata.translated_stack + (translated_stack[-1],))
      self.cause_message = cause_metadata.cause_message

  def get_message(self):
    """Returns the message for the underlying exception."""
    lines = []

    lines.append('in user code:')
    lines.append('')

    for frame_info in reversed(self.translated_stack):
      if (traceback_utils.is_traceback_filtering_enabled() and
          not traceback_utils.include_frame(frame_info.filename)):
        continue

      # Same format with Python traceback.
      formatted_line = (f'    File "{frame_info.filename}", line '
                        f'{frame_info.lineno}, in {frame_info.function_name}')
      if frame_info.is_converted:
        formatted_line += '  *'
      elif frame_info.is_allowlisted:
        formatted_line += '  **'
      lines.append(formatted_line)

      if frame_info.code is None:
        code_snippet = '<source unavailable>'
      else:
        code_snippet = frame_info.code.strip()
      lines.append('        {}'.format(code_snippet))

    lines.append('')

    message_lines = self.cause_message.split('\n')
    for i in range(len(message_lines)):
      message_lines[i] = '    ' + message_lines[i]
    lines.extend(message_lines)

    lines.append('')

    return '\n'.join(lines)

  def create_exception(self, source_error):
    preferred_type = type(source_error)
    if preferred_type.__init__ is Exception.__init__:
      return preferred_type(self.get_message())
    if preferred_type in KNOWN_STRING_CONSTRUCTOR_ERRORS:
      return preferred_type(self.get_message())
    elif preferred_type is KeyError:
      return MultilineMessageKeyError(self.get_message(), self.cause_message)
    return None

  def to_exception(self, source_error):
    exc = self.create_exception(source_error)
    exc.__suppress_context__ = True
    exc.ag_error_metadata = self
    return exc
