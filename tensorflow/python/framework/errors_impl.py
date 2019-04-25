# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Exception types for TensorFlow errors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
import warnings

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import error_interpolation
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export


def _compact_stack_trace(op):
  """Returns a traceback for `op` with common file prefixes stripped."""
  compact_traces = []
  common_prefix = error_interpolation.traceback_files_common_prefix([[op]])
  for frame in op.traceback:
    frame = list(frame)
    filename = frame[tf_stack.TB_FILENAME]
    if filename.startswith(common_prefix):
      filename = filename[len(common_prefix):]
      frame[tf_stack.TB_FILENAME] = filename
    compact_traces.append(tuple(frame))
  return compact_traces


@tf_export("errors.OpError", v1=["errors.OpError", "OpError"])
@deprecation.deprecated_endpoints("OpError")
class OpError(Exception):
  """A generic error that is raised when TensorFlow execution fails.

  Whenever possible, the session will raise a more specific subclass
  of `OpError` from the `tf.errors` module.
  """

  def __init__(self, node_def, op, message, error_code):
    """Creates a new `OpError` indicating that a particular op failed.

    Args:
      node_def: The `node_def_pb2.NodeDef` proto representing the op that
        failed, if known; otherwise None.
      op: The `ops.Operation` that failed, if known; otherwise None.
      message: The message string describing the failure.
      error_code: The `error_codes_pb2.Code` describing the error.
    """
    super(OpError, self).__init__()
    self._node_def = node_def
    self._op = op
    self._message = message
    self._error_code = error_code

  def __reduce__(self):
    # Allow the subclasses to accept less arguments in their __init__.
    init_argspec = tf_inspect.getargspec(self.__class__.__init__)
    args = tuple(getattr(self, arg) for arg in init_argspec.args[1:])
    return self.__class__, args

  @property
  def message(self):
    """The error message that describes the error."""
    return self._message

  @property
  def op(self):
    """The operation that failed, if known.

    *N.B.* If the failed op was synthesized at runtime, e.g. a `Send`
    or `Recv` op, there will be no corresponding
    `tf.Operation`
    object.  In that case, this will return `None`, and you should
    instead use the `tf.errors.OpError.node_def` to
    discover information about the op.

    Returns:
      The `Operation` that failed, or None.
    """
    return self._op

  @property
  def error_code(self):
    """The integer error code that describes the error."""
    return self._error_code

  @property
  def node_def(self):
    """The `NodeDef` proto representing the op that failed."""
    return self._node_def

  def __str__(self):
    if self._op is not None:
      output = ["%s\n\nOriginal stack trace for %r:\n" % (self.message,
                                                          self._op.name,)]
      curr_traceback_list = traceback.format_list(
          _compact_stack_trace(self._op))
      output.extend(curr_traceback_list)
      # pylint: disable=protected-access
      original_op = self._op._original_op
      # pylint: enable=protected-access
      while original_op is not None:
        output.append(
            "\n...which was originally created as op %r, defined at:\n"
            % (original_op.name,))
        prev_traceback_list = curr_traceback_list
        curr_traceback_list = traceback.format_list(
            _compact_stack_trace(original_op))

        # Attempt to elide large common subsequences of the subsequent
        # stack traces.
        #
        # TODO(mrry): Consider computing the actual longest common subsequence.
        is_eliding = False
        elide_count = 0
        last_elided_line = None
        for line, line_in_prev in zip(curr_traceback_list, prev_traceback_list):
          if line == line_in_prev:
            if is_eliding:
              elide_count += 1
              last_elided_line = line
            else:
              output.append(line)
              is_eliding = True
              elide_count = 0
          else:
            if is_eliding:
              if elide_count > 0:
                output.extend(
                    ["[elided %d identical lines from previous traceback]\n"
                     % (elide_count - 1,), last_elided_line])
              is_eliding = False
            output.extend(line)

        # pylint: disable=protected-access
        original_op = original_op._original_op
        # pylint: enable=protected-access
      return "".join(output)
    else:
      return self.message


OK = error_codes_pb2.OK
tf_export("errors.OK").export_constant(__name__, "OK")
CANCELLED = error_codes_pb2.CANCELLED
tf_export("errors.CANCELLED").export_constant(__name__, "CANCELLED")
UNKNOWN = error_codes_pb2.UNKNOWN
tf_export("errors.UNKNOWN").export_constant(__name__, "UNKNOWN")
INVALID_ARGUMENT = error_codes_pb2.INVALID_ARGUMENT
tf_export("errors.INVALID_ARGUMENT").export_constant(__name__,
                                                     "INVALID_ARGUMENT")
DEADLINE_EXCEEDED = error_codes_pb2.DEADLINE_EXCEEDED
tf_export("errors.DEADLINE_EXCEEDED").export_constant(__name__,
                                                      "DEADLINE_EXCEEDED")
NOT_FOUND = error_codes_pb2.NOT_FOUND
tf_export("errors.NOT_FOUND").export_constant(__name__, "NOT_FOUND")
ALREADY_EXISTS = error_codes_pb2.ALREADY_EXISTS
tf_export("errors.ALREADY_EXISTS").export_constant(__name__, "ALREADY_EXISTS")
PERMISSION_DENIED = error_codes_pb2.PERMISSION_DENIED
tf_export("errors.PERMISSION_DENIED").export_constant(__name__,
                                                      "PERMISSION_DENIED")
UNAUTHENTICATED = error_codes_pb2.UNAUTHENTICATED
tf_export("errors.UNAUTHENTICATED").export_constant(__name__, "UNAUTHENTICATED")
RESOURCE_EXHAUSTED = error_codes_pb2.RESOURCE_EXHAUSTED
tf_export("errors.RESOURCE_EXHAUSTED").export_constant(__name__,
                                                       "RESOURCE_EXHAUSTED")
FAILED_PRECONDITION = error_codes_pb2.FAILED_PRECONDITION
tf_export("errors.FAILED_PRECONDITION").export_constant(__name__,
                                                        "FAILED_PRECONDITION")
ABORTED = error_codes_pb2.ABORTED
tf_export("errors.ABORTED").export_constant(__name__, "ABORTED")
OUT_OF_RANGE = error_codes_pb2.OUT_OF_RANGE
tf_export("errors.OUT_OF_RANGE").export_constant(__name__, "OUT_OF_RANGE")
UNIMPLEMENTED = error_codes_pb2.UNIMPLEMENTED
tf_export("errors.UNIMPLEMENTED").export_constant(__name__, "UNIMPLEMENTED")
INTERNAL = error_codes_pb2.INTERNAL
tf_export("errors.INTERNAL").export_constant(__name__, "INTERNAL")
UNAVAILABLE = error_codes_pb2.UNAVAILABLE
tf_export("errors.UNAVAILABLE").export_constant(__name__, "UNAVAILABLE")
DATA_LOSS = error_codes_pb2.DATA_LOSS
tf_export("errors.DATA_LOSS").export_constant(__name__, "DATA_LOSS")


# pylint: disable=line-too-long
@tf_export("errors.CancelledError")
class CancelledError(OpError):
  """Raised when an operation or step is cancelled.

  For example, a long-running operation (e.g.
  `tf.QueueBase.enqueue` may be
  cancelled by running another operation (e.g.
  `tf.QueueBase.close`,
  or by `tf.Session.close`.
  A step that is running such a long-running operation will fail by raising
  `CancelledError`.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `CancelledError`."""
    super(CancelledError, self).__init__(node_def, op, message, CANCELLED)
# pylint: enable=line-too-long


@tf_export("errors.UnknownError")
class UnknownError(OpError):
  """Unknown error.

  An example of where this error may be returned is if a Status value
  received from another address space belongs to an error-space that
  is not known to this address space. Also errors raised by APIs that
  do not return enough error information may be converted to this
  error.

  @@__init__
  """

  def __init__(self, node_def, op, message, error_code=UNKNOWN):
    """Creates an `UnknownError`."""
    super(UnknownError, self).__init__(node_def, op, message, error_code)


@tf_export("errors.InvalidArgumentError")
class InvalidArgumentError(OpError):
  """Raised when an operation receives an invalid argument.

  This may occur, for example, if an operation is receives an input
  tensor that has an invalid value or shape. For example, the
  `tf.matmul` op will raise this
  error if it receives an input that is not a matrix, and the
  `tf.reshape` op will raise
  this error if the new shape does not match the number of elements in the input
  tensor.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `InvalidArgumentError`."""
    super(InvalidArgumentError, self).__init__(node_def, op, message,
                                               INVALID_ARGUMENT)


@tf_export("errors.DeadlineExceededError")
class DeadlineExceededError(OpError):
  """Raised when a deadline expires before an operation could complete.

  This exception is not currently used.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `DeadlineExceededError`."""
    super(DeadlineExceededError, self).__init__(node_def, op, message,
                                                DEADLINE_EXCEEDED)


@tf_export("errors.NotFoundError")
class NotFoundError(OpError):
  """Raised when a requested entity (e.g., a file or directory) was not found.

  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `NotFoundError` if it receives the name of a file that
  does not exist.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `NotFoundError`."""
    super(NotFoundError, self).__init__(node_def, op, message, NOT_FOUND)


@tf_export("errors.AlreadyExistsError")
class AlreadyExistsError(OpError):
  """Raised when an entity that we attempted to create already exists.

  For example, running an operation that saves a file
  (e.g. `tf.train.Saver.save`)
  could potentially raise this exception if an explicit filename for an
  existing file was passed.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `AlreadyExistsError`."""
    super(AlreadyExistsError, self).__init__(node_def, op, message,
                                             ALREADY_EXISTS)


@tf_export("errors.PermissionDeniedError")
class PermissionDeniedError(OpError):
  """Raised when the caller does not have permission to run an operation.

  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `PermissionDeniedError` if it receives the name of a
  file for which the user does not have the read file permission.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `PermissionDeniedError`."""
    super(PermissionDeniedError, self).__init__(node_def, op, message,
                                                PERMISSION_DENIED)


@tf_export("errors.UnauthenticatedError")
class UnauthenticatedError(OpError):
  """The request does not have valid authentication credentials.

  This exception is not currently used.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `UnauthenticatedError`."""
    super(UnauthenticatedError, self).__init__(node_def, op, message,
                                               UNAUTHENTICATED)


@tf_export("errors.ResourceExhaustedError")
class ResourceExhaustedError(OpError):
  """Some resource has been exhausted.

  For example, this error might be raised if a per-user quota is
  exhausted, or perhaps the entire file system is out of space.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `ResourceExhaustedError`."""
    super(ResourceExhaustedError, self).__init__(node_def, op, message,
                                                 RESOURCE_EXHAUSTED)


@tf_export("errors.FailedPreconditionError")
class FailedPreconditionError(OpError):
  """Operation was rejected because the system is not in a state to execute it.

  This exception is most commonly raised when running an operation
  that reads a `tf.Variable`
  before it has been initialized.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `FailedPreconditionError`."""
    super(FailedPreconditionError, self).__init__(node_def, op, message,
                                                  FAILED_PRECONDITION)


@tf_export("errors.AbortedError")
class AbortedError(OpError):
  """The operation was aborted, typically due to a concurrent action.

  For example, running a
  `tf.QueueBase.enqueue`
  operation may raise `AbortedError` if a
  `tf.QueueBase.close` operation
  previously ran.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `AbortedError`."""
    super(AbortedError, self).__init__(node_def, op, message, ABORTED)


@tf_export("errors.OutOfRangeError")
class OutOfRangeError(OpError):
  """Raised when an operation iterates past the valid input range.

  This exception is raised in "end-of-file" conditions, such as when a
  `tf.QueueBase.dequeue`
  operation is blocked on an empty queue, and a
  `tf.QueueBase.close`
  operation executes.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `OutOfRangeError`."""
    super(OutOfRangeError, self).__init__(node_def, op, message,
                                          OUT_OF_RANGE)


@tf_export("errors.UnimplementedError")
class UnimplementedError(OpError):
  """Raised when an operation has not been implemented.

  Some operations may raise this error when passed otherwise-valid
  arguments that it does not currently support. For example, running
  the `tf.nn.max_pool2d` operation
  would raise this error if pooling was requested on the batch dimension,
  because this is not yet supported.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `UnimplementedError`."""
    super(UnimplementedError, self).__init__(node_def, op, message,
                                             UNIMPLEMENTED)


@tf_export("errors.InternalError")
class InternalError(OpError):
  """Raised when the system experiences an internal error.

  This exception is raised when some invariant expected by the runtime
  has been broken. Catching this exception is not recommended.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `InternalError`."""
    super(InternalError, self).__init__(node_def, op, message, INTERNAL)


@tf_export("errors.UnavailableError")
class UnavailableError(OpError):
  """Raised when the runtime is currently unavailable.

  This exception is not currently used.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `UnavailableError`."""
    super(UnavailableError, self).__init__(node_def, op, message,
                                           UNAVAILABLE)


@tf_export("errors.DataLossError")
class DataLossError(OpError):
  """Raised when unrecoverable data loss or corruption is encountered.

  For example, this may be raised by running a
  `tf.WholeFileReader.read`
  operation, if the file is truncated while it is being read.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `DataLossError`."""
    super(DataLossError, self).__init__(node_def, op, message, DATA_LOSS)


_CODE_TO_EXCEPTION_CLASS = {
    CANCELLED: CancelledError,
    UNKNOWN: UnknownError,
    INVALID_ARGUMENT: InvalidArgumentError,
    DEADLINE_EXCEEDED: DeadlineExceededError,
    NOT_FOUND: NotFoundError,
    ALREADY_EXISTS: AlreadyExistsError,
    PERMISSION_DENIED: PermissionDeniedError,
    UNAUTHENTICATED: UnauthenticatedError,
    RESOURCE_EXHAUSTED: ResourceExhaustedError,
    FAILED_PRECONDITION: FailedPreconditionError,
    ABORTED: AbortedError,
    OUT_OF_RANGE: OutOfRangeError,
    UNIMPLEMENTED: UnimplementedError,
    INTERNAL: InternalError,
    UNAVAILABLE: UnavailableError,
    DATA_LOSS: DataLossError,
}

c_api.PyExceptionRegistry_Init(_CODE_TO_EXCEPTION_CLASS)

_EXCEPTION_CLASS_TO_CODE = {
    class_: code for code, class_ in _CODE_TO_EXCEPTION_CLASS.items()}


@tf_export(v1=["errors.exception_type_from_error_code"])
def exception_type_from_error_code(error_code):
  return _CODE_TO_EXCEPTION_CLASS[error_code]


@tf_export(v1=["errors.error_code_from_exception_type"])
def error_code_from_exception_type(cls):
  try:
    return _EXCEPTION_CLASS_TO_CODE[cls]
  except KeyError:
    warnings.warn("Unknown class exception")
    return UnknownError(None, None, "Unknown class exception", None)


def _make_specific_exception(node_def, op, message, error_code):
  try:
    exc_type = exception_type_from_error_code(error_code)
    return exc_type(node_def, op, message)
  except KeyError:
    warnings.warn("Unknown error code: %d" % error_code)
    return UnknownError(node_def, op, message, error_code)


# Named like a function for backwards compatibility with the
# @tf_contextlib.contextmanager version, which was switched to a class to avoid
# some object creation overhead.
# TODO(b/77295559): expand use of TF_Status* SWIG typemap and deprecate this.
@tf_export(v1=["errors.raise_exception_on_not_ok_status"])  # pylint: disable=invalid-name
class raise_exception_on_not_ok_status(object):
  """Context manager to check for C API status."""

  def __enter__(self):
    self.status = c_api_util.ScopedTFStatus()
    return self.status.status

  def __exit__(self, type_arg, value_arg, traceback_arg):
    try:
      if c_api.TF_GetCode(self.status.status) != 0:
        raise _make_specific_exception(
            None, None,
            compat.as_text(c_api.TF_Message(self.status.status)),
            c_api.TF_GetCode(self.status.status))
    # Delete the underlying status object from memory otherwise it stays alive
    # as there is a reference to status from this from the traceback due to
    # raise.
    finally:
      del self.status
    return False  # False values do not suppress exceptions
