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

# pylint: disable=unused-import
from tensorflow.python.framework import errors_impl as _impl
# pylint: enable=unused-import
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.framework.errors_impl import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented

# These are referenced in client/client_lib.py.
# Unfortunately, we can't import client_lib to examine
# the references, since it would create a dependency cycle.
_allowed_symbols = [
    "AbortedError",
    "AlreadyExistsError",
    "CancelledError",
    "DataLossError",
    "DeadlineExceededError",
    "FailedPreconditionError",
    "InternalError",
    "InvalidArgumentError",
    "NotFoundError",
    "OpError",
    "OutOfRangeError",
    "PermissionDeniedError",
    "ResourceExhaustedError",
    "UnauthenticatedError",
    "UnavailableError",
    "UnimplementedError",
    "UnknownError",
    "error_code_from_exception_type",
    "exception_type_from_error_code",
    "raise_exception_on_not_ok_status",
    # Scalars that have no docstrings:
    "OK",
    "CANCELLED",
    "UNKNOWN",
    "INVALID_ARGUMENT",
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "ALREADY_EXISTS",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION",
    "ABORTED",
    "OUT_OF_RANGE",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
]

remove_undocumented(__name__, _allowed_symbols)
