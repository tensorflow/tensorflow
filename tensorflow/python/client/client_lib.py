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

"""This library contains classes for launching graphs and executing operations.

The [basic usage](../../get_started/index.md#basic-usage) guide has
examples of how a graph is launched in a [`tf.Session`](#Session).

## Session management

@@Session
@@InteractiveSession

@@get_default_session

## Error classes and convenience functions

@@OpError
@@CancelledError
@@UnknownError
@@InvalidArgumentError
@@DeadlineExceededError
@@NotFoundError
@@AlreadyExistsError
@@PermissionDeniedError
@@UnauthenticatedError
@@ResourceExhaustedError
@@FailedPreconditionError
@@AbortedError
@@OutOfRangeError
@@UnimplementedError
@@InternalError
@@UnavailableError
@@DataLossError

@@exception_type_from_error_code
@@error_code_from_exception_type
@@raise_exception_on_not_ok_status
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.client.session import InteractiveSession
from tensorflow.python.client.session import Session

from tensorflow.python.framework import errors
from tensorflow.python.framework.errors import OpError

from tensorflow.python.framework.ops import get_default_session
