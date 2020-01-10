# pylint: disable=wildcard-import,unused-import,g-bad-import-order,line-too-long
"""This library contains classes for launching graphs and executing operations.

The [basic usage](../../get_started/index.md#basic-usage) guide has
examples of how a graph is launched in a [`tf.Session`](#Session).

## Session management

@@Session
@@InteractiveSession

@@get_default_session

## Error classes

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
"""

from tensorflow.python.client.session import InteractiveSession
from tensorflow.python.client.session import Session

from tensorflow.python.framework import errors
from tensorflow.python.framework.errors import OpError

from tensorflow.python.framework.ops import get_default_session
