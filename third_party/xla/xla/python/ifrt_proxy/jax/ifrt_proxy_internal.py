# Copyright 2023 The OpenXLA Authors.
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

"""Library to help create a IFRT proxy client.

This library is no longer recommended nor used in OSS; it is used internally
within google code. TODO(madthanu): Remove library.
"""

import dataclasses
from typing import Callable, Optional

from xla.python import xla_client


@dataclasses.dataclass
class ConnectionOptions:
  """Various connection options.

  Attributes:
    on_disconnect: Optional, a callback that will be called if there was a
      successful connection to the proxy server and Jax commands could be
      issued, but there was a later disconnect before the Client is destroyed.
    on_connection_update: Optional, a callback that will be called with status
      updates about initial connection establishment. The updates will be
      provided as human-readable strings, and an end-user may find them helpful.
  """

  on_disconnect: Optional[Callable[[str], None]] = None
  on_connection_update: Optional[Callable[[str], None]] = None


_backend_created: bool = False
_connection_options: ConnectionOptions = ConnectionOptions()


def get_client(proxy_server_address: str) -> xla_client.Client:
  """Creates an IFRT Proxy client for the given server address."""
  global _backend_created
  _backend_created = True
  py_module = xla_client._xla.ifrt_proxy  # pylint: disable=protected-access
  cpp_options = py_module.ClientConnectionOptions()
  cpp_options.on_disconnect = _connection_options.on_disconnect
  cpp_options.on_connection_update = _connection_options.on_connection_update
  client = py_module.get_client(proxy_server_address, cpp_options)
  return client


def set_connection_options(
    options: ConnectionOptions,
) -> None:
  """Sets the connection options for the "proxy" jax_platforms.

  Args:
    options: See documentation for ConnectionOptions class.

  Raises:
    ValueError: If this function is called after the proxy backend has already
    been created.
  """
  global _connection_options
  if _backend_created:
    raise ValueError(
        "set_connection_options() called after proxy backend was created."
    )
  _connection_options = options
