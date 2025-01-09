# Copyright 2024 The OpenXLA Authors.
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

import datetime
from typing import Any, Optional, Callable

from xla.python import xla_extension

_Status = Any
Client = xla_extension.Client


class ClientConnectionOptions:
  on_disconnect: Optional[Callable[[_Status], None]] = None
  on_connection_update: Optional[Callable[[str], None]] = None
  connection_timeout_in_seconds: Optional[int] = None


def get_client(
    proxy_server_address: str,
    options: ClientConnectionOptions
) -> Client: ...
