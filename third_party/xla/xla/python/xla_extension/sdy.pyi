# Copyright 2021 The OpenXLA Authors.
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

from typing import Union
from mlir import ir

def sdy_round_trip_export_pipeline(
    module: ir.module
) -> str: ...

def sdy_round_trip_import_shardings(
    module: ir.module
) -> str: ...

def get_mesh(
    module: ir.module
) -> tuple[tuple[str, int], ...]: ...

def lowered_with_shardy(
    module: ir.module
) -> bool: ...
