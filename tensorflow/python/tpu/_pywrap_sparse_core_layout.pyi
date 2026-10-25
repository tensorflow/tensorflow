# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import typing

class SparseCoreLayoutStacker:
    def __init__(self, num_partitions: typing.SupportsInt | typing.SupportsIndex, disable_table_stacking: bool, sparse_cores_per_partition: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def AddTable(self, table_name: str, table_height: typing.SupportsInt | typing.SupportsIndex, table_width: typing.SupportsInt | typing.SupportsIndex, group: str, output_samples: typing.SupportsInt | typing.SupportsIndex, num_features: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def GetLayouts(self, *args, **kwargs): ...
    def SetActivationMemoryBytesLimit(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def SetStackingEnabled(self, arg0: bool) -> None: ...
    def SetVariableShardBytesLimit(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
