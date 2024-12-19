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

from typing import Callable, overload

class CalibrationWrapper:
    def __init__(self, arg0: object, arg1: list[str], arg2: list[Callable[[int], None]]) -> None: ...
    def Calibrate(self) -> object: ...
    @overload
    def FeedTensor(self, arg0: object, arg1: str) -> object: ...
    @overload
    def FeedTensor(self, arg0: object) -> object: ...
    @overload
    def Prepare(self, arg0: object, arg1: str) -> object: ...
    @overload
    def Prepare(self, arg0: object) -> object: ...
    @overload
    def Prepare(self, arg0: str) -> object: ...
    @overload
    def Prepare(self) -> object: ...
    @overload
    def QuantizeModel(self, arg0: int, arg1: int, arg2: bool, arg3: int, arg4: int, arg5: bool, arg6: bool) -> object: ...
    @overload
    def QuantizeModel(self, arg0: int, arg1: int, arg2: bool, arg3: str) -> object: ...

def AddIntermediateTensors(arg0: object) -> object: ...
