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

class EventsWriter:
    def __init__(self, arg0: str) -> None: ...
    def Close(self) -> Status: ...
    def FileName(self) -> str: ...
    def Flush(self) -> Status: ...
    def InitWithSuffix(self, arg0: str) -> Status: ...
    def WriteEvent(self, arg0: object) -> None: ...

class Status:
    def __init__(self, *args, **kwargs) -> None: ...
