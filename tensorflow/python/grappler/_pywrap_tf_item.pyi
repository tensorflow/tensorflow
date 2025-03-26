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

class GrapplerItem:
    def __init__(self, *args, **kwargs) -> None: ...

def TF_GetColocationGroups(arg0: GrapplerItem) -> list[list[str]]: ...
def TF_GetOpProperties(arg0: GrapplerItem) -> dict[str, list[bytes]]: ...
def TF_IdentifyImportantOps(arg0: GrapplerItem, arg1: bool) -> list[str]: ...
def TF_NewItem(arg0: bytes, arg1: bool, arg2: bool) -> GrapplerItem: ...
