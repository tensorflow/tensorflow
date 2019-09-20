# Copyright 2016 Google Inc. All rights reserved.
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

"""
Provide pre-compiled struct packers for encoding and decoding.

See: https://docs.python.org/2/library/struct.html#format-characters
"""

import struct
from . import compat


boolean = struct.Struct(compat.struct_bool_decl)

uint8 = struct.Struct("<B")
uint16 = struct.Struct("<H")
uint32 = struct.Struct("<I")
uint64 = struct.Struct("<Q")

int8 = struct.Struct("<b")
int16 = struct.Struct("<h")
int32 = struct.Struct("<i")
int64 = struct.Struct("<q")

float32 = struct.Struct("<f")
float64 = struct.Struct("<d")

uoffset = uint32
soffset = int32
voffset = uint16
