# Copyright 2014 Google Inc. All rights reserved.
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

import collections
import struct

from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature

np = import_numpy()

# For reference, see:
# https://docs.python.org/2/library/ctypes.html#ctypes-fundamental-data-types-2

# These classes could be collections.namedtuple instances, but those are new
# in 2.6 and we want to work towards 2.5 compatability.

class BoolFlags(object):
    bytewidth = 1
    min_val = False
    max_val = True
    py_type = bool
    name = "bool"
    packer_type = packer.boolean


class Uint8Flags(object):
    bytewidth = 1
    min_val = 0
    max_val = (2**8) - 1
    py_type = int
    name = "uint8"
    packer_type = packer.uint8


class Uint16Flags(object):
    bytewidth = 2
    min_val = 0
    max_val = (2**16) - 1
    py_type = int
    name = "uint16"
    packer_type = packer.uint16


class Uint32Flags(object):
    bytewidth = 4
    min_val = 0
    max_val = (2**32) - 1
    py_type = int
    name = "uint32"
    packer_type = packer.uint32


class Uint64Flags(object):
    bytewidth = 8
    min_val = 0
    max_val = (2**64) - 1
    py_type = int
    name = "uint64"
    packer_type = packer.uint64


class Int8Flags(object):
    bytewidth = 1
    min_val = -(2**7)
    max_val = (2**7) - 1
    py_type = int
    name = "int8"
    packer_type = packer.int8


class Int16Flags(object):
    bytewidth = 2
    min_val = -(2**15)
    max_val = (2**15) - 1
    py_type = int
    name = "int16"
    packer_type = packer.int16


class Int32Flags(object):
    bytewidth = 4
    min_val = -(2**31)
    max_val = (2**31) - 1
    py_type = int
    name = "int32"
    packer_type = packer.int32


class Int64Flags(object):
    bytewidth = 8
    min_val = -(2**63)
    max_val = (2**63) - 1
    py_type = int
    name = "int64"
    packer_type = packer.int64


class Float32Flags(object):
    bytewidth = 4
    min_val = None
    max_val = None
    py_type = float
    name = "float32"
    packer_type = packer.float32


class Float64Flags(object):
    bytewidth = 8
    min_val = None
    max_val = None
    py_type = float
    name = "float64"
    packer_type = packer.float64


class SOffsetTFlags(Int32Flags):
    pass


class UOffsetTFlags(Uint32Flags):
    pass


class VOffsetTFlags(Uint16Flags):
    pass


def valid_number(n, flags):
    if flags.min_val is None and flags.max_val is None:
        return True
    return flags.min_val <= n <= flags.max_val


def enforce_number(n, flags):
    if flags.min_val is None and flags.max_val is None:
        return
    if not flags.min_val <= n <= flags.max_val:
        raise TypeError("bad number %s for type %s" % (str(n), flags.name))


def float32_to_uint32(n):
    packed = struct.pack("<1f", n)
    (converted,) = struct.unpack("<1L", packed)
    return converted


def uint32_to_float32(n):
    packed = struct.pack("<1L", n)
    (unpacked,) = struct.unpack("<1f", packed)
    return unpacked


def float64_to_uint64(n):
    packed = struct.pack("<1d", n)
    (converted,) = struct.unpack("<1Q", packed)
    return converted


def uint64_to_float64(n):
    packed = struct.pack("<1Q", n)
    (unpacked,) = struct.unpack("<1d", packed)
    return unpacked


def to_numpy_type(number_type):
    if np is not None:
        return np.dtype(number_type.name).newbyteorder('<')
    else:
        raise NumpyRequiredForThisFeature('Numpy was not found.')
