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

from . import encode
from . import number_types as N


class Table(object):
    """Table wraps a byte slice and provides read access to its data.

    The variable `Pos` indicates the root of the FlatBuffers object therein."""

    __slots__ = ("Bytes", "Pos")

    def __init__(self, buf, pos):
        N.enforce_number(pos, N.UOffsetTFlags)

        self.Bytes = buf
        self.Pos = pos

    def Offset(self, vtableOffset):
        """Offset provides access into the Table's vtable.

        Deprecated fields are ignored by checking the vtable's length."""

        vtable = self.Pos - self.Get(N.SOffsetTFlags, self.Pos)
        vtableEnd = self.Get(N.VOffsetTFlags, vtable)
        if vtableOffset < vtableEnd:
            return self.Get(N.VOffsetTFlags, vtable + vtableOffset)
        return 0

    def Indirect(self, off):
        """Indirect retrieves the relative offset stored at `offset`."""
        N.enforce_number(off, N.UOffsetTFlags)
        return off + encode.Get(N.UOffsetTFlags.packer_type, self.Bytes, off)

    def String(self, off):
        """String gets a string from data stored inside the flatbuffer."""
        N.enforce_number(off, N.UOffsetTFlags)
        off += encode.Get(N.UOffsetTFlags.packer_type, self.Bytes, off)
        start = off + N.UOffsetTFlags.bytewidth
        length = encode.Get(N.UOffsetTFlags.packer_type, self.Bytes, off)
        return bytes(self.Bytes[start:start+length])

    def VectorLen(self, off):
        """VectorLen retrieves the length of the vector whose offset is stored
           at "off" in this object."""
        N.enforce_number(off, N.UOffsetTFlags)

        off += self.Pos
        off += encode.Get(N.UOffsetTFlags.packer_type, self.Bytes, off)
        ret = encode.Get(N.UOffsetTFlags.packer_type, self.Bytes, off)
        return ret

    def Vector(self, off):
        """Vector retrieves the start of data of the vector whose offset is
           stored at "off" in this object."""
        N.enforce_number(off, N.UOffsetTFlags)

        off += self.Pos
        x = off + self.Get(N.UOffsetTFlags, off)
        # data starts after metadata containing the vector length
        x += N.UOffsetTFlags.bytewidth
        return x

    def Union(self, t2, off):
        """Union initializes any Table-derived type to point to the union at
           the given offset."""
        assert type(t2) is Table
        N.enforce_number(off, N.UOffsetTFlags)

        off += self.Pos
        t2.Pos = off + self.Get(N.UOffsetTFlags, off)
        t2.Bytes = self.Bytes

    def Get(self, flags, off):
        """
        Get retrieves a value of the type specified by `flags`  at the
        given offset.
        """
        N.enforce_number(off, N.UOffsetTFlags)
        return flags.py_type(encode.Get(flags.packer_type, self.Bytes, off))

    def GetSlot(self, slot, d, validator_flags):
        N.enforce_number(slot, N.VOffsetTFlags)
        if validator_flags is not None:
            N.enforce_number(d, validator_flags)
        off = self.Offset(slot)
        if off == 0:
            return d
        return self.Get(validator_flags, self.Pos + off)

    def GetVectorAsNumpy(self, flags, off):
        """
        GetVectorAsNumpy returns the vector that starts at `Vector(off)`
        as a numpy array with the type specified by `flags`. The array is
        a `view` into Bytes, so modifying the returned array will
        modify Bytes in place.
        """
        offset = self.Vector(off)
        length = self.VectorLen(off) # TODO: length accounts for bytewidth, right?
        numpy_dtype = N.to_numpy_type(flags)
        return encode.GetVectorAsNumpy(numpy_dtype, self.Bytes, length, offset)

    def GetVOffsetTSlot(self, slot, d):
        """
        GetVOffsetTSlot retrieves the VOffsetT that the given vtable location
        points to. If the vtable value is zero, the default value `d`
        will be returned.
        """

        N.enforce_number(slot, N.VOffsetTFlags)
        N.enforce_number(d, N.VOffsetTFlags)

        off = self.Offset(slot)
        if off == 0:
                return d
        return off
