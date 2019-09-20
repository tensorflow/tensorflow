// Copyright 2018 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import std

namespace flatbuffers

class handle:
    buf_:string
    pos_:int

// More strongly typed than a naked int, at no cost.
struct offset:
    o:int

enum sizeof:
    sz_8 = 1
    sz_16 = 2
    sz_32 = 4
    sz_64 = 8
    sz_voffset = 2
    sz_uoffset = 4
    sz_soffset = 4
    sz_metadata_fields = 2

class builder:
    buf = "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    current_vtable:[int] = []
    head = 0
    minalign = 1
    object_end = 0
    vtables:[int] = []
    nested = false
    finished = false

    // Optionally call this right after creating the builder for a larger initial buffer.
    def Initial(initial_size:int):
        buf = "\x00".repeat_string(initial_size)

    def Start():
        // Get the start of useful data in the underlying byte buffer.
        return buf.length - head

    def Offset():
        // Offset relative to the end of the buffer.
        return offset { head }

    // Returns a copy of the part of the buffer containing only the finished FlatBuffer
    def SizedCopy():
        assert finished
        return buf.substring(Start(), -1)

    def StartNesting():
        assert not nested
        nested = true

    def EndNesting():
        assert nested
        nested = false

    def StartObject(numfields):
        StartNesting()
        current_vtable = map(numfields): 0
        object_end = head
        minalign = 1

    def EndObject():
        EndNesting()
        // Prepend a zero scalar to the object. Later in this function we'll
        // write an offset here that points to the object's vtable:
        PrependInt32(0)
        let object_offset = head
        // Write out new vtable speculatively.
        let vtable_size = (current_vtable.length + sz_metadata_fields) * sz_voffset
        while current_vtable.length:
            let o = current_vtable.pop()
            PrependVOffsetT(if o: object_offset - o else: 0)
        // The two metadata fields are written last.
        // First, store the object bytesize:
        PrependVOffsetT(object_offset - object_end)
        // Second, store the vtable bytesize:
        PrependVOffsetT(vtable_size)
        // Search backwards through existing vtables, because similar vtables
        // are likely to have been recently appended. See
        // BenchmarkVtableDeduplication for a case in which this heuristic
        // saves about 30% of the time used in writing objects with duplicate
        // tables.
        def find_existing_table():
            reverse(vtables) vt2_offset:
                // Find the other vtable:
                let vt2_start = buf.length - vt2_offset
                let vt2_len = buf.read_int16_le(vt2_start)
                // Compare the other vtable to the one under consideration.
                // If they are equal, return the offset:
                if vtable_size == vt2_len and
                    not compare_substring(buf, Start(), buf, vt2_start, vtable_size):
                        return vt2_offset
            return 0
        let existing_vtable = find_existing_table()
        if existing_vtable:
            // Found a duplicate vtable, remove the one we wrote.
            head = object_offset
            // Write the offset to the found vtable in the
            // already-allocated offset at the beginning of this object:
            buf.write_int32_le(Start(), existing_vtable - object_offset)
        else:
            // Did not find a vtable, so keep the one we wrote.
            // Next, write the offset to the new vtable in the
            // already-allocated offset at the beginning of this object:
            buf.write_int32_le(buf.length - object_offset, head - object_offset)
            // Finally, store this vtable in memory for future
            // deduplication:
            vtables.push(head)
        return offset { object_offset }

    def Pad(n):
        for(n):
            buf, head = buf.write_int8_le_back(head, 0)

    def Prep(size, additional_bytes):
        // Track the biggest thing we've ever aligned to.
        if size > minalign:
            minalign = size
        // Find the amount of alignment needed such that `size` is properly
        // aligned after `additionalBytes`:
        let align_size = ((~(head + additional_bytes)) + 1) & (size - 1)
        Pad(align_size)

    def PrependUOffsetTRelative(off:offset):
        // Prepends an unsigned offset into vector data, relative to where it will be written.
        Prep(sz_uoffset, 0)
        assert off.o <= head
        PlaceUOffsetT(head - off.o + sz_uoffset)

    def StartVector(elem_size, num_elems, alignment):
        // Initializes bookkeeping for writing a new vector.
        StartNesting()
        Prep(sz_32, elem_size * num_elems)
        Prep(alignment, elem_size * num_elems)  // In case alignment > int.
        return Offset()

    def EndVector(vector_num_elems):
        EndNesting()
        // we already made space for this, so write without PrependUint32
        PlaceUOffsetT(vector_num_elems)
        return Offset()

    def CreateString(s:string):
        // writes a null-terminated byte string.
        StartNesting()
        Prep(sz_32, s.length + 1)
        buf, head = buf.write_substring_back(head, s, true)
        return EndVector(s.length)

    def CreateByteVector(s:string):
        // writes a non-null-terminated byte string.
        StartNesting()
        Prep(sz_32, s.length)
        buf, head = buf.write_substring_back(head, s, false)
        return EndVector(s.length)

    def Slot(slotnum):
        assert nested
        while current_vtable.length <= slotnum: current_vtable.push(0)
        current_vtable[slotnum] = head

    def __Finish(root_table:offset, size_prefix:int):
        // Finish finalizes a buffer, pointing to the given root_table
        assert not finished
        assert not nested
        var prep_size = sz_32
        if size_prefix:
            prep_size += sz_32
        Prep(minalign, prep_size)
        PrependUOffsetTRelative(root_table)
        if size_prefix:
            PrependInt32(head)
        finished = true
        return Start()

    def Finish(root_table:offset):
        return __Finish(root_table, false)

    def FinishSizePrefixed(root_table:offset):
        return __Finish(root_table, true)

    def PrependBool(x):
        buf, head = buf.write_int8_le_back(head, x)

    def PrependByte(x):
        buf, head = buf.write_int8_le_back(head, x)

    def PrependUint8(x):
        buf, head = buf.write_int8_le_back(head, x)

    def PrependUint16(x):
        Prep(sz_16, 0)
        buf, head = buf.write_int16_le_back(head, x)

    def PrependUint32(x):
        Prep(sz_32, 0)
        buf, head = buf.write_int32_le_back(head, x)

    def PrependUint64(x):
        Prep(sz_64, 0)
        buf, head = buf.write_int64_le_back(head, x)

    def PrependInt8(x):
        buf, head = buf.write_int8_le_back(head, x)

    def PrependInt16(x):
        Prep(sz_16, 0)
        buf, head = buf.write_int16_le_back(head, x)

    def PrependInt32(x):
        Prep(sz_32, 0)
        buf, head = buf.write_int32_le_back(head, x)

    def PrependInt64(x):
        Prep(sz_64, 0)
        buf, head = buf.write_int64_le_back(head, x)

    def PrependFloat32(x):
        Prep(sz_32, 0)
        buf, head = buf.write_float32_le_back(head, x)

    def PrependFloat64(x):
        Prep(sz_64, 0)
        buf, head = buf.write_float64_le_back(head, x)

    def PrependVOffsetT(x):
        Prep(sz_voffset, 0)
        buf, head = buf.write_int16_le_back(head, x)

    def PlaceVOffsetT(x):
        buf, head = buf.write_int16_le_back(head, x)

    def PlaceSOffsetT(x):
        buf, head = buf.write_int32_le_back(head, x)

    def PlaceUOffsetT(x):
        buf, head = buf.write_int32_le_back(head, x)

    def PrependSlot(o:int, x, d, f):
        if x != d:
            f(x)
            Slot(o)

    def PrependBoolSlot(o, x, d): PrependSlot(o, x, d): PrependBool(_)
    def PrependByteSlot(o, x, d): PrependSlot(o, x, d): PrependByte(_)
    def PrependUint8Slot(o, x, d): PrependSlot(o, x, d): PrependUint8(_)
    def PrependUint16Slot(o, x, d): PrependSlot(o, x, d): PrependUint16(_)
    def PrependUint32Slot(o, x, d): PrependSlot(o, x, d): PrependUint32(_)
    def PrependUint64Slot(o, x, d): PrependSlot(o, x, d): PrependUint64(_)
    def PrependInt8Slot(o, x, d): PrependSlot(o, x, d): PrependInt8(_)
    def PrependInt16Slot(o, x, d): PrependSlot(o, x, d): PrependInt16(_)
    def PrependInt32Slot(o, x, d): PrependSlot(o, x, d): PrependInt32(_)
    def PrependInt64Slot(o, x, d): PrependSlot(o, x, d): PrependInt64(_)
    def PrependFloat32Slot(o, x, d): PrependSlot(o, x, d): PrependFloat32(_)
    def PrependFloat64Slot(o, x, d): PrependSlot(o, x, d): PrependFloat64(_)

    def PrependUOffsetTRelativeSlot(o:int, x:offset):
        if x.o:
            PrependUOffsetTRelative(x)
            Slot(o)

    def PrependStructSlot(v:int, x:offset):
        if x.o:
            // Structs are always stored inline, so need to be created right
            // where they are used. You'll get this error if you created it
            // elsewhere.
            assert x.o == head
            Slot(v)
