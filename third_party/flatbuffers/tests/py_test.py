# coding=utf-8
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

import os.path
import sys
import imp
PY_VERSION = sys.version_info[:2]

import ctypes
from collections import defaultdict
import math
import random
import timeit
import unittest

from flatbuffers import compat
from flatbuffers import util
from flatbuffers.compat import range_func as compat_range
from flatbuffers.compat import NumpyRequiredForThisFeature

import flatbuffers
from flatbuffers import number_types as N

import MyGame  # refers to generated code
import MyGame.Example  # refers to generated code
import MyGame.Example.Any  # refers to generated code
import MyGame.Example.Color  # refers to generated code
import MyGame.Example.Monster  # refers to generated code
import MyGame.Example.Test  # refers to generated code
import MyGame.Example.Stat  # refers to generated code
import MyGame.Example.Vec3  # refers to generated code
import MyGame.MonsterExtra  # refers to generated code
import MyGame.Example.ArrayTable  # refers to generated code
import MyGame.Example.ArrayStruct  # refers to generated code
import MyGame.Example.NestedStruct  # refers to generated code
import MyGame.Example.TestEnum  # refers to generated code

def assertRaises(test_case, fn, exception_class):
    ''' Backwards-compatible assertion for exceptions raised. '''

    exc = None
    try:
        fn()
    except Exception as e:
        exc = e
    test_case.assertTrue(exc is not None)
    test_case.assertTrue(isinstance(exc, exception_class))


class TestWireFormat(unittest.TestCase):
    def test_wire_format(self):
        # Verify that using the generated Python code builds a buffer without
        # returning errors, and is interpreted correctly, for size prefixed
        # representation and regular:
        for sizePrefix in [True, False]:
            for file_identifier in [None, b"MONS"]:
                gen_buf, gen_off = make_monster_from_generated_code(sizePrefix=sizePrefix, file_identifier=file_identifier)
                CheckReadBuffer(gen_buf, gen_off, sizePrefix=sizePrefix, file_identifier=file_identifier)

        # Verify that the canonical flatbuffer file is readable by the
        # generated Python code. Note that context managers are not part of
        # Python 2.5, so we use the simpler open/close methods here:
        f = open('monsterdata_test.mon', 'rb')
        canonicalWireData = f.read()
        f.close()
        CheckReadBuffer(bytearray(canonicalWireData), 0, file_identifier=b'MONS')

        # Write the generated buffer out to a file:
        f = open('monsterdata_python_wire.mon', 'wb')
        f.write(gen_buf[gen_off:])
        f.close()


def CheckReadBuffer(buf, offset, sizePrefix=False, file_identifier=None):
    ''' CheckReadBuffer checks that the given buffer is evaluated correctly
        as the example Monster. '''

    def asserter(stmt):
        ''' An assertion helper that is separated from TestCase classes. '''
        if not stmt:
            raise AssertionError('CheckReadBuffer case failed')
    if file_identifier:
        # test prior to removal of size_prefix
        asserter(util.GetBufferIdentifier(buf, offset, size_prefixed=sizePrefix) == file_identifier)
        asserter(util.BufferHasIdentifier(buf, offset, file_identifier=file_identifier, size_prefixed=sizePrefix))
    if sizePrefix:
        size = util.GetSizePrefix(buf, offset)
        asserter(size == len(buf[offset:])-4)
        buf, offset = util.RemoveSizePrefix(buf, offset)
    if file_identifier:
        asserter(MyGame.Example.Monster.Monster.MonsterBufferHasIdentifier(buf, offset))
    else:
        asserter(not MyGame.Example.Monster.Monster.MonsterBufferHasIdentifier(buf, offset))
    monster = MyGame.Example.Monster.Monster.GetRootAsMonster(buf, offset)

    asserter(monster.Hp() == 80)
    asserter(monster.Mana() == 150)
    asserter(monster.Name() == b'MyMonster')

    # initialize a Vec3 from Pos()
    vec = monster.Pos()
    asserter(vec is not None)

    # verify the properties of the Vec3
    asserter(vec.X() == 1.0)
    asserter(vec.Y() == 2.0)
    asserter(vec.Z() == 3.0)
    asserter(vec.Test1() == 3.0)
    asserter(vec.Test2() == 2)

    # initialize a Test from Test3(...)
    t = MyGame.Example.Test.Test()
    t = vec.Test3(t)
    asserter(t is not None)

    # verify the properties of the Test
    asserter(t.A() == 5)
    asserter(t.B() == 6)

    # verify that the enum code matches the enum declaration:
    union_type = MyGame.Example.Any.Any
    asserter(monster.TestType() == union_type.Monster)

    # initialize a Table from a union field Test(...)
    table2 = monster.Test()
    asserter(type(table2) is flatbuffers.table.Table)

    # initialize a Monster from the Table from the union
    monster2 = MyGame.Example.Monster.Monster()
    monster2.Init(table2.Bytes, table2.Pos)

    asserter(monster2.Name() == b"Fred")

    # iterate through the first monster's inventory:
    asserter(monster.InventoryLength() == 5)

    invsum = 0
    for i in compat_range(monster.InventoryLength()):
        v = monster.Inventory(i)
        invsum += int(v)
    asserter(invsum == 10)

    for i in range(5):
        asserter(monster.VectorOfLongs(i) == 10 ** (i * 2))

    asserter(([-1.7976931348623157e+308, 0, 1.7976931348623157e+308]
              == [monster.VectorOfDoubles(i)
                  for i in range(monster.VectorOfDoublesLength())]))

    try:
        imp.find_module('numpy')
        # if numpy exists, then we should be able to get the
        # vector as a numpy array
        import numpy as np

        asserter(monster.InventoryAsNumpy().sum() == 10)
        asserter(monster.InventoryAsNumpy().dtype == np.dtype('uint8'))

        VectorOfLongs = monster.VectorOfLongsAsNumpy()
        asserter(VectorOfLongs.dtype == np.dtype('int64'))
        for i in range(5):
            asserter(VectorOfLongs[i] == 10 ** (i * 2))

        VectorOfDoubles = monster.VectorOfDoublesAsNumpy()
        asserter(VectorOfDoubles.dtype == np.dtype('float64'))
        asserter(VectorOfDoubles[0] == np.finfo('float64').min)
        asserter(VectorOfDoubles[1] == 0.0)
        asserter(VectorOfDoubles[2] == np.finfo('float64').max)

    except ImportError:
        # If numpy does not exist, trying to get vector as numpy
        # array should raise NumpyRequiredForThisFeature. The way
        # assertRaises has been implemented prevents us from
        # asserting this error is raised outside of a test case.
        pass

    asserter(monster.Test4Length() == 2)

    # create a 'Test' object and populate it:
    test0 = monster.Test4(0)
    asserter(type(test0) is MyGame.Example.Test.Test)

    test1 = monster.Test4(1)
    asserter(type(test1) is MyGame.Example.Test.Test)

    # the position of test0 and test1 are swapped in monsterdata_java_wire
    # and monsterdata_test_wire, so ignore ordering
    v0 = test0.A()
    v1 = test0.B()
    v2 = test1.A()
    v3 = test1.B()
    sumtest12 = int(v0) + int(v1) + int(v2) + int(v3)

    asserter(sumtest12 == 100)

    asserter(monster.TestarrayofstringLength() == 2)
    asserter(monster.Testarrayofstring(0) == b"test1")
    asserter(monster.Testarrayofstring(1) == b"test2")

    asserter(monster.TestarrayoftablesLength() == 0)
    asserter(monster.TestnestedflatbufferLength() == 0)
    asserter(monster.Testempty() is None)


class TestFuzz(unittest.TestCase):
    ''' Low level stress/fuzz test: serialize/deserialize a variety of
        different kinds of data in different combinations '''

    binary_type = compat.binary_types[0] # this will always exist
    ofInt32Bytes = binary_type([0x83, 0x33, 0x33, 0x33])
    ofInt64Bytes = binary_type([0x84, 0x44, 0x44, 0x44,
                                0x44, 0x44, 0x44, 0x44])
    overflowingInt32Val = flatbuffers.encode.Get(flatbuffers.packer.int32,
                                                 ofInt32Bytes, 0)
    overflowingInt64Val = flatbuffers.encode.Get(flatbuffers.packer.int64,
                                                 ofInt64Bytes, 0)

    # Values we're testing against: chosen to ensure no bits get chopped
    # off anywhere, and also be different from eachother.
    boolVal = True
    int8Val = N.Int8Flags.py_type(-127) # 0x81
    uint8Val = N.Uint8Flags.py_type(0xFF)
    int16Val = N.Int16Flags.py_type(-32222) # 0x8222
    uint16Val = N.Uint16Flags.py_type(0xFEEE)
    int32Val = N.Int32Flags.py_type(overflowingInt32Val)
    uint32Val = N.Uint32Flags.py_type(0xFDDDDDDD)
    int64Val = N.Int64Flags.py_type(overflowingInt64Val)
    uint64Val = N.Uint64Flags.py_type(0xFCCCCCCCCCCCCCCC)
    # Python uses doubles, so force it here
    float32Val = N.Float32Flags.py_type(ctypes.c_float(3.14159).value)
    float64Val = N.Float64Flags.py_type(3.14159265359)

    def test_fuzz(self):
        return self.check_once(11, 100)

    def check_once(self, fuzzFields, fuzzObjects):
        testValuesMax = 11 # hardcoded to the number of scalar types

        builder = flatbuffers.Builder(0)
        l = LCG()

        objects = [0 for _ in compat_range(fuzzObjects)]

        # Generate fuzzObjects random objects each consisting of
        # fuzzFields fields, each of a random type.
        for i in compat_range(fuzzObjects):
            builder.StartObject(fuzzFields)

            for j in compat_range(fuzzFields):
                choice = int(l.Next()) % testValuesMax
                if choice == 0:
                    builder.PrependBoolSlot(int(j), self.boolVal, False)
                elif choice == 1:
                    builder.PrependInt8Slot(int(j), self.int8Val, 0)
                elif choice == 2:
                    builder.PrependUint8Slot(int(j), self.uint8Val, 0)
                elif choice == 3:
                    builder.PrependInt16Slot(int(j), self.int16Val, 0)
                elif choice == 4:
                    builder.PrependUint16Slot(int(j), self.uint16Val, 0)
                elif choice == 5:
                    builder.PrependInt32Slot(int(j), self.int32Val, 0)
                elif choice == 6:
                    builder.PrependUint32Slot(int(j), self.uint32Val, 0)
                elif choice == 7:
                    builder.PrependInt64Slot(int(j), self.int64Val, 0)
                elif choice == 8:
                    builder.PrependUint64Slot(int(j), self.uint64Val, 0)
                elif choice == 9:
                    builder.PrependFloat32Slot(int(j), self.float32Val, 0)
                elif choice == 10:
                    builder.PrependFloat64Slot(int(j), self.float64Val, 0)
                else:
                    raise RuntimeError('unreachable')

            off = builder.EndObject()

            # store the offset from the end of the builder buffer,
            # since it will keep growing:
            objects[i] = off

        # Do some bookkeeping to generate stats on fuzzes:
        stats = defaultdict(int)
        def check(table, desc, want, got):
            stats[desc] += 1
            self.assertEqual(want, got, "%s != %s, %s" % (want, got, desc))

        l = LCG()  # Reset.

        # Test that all objects we generated are readable and return the
        # expected values. We generate random objects in the same order
        # so this is deterministic.
        for i in compat_range(fuzzObjects):

            table = flatbuffers.table.Table(builder.Bytes,
                                            len(builder.Bytes) - objects[i])

            for j in compat_range(fuzzFields):
                field_count = flatbuffers.builder.VtableMetadataFields + j
                f = N.VOffsetTFlags.py_type(field_count *
                                            N.VOffsetTFlags.bytewidth)
                choice = int(l.Next()) % testValuesMax

                if choice == 0:
                    check(table, "bool", self.boolVal,
                          table.GetSlot(f, False, N.BoolFlags))
                elif choice == 1:
                    check(table, "int8", self.int8Val,
                          table.GetSlot(f, 0, N.Int8Flags))
                elif choice == 2:
                    check(table, "uint8", self.uint8Val,
                          table.GetSlot(f, 0, N.Uint8Flags))
                elif choice == 3:
                    check(table, "int16", self.int16Val,
                          table.GetSlot(f, 0, N.Int16Flags))
                elif choice == 4:
                    check(table, "uint16", self.uint16Val,
                          table.GetSlot(f, 0, N.Uint16Flags))
                elif choice == 5:
                    check(table, "int32", self.int32Val,
                          table.GetSlot(f, 0, N.Int32Flags))
                elif choice == 6:
                    check(table, "uint32", self.uint32Val,
                          table.GetSlot(f, 0, N.Uint32Flags))
                elif choice == 7:
                    check(table, "int64", self.int64Val,
                          table.GetSlot(f, 0, N.Int64Flags))
                elif choice == 8:
                    check(table, "uint64", self.uint64Val,
                          table.GetSlot(f, 0, N.Uint64Flags))
                elif choice == 9:
                    check(table, "float32", self.float32Val,
                          table.GetSlot(f, 0, N.Float32Flags))
                elif choice == 10:
                    check(table, "float64", self.float64Val,
                          table.GetSlot(f, 0, N.Float64Flags))
                else:
                    raise RuntimeError('unreachable')

        # If enough checks were made, verify that all scalar types were used:
        self.assertEqual(testValuesMax, len(stats),
                "fuzzing failed to test all scalar types: %s" % stats)


class TestByteLayout(unittest.TestCase):
    ''' TestByteLayout checks the bytes of a Builder in various scenarios. '''

    def assertBuilderEquals(self, builder, want_chars_or_ints):
        def integerize(x):
            if isinstance(x, compat.string_types):
                return ord(x)
            return x

        want_ints = list(map(integerize, want_chars_or_ints))
        want = bytearray(want_ints)
        got = builder.Bytes[builder.Head():] # use the buffer directly
        self.assertEqual(want, got)

    def test_numbers(self):
        b = flatbuffers.Builder(0)
        self.assertBuilderEquals(b, [])
        b.PrependBool(True)
        self.assertBuilderEquals(b, [1])
        b.PrependInt8(-127)
        self.assertBuilderEquals(b, [129, 1])
        b.PrependUint8(255)
        self.assertBuilderEquals(b, [255, 129, 1])
        b.PrependInt16(-32222)
        self.assertBuilderEquals(b, [0x22, 0x82, 0, 255, 129, 1]) # first pad
        b.PrependUint16(0xFEEE)
        # no pad this time:
        self.assertBuilderEquals(b, [0xEE, 0xFE, 0x22, 0x82, 0, 255, 129, 1])
        b.PrependInt32(-53687092)
        self.assertBuilderEquals(b, [204, 204, 204, 252, 0xEE, 0xFE,
                                     0x22, 0x82, 0, 255, 129, 1])
        b.PrependUint32(0x98765432)
        self.assertBuilderEquals(b, [0x32, 0x54, 0x76, 0x98,
                                     204, 204, 204, 252,
                                     0xEE, 0xFE, 0x22, 0x82,
                                     0, 255, 129, 1])

    def test_numbers64(self):
        b = flatbuffers.Builder(0)
        b.PrependUint64(0x1122334455667788)
        self.assertBuilderEquals(b, [0x88, 0x77, 0x66, 0x55,
                                     0x44, 0x33, 0x22, 0x11])

        b = flatbuffers.Builder(0)
        b.PrependInt64(0x1122334455667788)
        self.assertBuilderEquals(b, [0x88, 0x77, 0x66, 0x55,
                                     0x44, 0x33, 0x22, 0x11])

    def test_1xbyte_vector(self):
        b = flatbuffers.Builder(0)
        self.assertBuilderEquals(b, [])
        b.StartVector(flatbuffers.number_types.Uint8Flags.bytewidth, 1, 1)
        self.assertBuilderEquals(b, [0, 0, 0]) # align to 4bytes
        b.PrependByte(1)
        self.assertBuilderEquals(b, [1, 0, 0, 0])
        b.EndVector(1)
        self.assertBuilderEquals(b, [1, 0, 0, 0, 1, 0, 0, 0]) # padding

    def test_2xbyte_vector(self):
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint8Flags.bytewidth, 2, 1)
        self.assertBuilderEquals(b, [0, 0]) # align to 4bytes
        b.PrependByte(1)
        self.assertBuilderEquals(b, [1, 0, 0])
        b.PrependByte(2)
        self.assertBuilderEquals(b, [2, 1, 0, 0])
        b.EndVector(2)
        self.assertBuilderEquals(b, [2, 0, 0, 0, 2, 1, 0, 0]) # padding

    def test_1xuint16_vector(self):
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint16Flags.bytewidth, 1, 1)
        self.assertBuilderEquals(b, [0, 0]) # align to 4bytes
        b.PrependUint16(1)
        self.assertBuilderEquals(b, [1, 0, 0, 0])
        b.EndVector(1)
        self.assertBuilderEquals(b, [1, 0, 0, 0, 1, 0, 0, 0]) # padding

    def test_2xuint16_vector(self):
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint16Flags.bytewidth, 2, 1)
        self.assertBuilderEquals(b, []) # align to 4bytes
        b.PrependUint16(0xABCD)
        self.assertBuilderEquals(b, [0xCD, 0xAB])
        b.PrependUint16(0xDCBA)
        self.assertBuilderEquals(b, [0xBA, 0xDC, 0xCD, 0xAB])
        b.EndVector(2)
        self.assertBuilderEquals(b, [2, 0, 0, 0, 0xBA, 0xDC, 0xCD, 0xAB])

    def test_create_ascii_string(self):
        b = flatbuffers.Builder(0)
        b.CreateString(u"foo", encoding='ascii')

        # 0-terminated, no pad:
        self.assertBuilderEquals(b, [3, 0, 0, 0, 'f', 'o', 'o', 0])
        b.CreateString(u"moop", encoding='ascii')
        # 0-terminated, 3-byte pad:
        self.assertBuilderEquals(b, [4, 0, 0, 0, 'm', 'o', 'o', 'p',
                                     0, 0, 0, 0,
                                     3, 0, 0, 0, 'f', 'o', 'o', 0])

    def test_create_utf8_string(self):
        b = flatbuffers.Builder(0)
        b.CreateString(u"Цлїςσδε")
        self.assertBuilderEquals(b, "\x0e\x00\x00\x00\xd0\xa6\xd0\xbb\xd1\x97" \
            "\xcf\x82\xcf\x83\xce\xb4\xce\xb5\x00\x00")

        b.CreateString(u"ﾌﾑｱﾑｶﾓｹﾓ")
        self.assertBuilderEquals(b, "\x18\x00\x00\x00\xef\xbe\x8c\xef\xbe\x91" \
            "\xef\xbd\xb1\xef\xbe\x91\xef\xbd\xb6\xef\xbe\x93\xef\xbd\xb9\xef" \
            "\xbe\x93\x00\x00\x00\x00\x0e\x00\x00\x00\xd0\xa6\xd0\xbb\xd1\x97" \
            "\xcf\x82\xcf\x83\xce\xb4\xce\xb5\x00\x00")

    def test_create_arbitrary_string(self):
        b = flatbuffers.Builder(0)
        s = "\x01\x02\x03"
        b.CreateString(s) # Default encoding is utf-8.
        # 0-terminated, no pad:
        self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 2, 3, 0])
        s2 = "\x04\x05\x06\x07"
        b.CreateString(s2) # Default encoding is utf-8.
        # 0-terminated, 3-byte pad:
        self.assertBuilderEquals(b, [4, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0,
                                     3, 0, 0, 0, 1, 2, 3, 0])

    def test_create_byte_vector(self):
        b = flatbuffers.Builder(0)
        b.CreateByteVector(b"")
        # 0-byte pad:
        self.assertBuilderEquals(b, [0, 0, 0, 0])

        b = flatbuffers.Builder(0)
        b.CreateByteVector(b"\x01\x02\x03")
        # 1-byte pad:
        self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 2, 3, 0])

    def test_create_numpy_vector_int8(self):
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            import numpy as np

            # Systems endian:
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, -3], dtype=np.int8)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,  # vector length
                1, 2, 256 - 3, 0   # vector value + padding
            ])

            # Reverse endian:
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,  # vector length
                1, 2, 256 - 3, 0   # vector value + padding
            ])
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                NumpyRequiredForThisFeature)

    def test_create_numpy_vector_uint16(self):
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            import numpy as np

            # Systems endian:
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, 312], dtype=np.uint16)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,     # vector length
                1, 0,           # 1
                2, 0,           # 2
                312 - 256, 1,   # 312
                0, 0            # padding
            ])

            # Reverse endian:
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,     # vector length
                1, 0,           # 1
                2, 0,           # 2
                312 - 256, 1,   # 312
                0, 0            # padding
            ])
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                NumpyRequiredForThisFeature)

    def test_create_numpy_vector_int64(self):
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            import numpy as np

            # Systems endian:
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, -12], dtype=np.int64)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,                     # vector length
                1, 0, 0, 0, 0, 0, 0, 0,         # 1
                2, 0, 0, 0, 0, 0, 0, 0,         # 2
                256 - 12, 255, 255, 255, 255, 255, 255, 255   # -12
            ])

            # Reverse endian:
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,                     # vector length
                1, 0, 0, 0, 0, 0, 0, 0,         # 1
                2, 0, 0, 0, 0, 0, 0, 0,         # 2
                256 - 12, 255, 255, 255, 255, 255, 255, 255   # -12
            ])

        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                NumpyRequiredForThisFeature)

    def test_create_numpy_vector_float32(self):
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            import numpy as np

            # Systems endian:
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, -12], dtype=np.float32)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,                     # vector length
                0, 0, 128, 63,                  # 1
                0, 0, 0, 64,                    # 2
                0, 0, 64, 193                   # -12
            ])

            # Reverse endian:
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,                     # vector length
                0, 0, 128, 63,                  # 1
                0, 0, 0, 64,                    # 2
                0, 0, 64, 193                   # -12
            ])

        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                NumpyRequiredForThisFeature)

    def test_create_numpy_vector_float64(self):
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            import numpy as np

            # Systems endian:
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, -12], dtype=np.float64)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,                     # vector length
                0, 0, 0, 0, 0, 0, 240, 63,                  # 1
                0, 0, 0, 0, 0, 0, 0, 64,                    # 2
                0, 0, 0, 0, 0, 0, 40, 192                   # -12
            ])

            # Reverse endian:
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0,                     # vector length
                0, 0, 0, 0, 0, 0, 240, 63,                  # 1
                0, 0, 0, 0, 0, 0, 0, 64,                    # 2
                0, 0, 0, 0, 0, 0, 40, 192                   # -12
            ])

        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                NumpyRequiredForThisFeature)

    def test_create_numpy_vector_bool(self):
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            import numpy as np

            # Systems endian:
            b = flatbuffers.Builder(0)
            x = np.array([True, False, True], dtype=np.bool)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0, # vector length
                1, 0, 1, 0  # vector values + padding
            ])

            # Reverse endian:
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [
                3, 0, 0, 0, # vector length
                1, 0, 1, 0  # vector values + padding
            ])

        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                NumpyRequiredForThisFeature)

    def test_create_numpy_vector_reject_strings(self):
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            import numpy as np

            # Create String array
            b = flatbuffers.Builder(0)
            x = np.array(["hello", "fb", "testing"])
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                TypeError)

        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                NumpyRequiredForThisFeature)

    def test_create_numpy_vector_reject_object(self):
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            import numpy as np

            # Create String array
            b = flatbuffers.Builder(0)
            x = np.array([{"m": 0}, {"as": -2.1, 'c': 'c'}])
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                TypeError)

        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(
                self,
                lambda: b.CreateNumpyVector(x),
                NumpyRequiredForThisFeature)

    def test_empty_vtable(self):
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        self.assertBuilderEquals(b, [])
        b.EndObject()
        self.assertBuilderEquals(b, [4, 0, 4, 0, 4, 0, 0, 0])

    def test_vtable_with_one_true_bool(self):
        b = flatbuffers.Builder(0)
        self.assertBuilderEquals(b, [])
        b.StartObject(1)
        self.assertBuilderEquals(b, [])
        b.PrependBoolSlot(0, True, False)
        b.EndObject()
        self.assertBuilderEquals(b, [
            6, 0,  # vtable bytes
            8, 0,  # length of object including vtable offset
            7, 0,  # start of bool value
            6, 0, 0, 0,  # offset for start of vtable (int32)
            0, 0, 0,  # padded to 4 bytes
            1,  # bool value
        ])

    def test_vtable_with_one_default_bool(self):
        b = flatbuffers.Builder(0)
        self.assertBuilderEquals(b, [])
        b.StartObject(1)
        self.assertBuilderEquals(b, [])
        b.PrependBoolSlot(0, False, False)
        b.EndObject()
        self.assertBuilderEquals(b, [
            4, 0,  # vtable bytes
            4, 0,  # end of object from here
            # entry 1 is zero and not stored
            4, 0, 0, 0,  # offset for start of vtable (int32)
        ])

    def test_vtable_with_one_int16(self):
        b = flatbuffers.Builder(0)
        b.StartObject(1)
        b.PrependInt16Slot(0, 0x789A, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [
            6, 0,  # vtable bytes
            8, 0,  # end of object from here
            6, 0,  # offset to value
            6, 0, 0, 0,  # offset for start of vtable (int32)
            0, 0,  # padding to 4 bytes
            0x9A, 0x78,
        ])

    def test_vtable_with_two_int16(self):
        b = flatbuffers.Builder(0)
        b.StartObject(2)
        b.PrependInt16Slot(0, 0x3456, 0)
        b.PrependInt16Slot(1, 0x789A, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [
            8, 0,  # vtable bytes
            8, 0,  # end of object from here
            6, 0,  # offset to value 0
            4, 0,  # offset to value 1
            8, 0, 0, 0,  # offset for start of vtable (int32)
            0x9A, 0x78,  # value 1
            0x56, 0x34,  # value 0
        ])

    def test_vtable_with_int16_and_bool(self):
        b = flatbuffers.Builder(0)
        b.StartObject(2)
        b.PrependInt16Slot(0, 0x3456, 0)
        b.PrependBoolSlot(1, True, False)
        b.EndObject()
        self.assertBuilderEquals(b, [
            8, 0,  # vtable bytes
            8, 0,  # end of object from here
            6, 0,  # offset to value 0
            5, 0,  # offset to value 1
            8, 0, 0, 0,  # offset for start of vtable (int32)
            0,          # padding
            1,          # value 1
            0x56, 0x34,  # value 0
        ])

    def test_vtable_with_empty_vector(self):
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint8Flags.bytewidth, 0, 1)
        vecend = b.EndVector(0)
        b.StartObject(1)
        b.PrependUOffsetTRelativeSlot(0, vecend, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [
            6, 0,  # vtable bytes
            8, 0,
            4, 0,  # offset to vector offset
            6, 0, 0, 0,  # offset for start of vtable (int32)
            4, 0, 0, 0,
            0, 0, 0, 0,  # length of vector (not in struct)
        ])

    def test_vtable_with_empty_vector_of_byte_and_some_scalars(self):
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint8Flags.bytewidth, 0, 1)
        vecend = b.EndVector(0)
        b.StartObject(2)
        b.PrependInt16Slot(0, 55, 0)
        b.PrependUOffsetTRelativeSlot(1, vecend, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [
            8, 0,  # vtable bytes
            12, 0,
            10, 0,  # offset to value 0
            4, 0,  # offset to vector offset
            8, 0, 0, 0,  # vtable loc
            8, 0, 0, 0,  # value 1
            0, 0, 55, 0,  # value 0

            0, 0, 0, 0,  # length of vector (not in struct)
        ])

    def test_vtable_with_1_int16_and_2vector_of_int16(self):
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Int16Flags.bytewidth, 2, 1)
        b.PrependInt16(0x1234)
        b.PrependInt16(0x5678)
        vecend = b.EndVector(2)
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(1, vecend, 0)
        b.PrependInt16Slot(0, 55, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [
            8, 0,  # vtable bytes
            12, 0,  # length of object
            6, 0,  # start of value 0 from end of vtable
            8, 0,  # start of value 1 from end of buffer
            8, 0, 0, 0,  # offset for start of vtable (int32)
            0, 0,  # padding
            55, 0,  # value 0
            4, 0, 0, 0,  # vector position from here
            2, 0, 0, 0,  # length of vector (uint32)
            0x78, 0x56,  # vector value 1
            0x34, 0x12,  # vector value 0
        ])

    def test_vtable_with_1_struct_of_1_int8__1_int16__1_int32(self):
        b = flatbuffers.Builder(0)
        b.StartObject(1)
        b.Prep(4+4+4, 0)
        b.PrependInt8(55)
        b.Pad(3)
        b.PrependInt16(0x1234)
        b.Pad(2)
        b.PrependInt32(0x12345678)
        structStart = b.Offset()
        b.PrependStructSlot(0, structStart, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [
            6, 0,  # vtable bytes
            16, 0,  # end of object from here
            4, 0,  # start of struct from here
            6, 0, 0, 0,  # offset for start of vtable (int32)
            0x78, 0x56, 0x34, 0x12,  # value 2
            0, 0,  # padding
            0x34, 0x12,  # value 1
            0, 0, 0,  # padding
            55,  # value 0
        ])

    def test_vtable_with_1_vector_of_2_struct_of_2_int8(self):
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Int8Flags.bytewidth*2, 2, 1)
        b.PrependInt8(33)
        b.PrependInt8(44)
        b.PrependInt8(55)
        b.PrependInt8(66)
        vecend = b.EndVector(2)
        b.StartObject(1)
        b.PrependUOffsetTRelativeSlot(0, vecend, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [
            6, 0,  # vtable bytes
            8, 0,
            4, 0,  # offset of vector offset
            6, 0, 0, 0,  # offset for start of vtable (int32)
            4, 0, 0, 0,  # vector start offset

            2, 0, 0, 0,  # vector length
            66,  # vector value 1,1
            55,  # vector value 1,0
            44,  # vector value 0,1
            33,  # vector value 0,0
        ])

    def test_table_with_some_elements(self):
        b = flatbuffers.Builder(0)
        b.StartObject(2)
        b.PrependInt8Slot(0, 33, 0)
        b.PrependInt16Slot(1, 66, 0)
        off = b.EndObject()
        b.Finish(off)

        self.assertBuilderEquals(b, [
            12, 0, 0, 0,  # root of table: points to vtable offset

            8, 0,  # vtable bytes
            8, 0,  # end of object from here
            7, 0,  # start of value 0
            4, 0,  # start of value 1

            8, 0, 0, 0,  # offset for start of vtable (int32)

            66, 0,  # value 1
            0,  # padding
            33,  # value 0
        ])

    def test__one_unfinished_table_and_one_finished_table(self):
        b = flatbuffers.Builder(0)
        b.StartObject(2)
        b.PrependInt8Slot(0, 33, 0)
        b.PrependInt8Slot(1, 44, 0)
        off = b.EndObject()
        b.Finish(off)

        b.StartObject(3)
        b.PrependInt8Slot(0, 55, 0)
        b.PrependInt8Slot(1, 66, 0)
        b.PrependInt8Slot(2, 77, 0)
        off = b.EndObject()
        b.Finish(off)

        self.assertBuilderEquals(b, [
            16, 0, 0, 0,  # root of table: points to object
            0, 0,  # padding

            10, 0,  # vtable bytes
            8, 0,  # size of object
            7, 0,  # start of value 0
            6, 0,  # start of value 1
            5, 0,  # start of value 2
            10, 0, 0, 0,  # offset for start of vtable (int32)
            0,  # padding
            77,  # value 2
            66,  # value 1
            55,  # value 0

            12, 0, 0, 0,  # root of table: points to object

            8, 0,  # vtable bytes
            8, 0,  # size of object
            7, 0,  # start of value 0
            6, 0,  # start of value 1
            8, 0, 0, 0,  # offset for start of vtable (int32)
            0, 0,  # padding
            44,  # value 1
            33,  # value 0
        ])

    def test_a_bunch_of_bools(self):
        b = flatbuffers.Builder(0)
        b.StartObject(8)
        b.PrependBoolSlot(0, True, False)
        b.PrependBoolSlot(1, True, False)
        b.PrependBoolSlot(2, True, False)
        b.PrependBoolSlot(3, True, False)
        b.PrependBoolSlot(4, True, False)
        b.PrependBoolSlot(5, True, False)
        b.PrependBoolSlot(6, True, False)
        b.PrependBoolSlot(7, True, False)
        off = b.EndObject()
        b.Finish(off)

        self.assertBuilderEquals(b, [
            24, 0, 0, 0,  # root of table: points to vtable offset

            20, 0,  # vtable bytes
            12, 0,  # size of object
            11, 0,  # start of value 0
            10, 0,  # start of value 1
            9, 0,  # start of value 2
            8, 0,  # start of value 3
            7, 0,  # start of value 4
            6, 0,  # start of value 5
            5, 0,  # start of value 6
            4, 0,  # start of value 7
            20, 0, 0, 0,  # vtable offset

            1,  # value 7
            1,  # value 6
            1,  # value 5
            1,  # value 4
            1,  # value 3
            1,  # value 2
            1,  # value 1
            1,  # value 0
        ])

    def test_three_bools(self):
        b = flatbuffers.Builder(0)
        b.StartObject(3)
        b.PrependBoolSlot(0, True, False)
        b.PrependBoolSlot(1, True, False)
        b.PrependBoolSlot(2, True, False)
        off = b.EndObject()
        b.Finish(off)

        self.assertBuilderEquals(b, [
            16, 0, 0, 0,  # root of table: points to vtable offset

            0, 0,  # padding

            10, 0,  # vtable bytes
            8, 0,  # size of object
            7, 0,  # start of value 0
            6, 0,  # start of value 1
            5, 0,  # start of value 2
            10, 0, 0, 0,  # vtable offset from here

            0,  # padding
            1,  # value 2
            1,  # value 1
            1,  # value 0
        ])

    def test_some_floats(self):
        b = flatbuffers.Builder(0)
        b.StartObject(1)
        b.PrependFloat32Slot(0, 1.0, 0.0)
        off = b.EndObject()

        self.assertBuilderEquals(b, [
            6, 0,  # vtable bytes
            8, 0,  # size of object
            4, 0,  # start of value 0
            6, 0, 0, 0,  # vtable offset

            0, 0, 128, 63,  # value 0
        ])


def make_monster_from_generated_code(sizePrefix = False, file_identifier=None):
    ''' Use generated code to build the example Monster. '''

    b = flatbuffers.Builder(0)
    string = b.CreateString("MyMonster")
    test1 = b.CreateString("test1")
    test2 = b.CreateString("test2")
    fred = b.CreateString("Fred")

    MyGame.Example.Monster.MonsterStartInventoryVector(b, 5)
    b.PrependByte(4)
    b.PrependByte(3)
    b.PrependByte(2)
    b.PrependByte(1)
    b.PrependByte(0)
    inv = b.EndVector(5)

    MyGame.Example.Monster.MonsterStart(b)
    MyGame.Example.Monster.MonsterAddName(b, fred)
    mon2 = MyGame.Example.Monster.MonsterEnd(b)

    MyGame.Example.Monster.MonsterStartTest4Vector(b, 2)
    MyGame.Example.Test.CreateTest(b, 10, 20)
    MyGame.Example.Test.CreateTest(b, 30, 40)
    test4 = b.EndVector(2)

    MyGame.Example.Monster.MonsterStartTestarrayofstringVector(b, 2)
    b.PrependUOffsetTRelative(test2)
    b.PrependUOffsetTRelative(test1)
    testArrayOfString = b.EndVector(2)

    MyGame.Example.Monster.MonsterStartVectorOfLongsVector(b, 5)
    b.PrependInt64(100000000)
    b.PrependInt64(1000000)
    b.PrependInt64(10000)
    b.PrependInt64(100)
    b.PrependInt64(1)
    VectorOfLongs = b.EndVector(5)

    MyGame.Example.Monster.MonsterStartVectorOfDoublesVector(b, 3)
    b.PrependFloat64(1.7976931348623157e+308)
    b.PrependFloat64(0)
    b.PrependFloat64(-1.7976931348623157e+308)
    VectorOfDoubles = b.EndVector(3)

    MyGame.Example.Monster.MonsterStart(b)

    pos = MyGame.Example.Vec3.CreateVec3(b, 1.0, 2.0, 3.0, 3.0, 2, 5, 6)
    MyGame.Example.Monster.MonsterAddPos(b, pos)

    MyGame.Example.Monster.MonsterAddHp(b, 80)
    MyGame.Example.Monster.MonsterAddName(b, string)
    MyGame.Example.Monster.MonsterAddInventory(b, inv)
    MyGame.Example.Monster.MonsterAddTestType(b, 1)
    MyGame.Example.Monster.MonsterAddTest(b, mon2)
    MyGame.Example.Monster.MonsterAddTest4(b, test4)
    MyGame.Example.Monster.MonsterAddTestarrayofstring(b, testArrayOfString)
    MyGame.Example.Monster.MonsterAddVectorOfLongs(b, VectorOfLongs)
    MyGame.Example.Monster.MonsterAddVectorOfDoubles(b, VectorOfDoubles)
    mon = MyGame.Example.Monster.MonsterEnd(b)

    if sizePrefix:
        b.FinishSizePrefixed(mon, file_identifier)
    else:
        b.Finish(mon, file_identifier)

    return b.Bytes, b.Head()


class TestAllCodePathsOfExampleSchema(unittest.TestCase):
    def setUp(self, *args, **kwargs):
        super(TestAllCodePathsOfExampleSchema, self).setUp(*args, **kwargs)

        b = flatbuffers.Builder(0)
        MyGame.Example.Monster.MonsterStart(b)
        gen_mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(gen_mon)

        self.mon = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Bytes,
                                                                   b.Head())

    def test_default_monster_pos(self):
        self.assertTrue(self.mon.Pos() is None)

    def test_nondefault_monster_mana(self):
        b = flatbuffers.Builder(0)
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddMana(b, 50)
        mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(mon)

        got_mon = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Bytes,
                                                                  b.Head())
        self.assertEqual(50, got_mon.Mana())

    def test_default_monster_hp(self):
        self.assertEqual(100, self.mon.Hp())

    def test_default_monster_name(self):
        self.assertEqual(None, self.mon.Name())

    def test_default_monster_inventory_item(self):
        self.assertEqual(0, self.mon.Inventory(0))

    def test_default_monster_inventory_length(self):
        self.assertEqual(0, self.mon.InventoryLength())

    def test_default_monster_color(self):
        self.assertEqual(MyGame.Example.Color.Color.Blue, self.mon.Color())

    def test_nondefault_monster_color(self):
        b = flatbuffers.Builder(0)
        color = MyGame.Example.Color.Color.Red
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddColor(b, color)
        mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(mon)

        mon2 = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Bytes,
                                                               b.Head())
        self.assertEqual(MyGame.Example.Color.Color.Red, mon2.Color())

    def test_default_monster_testtype(self):
        self.assertEqual(0, self.mon.TestType())

    def test_default_monster_test_field(self):
        self.assertEqual(None, self.mon.Test())

    def test_default_monster_test4_item(self):
        self.assertEqual(None, self.mon.Test4(0))

    def test_default_monster_test4_length(self):
        self.assertEqual(0, self.mon.Test4Length())

    def test_default_monster_testarrayofstring(self):
        self.assertEqual("", self.mon.Testarrayofstring(0))

    def test_default_monster_testarrayofstring_length(self):
        self.assertEqual(0, self.mon.TestarrayofstringLength())

    def test_default_monster_testarrayoftables(self):
        self.assertEqual(None, self.mon.Testarrayoftables(0))

    def test_nondefault_monster_testarrayoftables(self):
        b = flatbuffers.Builder(0)

        # make a child Monster within a vector of Monsters:
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddHp(b, 99)
        sub_monster = MyGame.Example.Monster.MonsterEnd(b)

        # build the vector:
        MyGame.Example.Monster.MonsterStartTestarrayoftablesVector(b, 1)
        b.PrependUOffsetTRelative(sub_monster)
        vec = b.EndVector(1)

        # make the parent monster and include the vector of Monster:
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddTestarrayoftables(b, vec)
        mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(mon)

        # inspect the resulting data:
        mon2 = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Output(), 0)
        self.assertEqual(99, mon2.Testarrayoftables(0).Hp())
        self.assertEqual(1, mon2.TestarrayoftablesLength())

    def test_default_monster_testarrayoftables_length(self):
        self.assertEqual(0, self.mon.TestarrayoftablesLength())

    def test_nondefault_monster_enemy(self):
        b = flatbuffers.Builder(0)

        # make an Enemy object:
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddHp(b, 88)
        enemy = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(enemy)

        # make the parent monster and include the vector of Monster:
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddEnemy(b, enemy)
        mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(mon)

        # inspect the resulting data:
        mon2 = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Bytes,
                                                               b.Head())
        self.assertEqual(88, mon2.Enemy().Hp())

    def test_default_monster_testnestedflatbuffer(self):
        self.assertEqual(0, self.mon.Testnestedflatbuffer(0))

    def test_default_monster_testnestedflatbuffer_length(self):
        self.assertEqual(0, self.mon.TestnestedflatbufferLength())

    def test_nondefault_monster_testnestedflatbuffer(self):
        b = flatbuffers.Builder(0)

        MyGame.Example.Monster.MonsterStartTestnestedflatbufferVector(b, 3)
        b.PrependByte(4)
        b.PrependByte(2)
        b.PrependByte(0)
        sub_buf = b.EndVector(3)

        # make the parent monster and include the vector of Monster:
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddTestnestedflatbuffer(b, sub_buf)
        mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(mon)

        # inspect the resulting data:
        mon2 = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Bytes,
                                                               b.Head())
        self.assertEqual(3, mon2.TestnestedflatbufferLength())
        self.assertEqual(0, mon2.Testnestedflatbuffer(0))
        self.assertEqual(2, mon2.Testnestedflatbuffer(1))
        self.assertEqual(4, mon2.Testnestedflatbuffer(2))
        try:
            imp.find_module('numpy')
            # if numpy exists, then we should be able to get the
            # vector as a numpy array
            self.assertEqual([0, 2, 4], mon2.TestnestedflatbufferAsNumpy().tolist())
        except ImportError:
            assertRaises(self,
                         lambda: mon2.TestnestedflatbufferAsNumpy(),
                         NumpyRequiredForThisFeature)

    def test_nondefault_monster_testempty(self):
        b = flatbuffers.Builder(0)

        # make a Stat object:
        MyGame.Example.Stat.StatStart(b)
        MyGame.Example.Stat.StatAddVal(b, 123)
        my_stat = MyGame.Example.Stat.StatEnd(b)
        b.Finish(my_stat)

        # include the stat object in a monster:
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddTestempty(b, my_stat)
        mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(mon)

        # inspect the resulting data:
        mon2 = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Bytes,
                                                               b.Head())
        self.assertEqual(123, mon2.Testempty().Val())

    def test_default_monster_testbool(self):
        self.assertFalse(self.mon.Testbool())

    def test_nondefault_monster_testbool(self):
        b = flatbuffers.Builder(0)
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddTestbool(b, True)
        mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(mon)

        # inspect the resulting data:
        mon2 = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Bytes,
                                                               b.Head())
        self.assertTrue(mon2.Testbool())

    def test_default_monster_testhashes(self):
        self.assertEqual(0, self.mon.Testhashs32Fnv1())
        self.assertEqual(0, self.mon.Testhashu32Fnv1())
        self.assertEqual(0, self.mon.Testhashs64Fnv1())
        self.assertEqual(0, self.mon.Testhashu64Fnv1())
        self.assertEqual(0, self.mon.Testhashs32Fnv1a())
        self.assertEqual(0, self.mon.Testhashu32Fnv1a())
        self.assertEqual(0, self.mon.Testhashs64Fnv1a())
        self.assertEqual(0, self.mon.Testhashu64Fnv1a())

    def test_nondefault_monster_testhashes(self):
        b = flatbuffers.Builder(0)
        MyGame.Example.Monster.MonsterStart(b)
        MyGame.Example.Monster.MonsterAddTesthashs32Fnv1(b, 1)
        MyGame.Example.Monster.MonsterAddTesthashu32Fnv1(b, 2)
        MyGame.Example.Monster.MonsterAddTesthashs64Fnv1(b, 3)
        MyGame.Example.Monster.MonsterAddTesthashu64Fnv1(b, 4)
        MyGame.Example.Monster.MonsterAddTesthashs32Fnv1a(b, 5)
        MyGame.Example.Monster.MonsterAddTesthashu32Fnv1a(b, 6)
        MyGame.Example.Monster.MonsterAddTesthashs64Fnv1a(b, 7)
        MyGame.Example.Monster.MonsterAddTesthashu64Fnv1a(b, 8)
        mon = MyGame.Example.Monster.MonsterEnd(b)
        b.Finish(mon)

        # inspect the resulting data:
        mon2 = MyGame.Example.Monster.Monster.GetRootAsMonster(b.Bytes,
                                                               b.Head())
        self.assertEqual(1, mon2.Testhashs32Fnv1())
        self.assertEqual(2, mon2.Testhashu32Fnv1())
        self.assertEqual(3, mon2.Testhashs64Fnv1())
        self.assertEqual(4, mon2.Testhashu64Fnv1())
        self.assertEqual(5, mon2.Testhashs32Fnv1a())
        self.assertEqual(6, mon2.Testhashu32Fnv1a())
        self.assertEqual(7, mon2.Testhashs64Fnv1a())
        self.assertEqual(8, mon2.Testhashu64Fnv1a())

    def test_getrootas_for_nonroot_table(self):
        b = flatbuffers.Builder(0)
        string = b.CreateString("MyStat")

        MyGame.Example.Stat.StatStart(b)
        MyGame.Example.Stat.StatAddId(b, string)
        MyGame.Example.Stat.StatAddVal(b, 12345678)
        MyGame.Example.Stat.StatAddCount(b, 12345)
        stat = MyGame.Example.Stat.StatEnd(b)
        b.Finish(stat)

        stat2 = MyGame.Example.Stat.Stat.GetRootAsStat(b.Bytes, b.Head())

        self.assertEqual(b"MyStat", stat2.Id())
        self.assertEqual(12345678, stat2.Val())
        self.assertEqual(12345, stat2.Count())


class TestAllCodePathsOfMonsterExtraSchema(unittest.TestCase):
    def setUp(self, *args, **kwargs):
        super(TestAllCodePathsOfMonsterExtraSchema, self).setUp(*args, **kwargs)

        b = flatbuffers.Builder(0)
        MyGame.MonsterExtra.MonsterExtraStart(b)
        gen_mon = MyGame.MonsterExtra.MonsterExtraEnd(b)
        b.Finish(gen_mon)

        self.mon = MyGame.MonsterExtra.MonsterExtra.GetRootAsMonsterExtra(b.Bytes, b.Head())

    def test_default_nan_inf(self):
        self.assertTrue(math.isnan(self.mon.F1()))
        self.assertEqual(self.mon.F2(), float("inf"))
        self.assertEqual(self.mon.F3(), float("-inf"))

        self.assertTrue(math.isnan(self.mon.D1()))
        self.assertEqual(self.mon.D2(), float("inf"))
        self.assertEqual(self.mon.D3(), float("-inf"))


class TestVtableDeduplication(unittest.TestCase):
    ''' TestVtableDeduplication verifies that vtables are deduplicated. '''

    def test_vtable_deduplication(self):
        b = flatbuffers.Builder(0)

        b.StartObject(4)
        b.PrependByteSlot(0, 0, 0)
        b.PrependByteSlot(1, 11, 0)
        b.PrependByteSlot(2, 22, 0)
        b.PrependInt16Slot(3, 33, 0)
        obj0 = b.EndObject()

        b.StartObject(4)
        b.PrependByteSlot(0, 0, 0)
        b.PrependByteSlot(1, 44, 0)
        b.PrependByteSlot(2, 55, 0)
        b.PrependInt16Slot(3, 66, 0)
        obj1 = b.EndObject()

        b.StartObject(4)
        b.PrependByteSlot(0, 0, 0)
        b.PrependByteSlot(1, 77, 0)
        b.PrependByteSlot(2, 88, 0)
        b.PrependInt16Slot(3, 99, 0)
        obj2 = b.EndObject()

        got = b.Bytes[b.Head():]

        want = bytearray([
            240, 255, 255, 255,  # == -12. offset to dedupped vtable.
            99, 0,
            88,
            77,
            248, 255, 255, 255,  # == -8. offset to dedupped vtable.
            66, 0,
            55,
            44,
            12, 0,
            8, 0,
            0, 0,
            7, 0,
            6, 0,
            4, 0,
            12, 0, 0, 0,
            33, 0,
            22,
            11,
        ])

        self.assertEqual((len(want), want), (len(got), got))

        table0 = flatbuffers.table.Table(b.Bytes, len(b.Bytes) - obj0)
        table1 = flatbuffers.table.Table(b.Bytes, len(b.Bytes) - obj1)
        table2 = flatbuffers.table.Table(b.Bytes, len(b.Bytes) - obj2)

        def _checkTable(tab, voffsett_value, b, c, d):
            # vtable size
            got = tab.GetVOffsetTSlot(0, 0)
            self.assertEqual(12, got, 'case 0, 0')

            # object size
            got = tab.GetVOffsetTSlot(2, 0)
            self.assertEqual(8, got, 'case 2, 0')

            # default value
            got = tab.GetVOffsetTSlot(4, 0)
            self.assertEqual(voffsett_value, got, 'case 4, 0')

            got = tab.GetSlot(6, 0, N.Uint8Flags)
            self.assertEqual(b, got, 'case 6, 0')

            val = tab.GetSlot(8, 0, N.Uint8Flags)
            self.assertEqual(c, val, 'failed 8, 0')

            got = tab.GetSlot(10, 0, N.Uint8Flags)
            self.assertEqual(d, got, 'failed 10, 0')

        _checkTable(table0, 0, 11, 22, 33)
        _checkTable(table1, 0, 44, 55, 66)
        _checkTable(table2, 0, 77, 88, 99)


class TestExceptions(unittest.TestCase):
    def test_object_is_nested_error(self):
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        assertRaises(self, lambda: b.StartObject(0),
                     flatbuffers.builder.IsNestedError)

    def test_object_is_not_nested_error(self):
        b = flatbuffers.Builder(0)
        assertRaises(self, lambda: b.EndObject(),
                     flatbuffers.builder.IsNotNestedError)

    def test_struct_is_not_inline_error(self):
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        assertRaises(self, lambda: b.PrependStructSlot(0, 1, 0),
                     flatbuffers.builder.StructIsNotInlineError)

    def test_unreachable_error(self):
        b = flatbuffers.Builder(0)
        assertRaises(self, lambda: b.PrependUOffsetTRelative(1),
                     flatbuffers.builder.OffsetArithmeticError)

    def test_create_string_is_nested_error(self):
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        s = 'test1'
        assertRaises(self, lambda: b.CreateString(s),
                     flatbuffers.builder.IsNestedError)

    def test_create_byte_vector_is_nested_error(self):
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        s = b'test1'
        assertRaises(self, lambda: b.CreateByteVector(s),
                     flatbuffers.builder.IsNestedError)

    def test_finished_bytes_error(self):
        b = flatbuffers.Builder(0)
        assertRaises(self, lambda: b.Output(),
                     flatbuffers.builder.BuilderNotFinishedError)


class TestFixedLengthArrays(unittest.TestCase):
    def test_fixed_length_array(self):
        builder = flatbuffers.Builder(0)

        a = 0.5
        b = range(0, 15)
        c = 1
        d_a = [[1, 2], [3, 4]]
        d_b = [MyGame.Example.TestEnum.TestEnum.B, \
                MyGame.Example.TestEnum.TestEnum.C]
        d_c = [[MyGame.Example.TestEnum.TestEnum.A, \
                MyGame.Example.TestEnum.TestEnum.B], \
                [MyGame.Example.TestEnum.TestEnum.C, \
                 MyGame.Example.TestEnum.TestEnum.B]]
        d_d = [[-1, 1], [-2, 2]]
        e = 2
        f = [-1, 1]

        arrayOffset = MyGame.Example.ArrayStruct.CreateArrayStruct(builder, \
            a, b, c, d_a, d_b, d_c, d_d, e, f)

        # Create a table with the ArrayStruct.
        MyGame.Example.ArrayTable.ArrayTableStart(builder)
        MyGame.Example.ArrayTable.ArrayTableAddA(builder, arrayOffset)
        tableOffset = MyGame.Example.ArrayTable.ArrayTableEnd(builder)

        builder.Finish(tableOffset)

        buf = builder.Output()

        table = MyGame.Example.ArrayTable.ArrayTable.GetRootAsArrayTable(buf, 0)

        # Verify structure.
        nested = MyGame.Example.NestedStruct.NestedStruct()
        self.assertEqual(table.A().A(), 0.5)
        self.assertEqual(table.A().B(), \
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        self.assertEqual(table.A().C(), 1)
        self.assertEqual(table.A().D(nested, 0).A(), [1, 2])
        self.assertEqual(table.A().D(nested, 1).A(), [3, 4])
        self.assertEqual(table.A().D(nested, 0).B(), \
            MyGame.Example.TestEnum.TestEnum.B)
        self.assertEqual(table.A().D(nested, 1).B(), \
            MyGame.Example.TestEnum.TestEnum.C)
        self.assertEqual(table.A().D(nested, 0).C(), \
            [MyGame.Example.TestEnum.TestEnum.A, \
             MyGame.Example.TestEnum.TestEnum.B])
        self.assertEqual(table.A().D(nested, 1).C(), \
            [MyGame.Example.TestEnum.TestEnum.C, \
             MyGame.Example.TestEnum.TestEnum.B])
        self.assertEqual(table.A().D(nested, 0).D(), [-1, 1])
        self.assertEqual(table.A().D(nested, 1).D(), [-2, 2])
        self.assertEqual(table.A().E(), 2)
        self.assertEqual(table.A().F(), [-1, 1])


def CheckAgainstGoldDataGo():
    try:
        gen_buf, gen_off = make_monster_from_generated_code()
        fn = 'monsterdata_go_wire.mon'
        if not os.path.exists(fn):
            print('Go-generated data does not exist, failed.')
            return False

        # would like to use a context manager here, but it's less
        # backwards-compatible:
        f = open(fn, 'rb')
        go_wire_data = f.read()
        f.close()

        CheckReadBuffer(bytearray(go_wire_data), 0)
        if not bytearray(gen_buf[gen_off:]) == bytearray(go_wire_data):
            raise AssertionError('CheckAgainstGoldDataGo failed')
    except:
        print('Failed to test against Go-generated test data.')
        return False

    print('Can read Go-generated test data, and Python generates bytewise identical data.')
    return True


def CheckAgainstGoldDataJava():
    try:
        gen_buf, gen_off = make_monster_from_generated_code()
        fn = 'monsterdata_java_wire.mon'
        if not os.path.exists(fn):
            print('Java-generated data does not exist, failed.')
            return False
        f = open(fn, 'rb')
        java_wire_data = f.read()
        f.close()

        CheckReadBuffer(bytearray(java_wire_data), 0)
    except:
        print('Failed to read Java-generated test data.')
        return False

    print('Can read Java-generated test data.')
    return True


class LCG(object):
    ''' Include simple random number generator to ensure results will be the
        same cross platform.
        http://en.wikipedia.org/wiki/Park%E2%80%93Miller_random_number_generator '''

    __slots__ = ['n']

    InitialLCGSeed = 48271

    def __init__(self):
        self.n = self.InitialLCGSeed

    def Reset(self):
        self.n = self.InitialLCGSeed

    def Next(self):
        self.n = ((self.n * 279470273) % 4294967291) & 0xFFFFFFFF
        return self.n


def BenchmarkVtableDeduplication(count):
    '''
    BenchmarkVtableDeduplication measures the speed of vtable deduplication
    by creating `prePop` vtables, then populating `count` objects with a
    different single vtable.

    When count is large (as in long benchmarks), memory usage may be high.
    '''

    for prePop in (1, 10, 100, 1000):
        builder = flatbuffers.Builder(0)
        n = 1 + int(math.log(prePop, 1.5))

        # generate some layouts:
        layouts = set()
        r = list(compat_range(n))
        while len(layouts) < prePop:
            layouts.add(tuple(sorted(random.sample(r, int(max(1, n / 2))))))

        layouts = list(layouts)

        # pre-populate vtables:
        for layout in layouts:
            builder.StartObject(n)
            for j in layout:
                builder.PrependInt16Slot(j, j, 0)
            builder.EndObject()

        # benchmark deduplication of a new vtable:
        def f():
            layout = random.choice(layouts)
            builder.StartObject(n)
            for j in layout:
                builder.PrependInt16Slot(j, j, 0)
            builder.EndObject()

        duration = timeit.timeit(stmt=f, number=count)
        rate = float(count) / duration
        print(('vtable deduplication rate (n=%d, vtables=%d): %.2f sec' % (
            prePop,
            len(builder.vtables),
            rate))
        )


def BenchmarkCheckReadBuffer(count, buf, off):
    '''
    BenchmarkCheckReadBuffer measures the speed of flatbuffer reading
    by re-using the CheckReadBuffer function with the gold data.
    '''

    def f():
        CheckReadBuffer(buf, off)

    duration = timeit.timeit(stmt=f, number=count)
    rate = float(count) / duration
    data = float(len(buf) * count) / float(1024 * 1024)
    data_rate = data / float(duration)

    print(('traversed %d %d-byte flatbuffers in %.2fsec: %.2f/sec, %.2fMB/sec')
          % (count, len(buf), duration, rate, data_rate))


def BenchmarkMakeMonsterFromGeneratedCode(count, length):
    '''
    BenchmarkMakeMonsterFromGeneratedCode measures the speed of flatbuffer
    creation by re-using the make_monster_from_generated_code function for
    generating gold data examples.
    '''

    duration = timeit.timeit(stmt=make_monster_from_generated_code,
                             number=count)
    rate = float(count) / duration
    data = float(length * count) / float(1024 * 1024)
    data_rate = data / float(duration)

    print(('built %d %d-byte flatbuffers in %.2fsec: %.2f/sec, %.2fMB/sec' % \
           (count, length, duration, rate, data_rate)))


def backward_compatible_run_tests(**kwargs):
    if PY_VERSION < (2, 6):
        sys.stderr.write("Python version less than 2.6 are not supported")
        sys.stderr.flush()
        return False

    # python2.6 has a reduced-functionality unittest.main function:
    if PY_VERSION == (2, 6):
        try:
            unittest.main(**kwargs)
        except SystemExit as e:
            if not e.code == 0:
                return False
        return True

    # python2.7 and above let us not exit once unittest.main is run:
    kwargs['exit'] = False
    kwargs['verbosity'] = 0
    ret = unittest.main(**kwargs)
    if ret.result.errors or ret.result.failures:
        return False

    return True

def main():
    import os
    import sys
    if not len(sys.argv) == 4:
       sys.stderr.write('Usage: %s <benchmark vtable count>'
                        '<benchmark read count> <benchmark build count>\n'
                        % sys.argv[0])
       sys.stderr.write('       Provide COMPARE_GENERATED_TO_GO=1   to check'
                        'for bytewise comparison to Go data.\n')
       sys.stderr.write('       Provide COMPARE_GENERATED_TO_JAVA=1 to check'
                        'for bytewise comparison to Java data.\n')
       sys.stderr.flush()
       sys.exit(1)

    kwargs = dict(argv=sys.argv[:-3])

    # run tests, and run some language comparison checks if needed:
    success = backward_compatible_run_tests(**kwargs)
    if success and os.environ.get('COMPARE_GENERATED_TO_GO', 0) == "1":
        success = success and CheckAgainstGoldDataGo()
    if success and os.environ.get('COMPARE_GENERATED_TO_JAVA', 0) == "1":
        success = success and CheckAgainstGoldDataJava()

    if not success:
        sys.stderr.write('Tests failed, skipping benchmarks.\n')
        sys.stderr.flush()
        sys.exit(1)

    # run benchmarks (if 0, they will be a noop):
    bench_vtable = int(sys.argv[1])
    bench_traverse = int(sys.argv[2])
    bench_build = int(sys.argv[3])
    if bench_vtable:
        BenchmarkVtableDeduplication(bench_vtable)
    if bench_traverse:
        buf, off = make_monster_from_generated_code()
        BenchmarkCheckReadBuffer(bench_traverse, buf, off)
    if bench_build:
        buf, off = make_monster_from_generated_code()
        BenchmarkMakeMonsterFromGeneratedCode(bench_build, len(buf))

if __name__ == '__main__':
    main()
