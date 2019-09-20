/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;

namespace FlatBuffers.Test
{
    [FlatBuffersTestClass]
    public class ByteBufferTests
    {

        [FlatBuffersTestMethod]
        public void ByteBuffer_Length_MatchesBufferLength()
        {
            var buffer = new byte[1000];
            var uut = new ByteBuffer(buffer);
            Assert.AreEqual(buffer.Length, uut.Length);
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutBytePopulatesBufferAtZeroOffset()
        {
            var buffer = new byte[1];
            var uut = new ByteBuffer(buffer);
            uut.PutByte(0, (byte)99);

            Assert.AreEqual((byte)99, buffer[0]);
        }

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_PutByteCannotPutAtOffsetPastLength()
        {
            var uut = new ByteBuffer(1);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutByte(1, 99));
        }
#endif

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutShortPopulatesBufferCorrectly()
        {
            var buffer = new byte[2];
            var uut = new ByteBuffer(buffer);
            uut.PutShort(0, (short)1);

            // Ensure Endianness was written correctly
            Assert.AreEqual((byte)1, buffer[0]);
            Assert.AreEqual((byte)0, buffer[1]);
        }

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_PutShortCannotPutAtOffsetPastLength()
        {
            var uut = new ByteBuffer(2);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutShort(2, 99));
        }
#endif

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_PutShortChecksLength()
        {
            var uut = new ByteBuffer(1);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutShort(0, 99));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutShortChecksLengthAndOffset()
        {
            var uut = new ByteBuffer(2);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutShort(1, 99));
        }
#endif

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutIntPopulatesBufferCorrectly()
        {
            var buffer = new byte[4];
            var uut = new ByteBuffer(buffer);
            uut.PutInt(0, 0x0A0B0C0D);

            // Ensure Endianness was written correctly
            Assert.AreEqual(0x0D, buffer[0]);
            Assert.AreEqual(0x0C, buffer[1]);
            Assert.AreEqual(0x0B, buffer[2]);
            Assert.AreEqual(0x0A, buffer[3]);
        }

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_PutIntCannotPutAtOffsetPastLength()
        {
            var uut = new ByteBuffer(4);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutInt(2, 0x0A0B0C0D));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutIntChecksLength()
        {
            var uut = new ByteBuffer(1);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutInt(0, 0x0A0B0C0D));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutIntChecksLengthAndOffset()
        {
            var uut = new ByteBuffer(4);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutInt(2, 0x0A0B0C0D));
        }
#endif

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutLongPopulatesBufferCorrectly()
        {
            var buffer = new byte[8];
            var uut = new ByteBuffer(buffer);
            uut.PutLong(0, 0x010203040A0B0C0D);

            // Ensure Endianness was written correctly
            Assert.AreEqual(0x0D, buffer[0]);
            Assert.AreEqual(0x0C, buffer[1]);
            Assert.AreEqual(0x0B, buffer[2]);
            Assert.AreEqual(0x0A, buffer[3]);
            Assert.AreEqual(0x04, buffer[4]);
            Assert.AreEqual(0x03, buffer[5]);
            Assert.AreEqual(0x02, buffer[6]);
            Assert.AreEqual(0x01, buffer[7]);
        }

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_PutLongCannotPutAtOffsetPastLength()
        {
            var uut = new ByteBuffer(8);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutLong(2, 0x010203040A0B0C0D));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutLongChecksLength()
        {
            var uut = new ByteBuffer(1);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutLong(0, 0x010203040A0B0C0D));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_PutLongChecksLengthAndOffset()
        {
            var uut = new ByteBuffer(8);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.PutLong(2, 0x010203040A0B0C0D));
        }
#endif

        [FlatBuffersTestMethod]
        public void ByteBuffer_GetByteReturnsCorrectData()
        {
            var buffer = new byte[1];
            buffer[0] = 99;
            var uut = new ByteBuffer(buffer);
            Assert.AreEqual((byte)99, uut.Get(0));
        }

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_GetByteChecksOffset()
        {
            var uut = new ByteBuffer(1);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.Get(1));
        }
#endif

        [FlatBuffersTestMethod]
        public void ByteBuffer_GetShortReturnsCorrectData()
        {
            var buffer = new byte[2];
            buffer[0] = 1;
            buffer[1] = 0;
            var uut = new ByteBuffer(buffer);
            Assert.AreEqual(1, uut.GetShort(0));
        }

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_GetShortChecksOffset()
        {
            var uut = new ByteBuffer(2);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.GetShort(2));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_GetShortChecksLength()
        {
            var uut = new ByteBuffer(2);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.GetShort(1));
        }
#endif

        [FlatBuffersTestMethod]
        public void ByteBuffer_GetIntReturnsCorrectData()
        {
            var buffer = new byte[4];
            buffer[0] = 0x0D;
            buffer[1] = 0x0C;
            buffer[2] = 0x0B;
            buffer[3] = 0x0A;
            var uut = new ByteBuffer(buffer);
            Assert.AreEqual(0x0A0B0C0D, uut.GetInt(0));
        }

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_GetIntChecksOffset()
        {
            var uut = new ByteBuffer(4);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.GetInt(4));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_GetIntChecksLength()
        {
            var uut = new ByteBuffer(2);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.GetInt(0));
        }
#endif

        [FlatBuffersTestMethod]
        public void ByteBuffer_GetLongReturnsCorrectData()
        {
            var buffer = new byte[8];
            buffer[0] = 0x0D;
            buffer[1] = 0x0C;
            buffer[2] = 0x0B;
            buffer[3] = 0x0A;
            buffer[4] = 0x04;
            buffer[5] = 0x03;
            buffer[6] = 0x02;
            buffer[7] = 0x01;
            var uut = new ByteBuffer(buffer);
            Assert.AreEqual(0x010203040A0B0C0D, uut.GetLong(0));
        }

#if !BYTEBUFFER_NO_BOUNDS_CHECK
        [FlatBuffersTestMethod]
        public void ByteBuffer_GetLongChecksOffset()
        {
            var uut = new ByteBuffer(8);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.GetLong(8));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_GetLongChecksLength()
        {
            var uut = new ByteBuffer(7);
            Assert.Throws<ArgumentOutOfRangeException>(() => uut.GetLong(0));
        }
#endif

        [FlatBuffersTestMethod]
        public void ByteBuffer_ReverseBytesUshort()
        {
            const ushort original = (ushort)0x1234U;
            var reverse = ByteBuffer.ReverseBytes(original);
            Assert.AreEqual(0x3412U, reverse);

            var rereverse = ByteBuffer.ReverseBytes(reverse);
            Assert.AreEqual(original, rereverse);
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_ReverseBytesUint()
        {
            const uint original = 0x12345678;
            var reverse = ByteBuffer.ReverseBytes(original);
            Assert.AreEqual(0x78563412U, reverse);

            var rereverse = ByteBuffer.ReverseBytes(reverse);
            Assert.AreEqual(original, rereverse);
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_ReverseBytesUlong()
        {
            const ulong original = 0x1234567890ABCDEFUL;
            var reverse = ByteBuffer.ReverseBytes(original);
            Assert.AreEqual(0xEFCDAB9078563412UL, reverse);

            var rereverse = ByteBuffer.ReverseBytes(reverse);
            Assert.AreEqual(original, rereverse);
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_ToFullArray_MatchesBuffer()
        {
            var buffer = new byte[4];
            buffer[0] = 0x0D;
            buffer[1] = 0x0C;
            buffer[2] = 0x0B;
            buffer[3] = 0x0A;
            var uut = new ByteBuffer(buffer);
            Assert.ArrayEqual(buffer, uut.ToFullArray());
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_ToSizedArray_MatchesBuffer()
        {
            var buffer = new byte[4];
            buffer[0] = 0x0D;
            buffer[1] = 0x0C;
            buffer[2] = 0x0B;
            buffer[3] = 0x0A;
            var uut = new ByteBuffer(buffer);
            Assert.ArrayEqual(buffer, uut.ToFullArray());
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Duplicate_MatchesBuffer()
        {
            var buffer = new byte[4];
            buffer[0] = 0x0D;
            buffer[1] = 0x0C;
            buffer[2] = 0x0B;
            buffer[3] = 0x0A;
            var uut = new ByteBuffer(buffer);
            Assert.AreEqual(0x0A0B0C0D, uut.GetInt(0));

            // Advance by two bytes
            uut.Position = 2; uut = uut.Duplicate();
            Assert.AreEqual(0x0A0B, uut.GetShort(2));

            // Advance by one more byte
            uut.Position = 1; uut = uut.Duplicate();
            Assert.AreEqual(0x0A, uut.Get(3));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_To_Array_Float()
        {
            const int len = 9;

            // Construct the data array
            var fData = new float[len];
            fData[0] = 1.0079F;
            fData[1] = 4.0026F;
            fData[2] = 6.941F;
            fData[3] = 9.0122F;
            fData[4] = 10.811F;
            fData[5] = 12.0107F;
            fData[6] = 14.0067F;
            fData[7] = 15.9994F;
            fData[8] = 18.9984F;

            // Tranfer it to a byte array
            var buffer = new byte[sizeof(float) * fData.Length];
            Buffer.BlockCopy(fData, 0, buffer, 0, buffer.Length);

            // Create the Byte Buffer from byte array
            var uut = new ByteBuffer(buffer);

            // Get the full array back out and ensure they are equivalent
            var bbArray = uut.ToArray<float>(0, len);
            Assert.ArrayEqual(fData, bbArray);

            // Get a portion of the full array back out and ensure the
            // subrange agrees
            var bbArray2 = uut.ToArray<float>(4, len - 1);
            Assert.AreEqual(bbArray2.Length, len - 1);
            for (int i = 1; i < len - 1; i++)
            {
                Assert.AreEqual(fData[i], bbArray2[i - 1]);
            }

            // Get a sub portion of the full array back out and ensure the
            // subrange agrees
            var bbArray3 = uut.ToArray<float>(8, len - 4);
            Assert.AreEqual(bbArray3.Length, len - 4);
            for (int i = 2; i < len - 4; i++)
            {
                Assert.AreEqual(fData[i], bbArray3[i - 2]);
            }
        }

        public void ByteBuffer_Put_Array_Helper<T>(T[] data, int typeSize)
            where T : struct
        {
            // Create the Byte Buffer
            var uut = new ByteBuffer(1024);

            // Put the data into the buffer and make sure the offset is
            // calculated correctly
            int nOffset = uut.Put(1024, data);
            Assert.AreEqual(1024 - typeSize * data.Length, nOffset);

            // Get the full array back out and ensure they are equivalent
            var bbArray = uut.ToArray<T>(nOffset, data.Length);
            Assert.ArrayEqual(data, bbArray);
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_Float()
        {
            const int len = 9;

            // Construct the data array
            var data = new float[len];
            data[0] = 1.0079F;
            data[1] = 4.0026F;
            data[2] = 6.941F;
            data[3] = 9.0122F;
            data[4] = 10.811F;
            data[5] = 12.0107F;
            data[6] = 14.0067F;
            data[7] = 15.9994F;
            data[8] = 18.9984F;

            ByteBuffer_Put_Array_Helper(data, sizeof(float));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_Double()
        {
            const int len = 9;

            // Construct the data array
            var data = new double[len];
            data[0] = 1.0079;
            data[1] = 4.0026;
            data[2] = 6.941;
            data[3] = 9.0122;
            data[4] = 10.811;
            data[5] = 12.0107;
            data[6] = 14.0067;
            data[7] = 15.9994;
            data[8] = 18.9984;

            ByteBuffer_Put_Array_Helper(data, sizeof(double));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_Int()
        {
            const int len = 9;

            // Construct the data array
            var data = new int[len];
            data[0] = 1;
            data[1] = 4;
            data[2] = 6;
            data[3] = 9;
            data[4] = 10;
            data[5] = 12;
            data[6] = 14;
            data[7] = 15;
            data[8] = 18;

            ByteBuffer_Put_Array_Helper(data, sizeof(int));
        }


        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_UInt()
        {
            const int len = 9;

            // Construct the data array
            var data = new uint[len];
            data[0] = 1;
            data[1] = 4;
            data[2] = 6;
            data[3] = 9;
            data[4] = 10;
            data[5] = 12;
            data[6] = 14;
            data[7] = 15;
            data[8] = 18;

            ByteBuffer_Put_Array_Helper(data, sizeof(uint));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_Bool()
        {
            const int len = 9;

            // Construct the data array
            var data = new bool[len];
            data[0] = true;
            data[1] = true;
            data[2] = false;
            data[3] = true;
            data[4] = false;
            data[5] = true;
            data[6] = true;
            data[7] = true;
            data[8] = false;

            ByteBuffer_Put_Array_Helper(data, sizeof(bool));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_Long()
        {
            const int len = 9;

            // Construct the data array
            var data = new long[len];
            data[0] = 1;
            data[1] = 4;
            data[2] = 6;
            data[3] = 9;
            data[4] = 10;
            data[5] = 12;
            data[6] = 14;
            data[7] = 15;
            data[8] = 18;

            ByteBuffer_Put_Array_Helper(data, sizeof(long));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_Byte()
        {
            const int len = 9;

            // Construct the data array
            var data = new byte[len];
            data[0] = 1;
            data[1] = 4;
            data[2] = 6;
            data[3] = 9;
            data[4] = 10;
            data[5] = 12;
            data[6] = 14;
            data[7] = 15;
            data[8] = 18;

            ByteBuffer_Put_Array_Helper(data, sizeof(byte));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_SByte()
        {
            const int len = 9;

            // Construct the data array
            var data = new sbyte[len];
            data[0] = 1;
            data[1] = 4;
            data[2] = 6;
            data[3] = 9;
            data[4] = 10;
            data[5] = 12;
            data[6] = 14;
            data[7] = 15;
            data[8] = 18;

            ByteBuffer_Put_Array_Helper(data, sizeof(sbyte));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_Null_Throws()
        {
            // Create the Byte Buffer
            var uut = new ByteBuffer(1024);

            // create a null array and try to put it into the buffer
            float[] data = null;
            Assert.Throws<ArgumentNullException>(() => uut.Put(1024, data));
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_Empty_Throws()
        {
            // Create the Byte Buffer
            var uut = new ByteBuffer(1024);

            // create an array of length == 0, and try to put it into the buffer
            float[] data = new float[0];
            Assert.Throws<ArgumentException>(() => uut.Put(1024, data));
        }

        private struct dummyStruct
        {
            int a;
            float b;
        }

        [FlatBuffersTestMethod]
        public void ByteBuffer_Put_Array_IncorrectType_Throws()
        {
            // Create the Byte Buffer
            var uut = new ByteBuffer(1024);

            // Create an array of dummy structures that shouldn't be
            // able to be put into the buffer
            var data = new dummyStruct[10];
            Assert.Throws<ArgumentException>(() => uut.Put(1024, data));
        }
    }
}
