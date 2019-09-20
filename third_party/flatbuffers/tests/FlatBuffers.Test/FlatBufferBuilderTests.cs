/*
 * Copyright 2016 Google Inc. All rights reserved.
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
    public class FlatBufferBuilderTests
    {
        private FlatBufferBuilder CreateBuffer(bool forceDefaults = true)
        {
            var fbb = new FlatBufferBuilder(16) {ForceDefaults = forceDefaults};
            fbb.StartTable(1);
            return fbb;
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddBool_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddBool(0, false, false);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(bool), endOffset-storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddSByte_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddSbyte(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(sbyte), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddByte_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddByte(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(byte), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddShort_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddShort(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(short), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddUShort_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddUshort(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(ushort), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddInt_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddInt(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(int), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddUInt_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddUint(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(uint), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddLong_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddLong(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(long), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddULong_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddUlong(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(ulong), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddFloat_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddFloat(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(float), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WithForceDefaults_WhenAddDouble_AndDefaultValue_OffsetIncreasesBySize()
        {
            var fbb = CreateBuffer();
            var storedOffset = fbb.Offset;
            fbb.AddDouble(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(sizeof(double), endOffset - storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddBool_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddBool(0, false, false);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddSByte_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddSbyte(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddByte_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddByte(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddShort_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddShort(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddUShort_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddUshort(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddInt_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddInt(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddUInt_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddUint(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddLong_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddLong(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddULong_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddUlong(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddFloat_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddFloat(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_WhenAddDouble_AndDefaultValue_OffsetIsUnchanged()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;
            fbb.AddDouble(0, 0, 0);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_Add_Array_Float()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;

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

            fbb.Add(data);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset + sizeof(float) * data.Length);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_Add_Array_Bool()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;

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

            fbb.Add(data);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset + sizeof(bool) * data.Length);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_Add_Array_Double()
        {
            var fbb = CreateBuffer(false);
            var storedOffset = fbb.Offset;

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

            fbb.Add(data);
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset + sizeof(double) * data.Length);
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_Add_Array_Null_Throws()
        {
            var fbb = CreateBuffer(false);

            // Construct the data array
            float[] data = null;

            Assert.Throws<ArgumentNullException>(() => fbb.Add(data));
        }

        [FlatBuffersTestMethod]
        public void FlatBufferBuilder_Add_Array_Empty_Noop()
        {
            var fbb = CreateBuffer(false);

            var storedOffset = fbb.Offset;

            // Construct an empty data array
            float[] data = new float[0];
            fbb.Add(data);

            // Make sure the offset didn't change since nothing
            // was really added
            var endOffset = fbb.Offset;
            Assert.AreEqual(endOffset, storedOffset);
        }
    }
}
