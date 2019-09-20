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

using System.IO;
using System.Text;
using MyGame.Example;

namespace FlatBuffers.Test
{
    [FlatBuffersTestClass]
    public class FlatBuffersExampleTests
    {
        public void RunTests()
        {
            CanCreateNewFlatBufferFromScratch();
            CanReadCppGeneratedWireFile();
            TestEnums();
        }

        [FlatBuffersTestMethod]
        public void CanCreateNewFlatBufferFromScratch()
        {
            CanCreateNewFlatBufferFromScratch(true);
            CanCreateNewFlatBufferFromScratch(false);
        }

        private void CanCreateNewFlatBufferFromScratch(bool sizePrefix)
        {
            // Second, let's create a FlatBuffer from scratch in C#, and test it also.
            // We use an initial size of 1 to exercise the reallocation algorithm,
            // normally a size larger than the typical FlatBuffer you generate would be
            // better for performance.
            var fbb = new FlatBufferBuilder(1);

            StringOffset[] names = { fbb.CreateString("Frodo"), fbb.CreateString("Barney"), fbb.CreateString("Wilma") };
            Offset<Monster>[] off = new Offset<Monster>[3];
            Monster.StartMonster(fbb);
            Monster.AddName(fbb, names[0]);
            off[0] = Monster.EndMonster(fbb);
            Monster.StartMonster(fbb);
            Monster.AddName(fbb, names[1]);
            off[1] = Monster.EndMonster(fbb);
            Monster.StartMonster(fbb);
            Monster.AddName(fbb, names[2]);
            off[2] = Monster.EndMonster(fbb);
            var sortMons = Monster.CreateSortedVectorOfMonster(fbb, off);

            // We set up the same values as monsterdata.json:

            var str = fbb.CreateString("MyMonster");
            var test1 = fbb.CreateString("test1");
            var test2 = fbb.CreateString("test2");


            Monster.StartInventoryVector(fbb, 5);
            for (int i = 4; i >= 0; i--)
            {
                fbb.AddByte((byte)i);
            }
            var inv = fbb.EndVector();

            var fred = fbb.CreateString("Fred");
            Monster.StartMonster(fbb);
            Monster.AddName(fbb, fred);
            var mon2 = Monster.EndMonster(fbb);

            Monster.StartTest4Vector(fbb, 2);
            MyGame.Example.Test.CreateTest(fbb, (short)10, (sbyte)20);
            MyGame.Example.Test.CreateTest(fbb, (short)30, (sbyte)40);
            var test4 = fbb.EndVector();

            Monster.StartTestarrayofstringVector(fbb, 2);
            fbb.AddOffset(test2.Value);
            fbb.AddOffset(test1.Value);
            var testArrayOfString = fbb.EndVector();

            Monster.StartMonster(fbb);
            Monster.AddPos(fbb, Vec3.CreateVec3(fbb, 1.0f, 2.0f, 3.0f, 3.0,
                                                     Color.Green, (short)5, (sbyte)6));
            Monster.AddHp(fbb, (short)80);
            Monster.AddName(fbb, str);
            Monster.AddInventory(fbb, inv);
            Monster.AddTestType(fbb, Any.Monster);
            Monster.AddTest(fbb, mon2.Value);
            Monster.AddTest4(fbb, test4);
            Monster.AddTestarrayofstring(fbb, testArrayOfString);
            Monster.AddTestbool(fbb, true);
            Monster.AddTestarrayoftables(fbb, sortMons);
            var mon = Monster.EndMonster(fbb);

            if (sizePrefix)
            {
                Monster.FinishSizePrefixedMonsterBuffer(fbb, mon);
            }
            else
            {
                Monster.FinishMonsterBuffer(fbb, mon);
            }

            // Dump to output directory so we can inspect later, if needed
#if ENABLE_SPAN_T
            var data = fbb.DataBuffer.ToSizedArray();
            string filename = @"Resources/monsterdata_cstest" + (sizePrefix ? "_sp" : "") + ".mon";
            File.WriteAllBytes(filename, data);
#else
            using (var ms = fbb.DataBuffer.ToMemoryStream(fbb.DataBuffer.Position, fbb.Offset))
            {
                var data = ms.ToArray();
                string filename = @"Resources/monsterdata_cstest" + (sizePrefix ? "_sp" : "") + ".mon";
                File.WriteAllBytes(filename, data);
            }
#endif

            // Remove the size prefix if necessary for further testing
            ByteBuffer dataBuffer = fbb.DataBuffer;
            if (sizePrefix)
            {
                Assert.AreEqual(ByteBufferUtil.GetSizePrefix(dataBuffer) + FlatBufferConstants.SizePrefixLength,
                                dataBuffer.Length - dataBuffer.Position);
                dataBuffer = ByteBufferUtil.RemoveSizePrefix(dataBuffer);
            }

            // Now assert the buffer
            TestBuffer(dataBuffer);

            //Attempt to mutate Monster fields and check whether the buffer has been mutated properly
            // revert to original values after testing
            Monster monster = Monster.GetRootAsMonster(dataBuffer);
            

            // mana is optional and does not exist in the buffer so the mutation should fail
            // the mana field should retain its default value
            Assert.AreEqual(monster.MutateMana((short)10), false);
            Assert.AreEqual(monster.Mana, (short)150);

            // Accessing a vector of sorted by the key tables
            Assert.AreEqual(monster.Testarrayoftables(0).Value.Name, "Barney");
            Assert.AreEqual(monster.Testarrayoftables(1).Value.Name, "Frodo");
            Assert.AreEqual(monster.Testarrayoftables(2).Value.Name, "Wilma");

            // Example of searching for a table by the key
            Assert.IsTrue(monster.TestarrayoftablesByKey("Frodo") != null);
            Assert.IsTrue(monster.TestarrayoftablesByKey("Barney") != null);
            Assert.IsTrue(monster.TestarrayoftablesByKey("Wilma") != null);

            // testType is an existing field and mutating it should succeed
            Assert.AreEqual(monster.TestType, Any.Monster);
            Assert.AreEqual(monster.MutateTestType(Any.NONE), true);
            Assert.AreEqual(monster.TestType, Any.NONE);
            Assert.AreEqual(monster.MutateTestType(Any.Monster), true);
            Assert.AreEqual(monster.TestType, Any.Monster);

            //mutate the inventory vector
            Assert.AreEqual(monster.MutateInventory(0, 1), true);
            Assert.AreEqual(monster.MutateInventory(1, 2), true);
            Assert.AreEqual(monster.MutateInventory(2, 3), true);
            Assert.AreEqual(monster.MutateInventory(3, 4), true);
            Assert.AreEqual(monster.MutateInventory(4, 5), true);

            for (int i = 0; i < monster.InventoryLength; i++)
            {
                Assert.AreEqual(monster.Inventory(i), i + 1);
            }

            //reverse mutation
            Assert.AreEqual(monster.MutateInventory(0, 0), true);
            Assert.AreEqual(monster.MutateInventory(1, 1), true);
            Assert.AreEqual(monster.MutateInventory(2, 2), true);
            Assert.AreEqual(monster.MutateInventory(3, 3), true);
            Assert.AreEqual(monster.MutateInventory(4, 4), true);

            // get a struct field and edit one of its fields
            Vec3 pos = (Vec3)monster.Pos;
            Assert.AreEqual(pos.X, 1.0f);
            pos.MutateX(55.0f);
            Assert.AreEqual(pos.X, 55.0f);
            pos.MutateX(1.0f);
            Assert.AreEqual(pos.X, 1.0f);

            TestBuffer(dataBuffer);
        }

        private void TestBuffer(ByteBuffer bb)
        {
            Monster monster = Monster.GetRootAsMonster(bb);

            Assert.AreEqual(80, monster.Hp);
            Assert.AreEqual(150, monster.Mana);
            Assert.AreEqual("MyMonster", monster.Name);

            var pos = monster.Pos.Value;
            Assert.AreEqual(1.0f, pos.X);
            Assert.AreEqual(2.0f, pos.Y);
            Assert.AreEqual(3.0f, pos.Z);

            Assert.AreEqual(3.0f, pos.Test1);
            Assert.AreEqual(Color.Green, pos.Test2);
            var t = (MyGame.Example.Test)pos.Test3;
            Assert.AreEqual((short)5, t.A);
            Assert.AreEqual((sbyte)6, t.B);

            Assert.AreEqual(Any.Monster, monster.TestType);

            var monster2 = monster.Test<Monster>().Value;
            Assert.AreEqual("Fred", monster2.Name);


            Assert.AreEqual(5, monster.InventoryLength);
            var invsum = 0;
            for (var i = 0; i < monster.InventoryLength; i++)
            {
                invsum += monster.Inventory(i);
            }
            Assert.AreEqual(10, invsum);

            // Get the inventory as an array and subtract the
            // sum to get it back to 0
            var inventoryArray = monster.GetInventoryArray();
            Assert.AreEqual(5, inventoryArray.Length);
            foreach(var inv in inventoryArray)
            {
                invsum -= inv;
            }
            Assert.AreEqual(0, invsum);

            var test0 = monster.Test4(0).Value;
            var test1 = monster.Test4(1).Value;
            Assert.AreEqual(2, monster.Test4Length);

            Assert.AreEqual(100, test0.A + test0.B + test1.A + test1.B);

            Assert.AreEqual(2, monster.TestarrayofstringLength);
            Assert.AreEqual("test1", monster.Testarrayofstring(0));
            Assert.AreEqual("test2", monster.Testarrayofstring(1));

            Assert.AreEqual(true, monster.Testbool);

#if ENABLE_SPAN_T
            var nameBytes = monster.GetNameBytes();
            Assert.AreEqual("MyMonster", Encoding.UTF8.GetString(nameBytes.ToArray(), 0, nameBytes.Length));

            if (0 == monster.TestarrayofboolsLength)
            {
                Assert.IsFalse(monster.GetTestarrayofboolsBytes().Length != 0);
            }
            else
            {
                Assert.IsTrue(monster.GetTestarrayofboolsBytes().Length != 0);
            }

            var longArrayBytes = monster.GetVectorOfLongsBytes();
            Assert.IsTrue(monster.VectorOfLongsLength * 8 == longArrayBytes.Length);

            var doubleArrayBytes = monster.GetVectorOfDoublesBytes();
            Assert.IsTrue(monster.VectorOfDoublesLength * 8 == doubleArrayBytes.Length);
#else
            var nameBytes = monster.GetNameBytes().Value;
            Assert.AreEqual("MyMonster", Encoding.UTF8.GetString(nameBytes.Array, nameBytes.Offset, nameBytes.Count));

            if (0 == monster.TestarrayofboolsLength)
            {
                Assert.IsFalse(monster.GetTestarrayofboolsBytes().HasValue);
            }
            else
            {
                Assert.IsTrue(monster.GetTestarrayofboolsBytes().HasValue);
            }
#endif
    }

        [FlatBuffersTestMethod]
        public void CanReadCppGeneratedWireFile()
        {
            var data = File.ReadAllBytes(@"Resources/monsterdata_test.mon");
            var bb = new ByteBuffer(data);
            TestBuffer(bb);
        }

        [FlatBuffersTestMethod]
        public void TestEnums()
        {
            Assert.AreEqual("Red", Color.Red.ToString());
            Assert.AreEqual("Blue", Color.Blue.ToString());
            Assert.AreEqual("NONE", Any.NONE.ToString());
            Assert.AreEqual("Monster", Any.Monster.ToString());
        }

        [FlatBuffersTestMethod]
        public void TestVectorOfEnums()
        {
            const string monsterName = "TestVectorOfEnumsMonster";
            var colorVec = new Color[] { Color.Red, Color.Green, Color.Blue };
            var fbb = new FlatBufferBuilder(32);
            var str1 = fbb.CreateString(monsterName);
            var vec1 = Monster.CreateVectorOfEnumsVector(fbb, colorVec);
            Monster.StartMonster(fbb);
            Monster.AddName(fbb, str1);
            Monster.AddVectorOfEnums(fbb, vec1);
            var monster1 = Monster.EndMonster(fbb);
            Monster.FinishMonsterBuffer(fbb, monster1);

            var mons = Monster.GetRootAsMonster(fbb.DataBuffer);
            var colors = mons.GetVectorOfEnumsArray();
            Assert.ArrayEqual(colorVec, colors);
        }

        [FlatBuffersTestMethod]
        public void TestNestedFlatBuffer()
        {
            const string nestedMonsterName = "NestedMonsterName";
            const short nestedMonsterHp = 600;
            const short nestedMonsterMana = 1024;
            // Create nested buffer as a Monster type
            var fbb1 = new FlatBufferBuilder(16);
            var str1 = fbb1.CreateString(nestedMonsterName);
            Monster.StartMonster(fbb1);
            Monster.AddName(fbb1, str1);
            Monster.AddHp(fbb1, nestedMonsterHp);
            Monster.AddMana(fbb1, nestedMonsterMana);
            var monster1 = Monster.EndMonster(fbb1);
            Monster.FinishMonsterBuffer(fbb1, monster1);
            var fbb1Bytes = fbb1.SizedByteArray();
            fbb1 = null;

            // Create a Monster which has the first buffer as a nested buffer
            var fbb2 = new FlatBufferBuilder(16);
            var str2 = fbb2.CreateString("My Monster");
            var nestedBuffer = Monster.CreateTestnestedflatbufferVector(fbb2, fbb1Bytes);
            Monster.StartMonster(fbb2);
            Monster.AddName(fbb2, str2);
            Monster.AddHp(fbb2, 50);
            Monster.AddMana(fbb2, 32);
            Monster.AddTestnestedflatbuffer(fbb2, nestedBuffer);
            var monster = Monster.EndMonster(fbb2);
            Monster.FinishMonsterBuffer(fbb2, monster);

            // Now test the data extracted from the nested buffer
            var mons = Monster.GetRootAsMonster(fbb2.DataBuffer);
            var nestedMonster = mons.GetTestnestedflatbufferAsMonster().Value;

            Assert.AreEqual(nestedMonsterMana, nestedMonster.Mana);
            Assert.AreEqual(nestedMonsterHp, nestedMonster.Hp);
            Assert.AreEqual(nestedMonsterName, nestedMonster.Name);
        }

        [FlatBuffersTestMethod]
        public void TestFixedLenghtArrays()
        {
            FlatBufferBuilder builder = new FlatBufferBuilder(100);

            float   a;
            int[]   b = new int[15];
            sbyte   c;
            int[,]  d_a = new int[2, 2];
            TestEnum[]  d_b = new TestEnum[2];
            TestEnum[,] d_c = new TestEnum[2, 2];
            long[,]     d_d = new long[2, 2];
            int         e;
            long[]      f = new long[2];

            a = 0.5f;
            for (int i = 0; i < 15; i++) b[i] = i;
            c = 1;
            d_a[0, 0] = 1;
            d_a[0, 1] = 2;
            d_a[1, 0] = 3;
            d_a[1, 1] = 4;
            d_b[0] = TestEnum.B;
            d_b[1] = TestEnum.C;
            d_c[0, 0] = TestEnum.A;
            d_c[0, 1] = TestEnum.B;
            d_c[1, 0] = TestEnum.C;
            d_c[1, 1] = TestEnum.B;
            d_d[0, 0] = -1;
            d_d[0, 1] = 1;
            d_d[1, 0] = -2;
            d_d[1, 1] = 2;
            e = 2;
            f[0] = -1;
            f[1] = 1;

            Offset<ArrayStruct> arrayOffset = ArrayStruct.CreateArrayStruct(
                builder, a, b, c, d_a, d_b, d_c, d_d, e, f);

            // Create a table with the ArrayStruct.
            ArrayTable.StartArrayTable(builder);
            ArrayTable.AddA(builder, arrayOffset);
            Offset<ArrayTable> tableOffset = ArrayTable.EndArrayTable(builder);

            ArrayTable.FinishArrayTableBuffer(builder, tableOffset);

            ArrayTable table = ArrayTable.GetRootAsArrayTable(builder.DataBuffer);

            Assert.AreEqual(table.A?.A, 0.5f);
            for (int i = 0; i < 15; i++) Assert.AreEqual(table.A?.B(i), i);
            Assert.AreEqual(table.A?.C, (sbyte)1);
            Assert.AreEqual(table.A?.D(0).A(0), 1);
            Assert.AreEqual(table.A?.D(0).A(1), 2);
            Assert.AreEqual(table.A?.D(1).A(0), 3);
            Assert.AreEqual(table.A?.D(1).A(1), 4);
            Assert.AreEqual(table.A?.D(0).B, TestEnum.B);
            Assert.AreEqual(table.A?.D(1).B, TestEnum.C);
            Assert.AreEqual(table.A?.D(0).C(0), TestEnum.A);
            Assert.AreEqual(table.A?.D(0).C(1), TestEnum.B);
            Assert.AreEqual(table.A?.D(1).C(0), TestEnum.C);
            Assert.AreEqual(table.A?.D(1).C(1), TestEnum.B);
            Assert.AreEqual(table.A?.D(0).D(0), -1);
            Assert.AreEqual(table.A?.D(0).D(1), 1);
            Assert.AreEqual(table.A?.D(1).D(0), -2);
            Assert.AreEqual(table.A?.D(1).D(1), 2);
            Assert.AreEqual(table.A?.E, 2);
            Assert.AreEqual(table.A?.F(0), -1);
            Assert.AreEqual(table.A?.F(1), 1);
        }
    }
}
