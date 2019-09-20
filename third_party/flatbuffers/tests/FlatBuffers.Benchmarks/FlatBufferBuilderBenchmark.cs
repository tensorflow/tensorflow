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

using BenchmarkDotNet.Attributes;
using MyGame.Example;

namespace FlatBuffers.Benchmarks
{
    //[EtwProfiler] - needs elevated privileges
    [MemoryDiagnoser]
    public class FlatBufferBuilderBenchmark
    {
        private const int NumberOfRows = 10_000;

        [Benchmark]
        public void BuildNestedMonster()
        {
            const string nestedMonsterName = "NestedMonsterName";
            const short nestedMonsterHp = 600;
            const short nestedMonsterMana = 1024;

            for (int i = 0; i < NumberOfRows; i++)
            {
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
            }
        }

        [Benchmark]
        public void BuildMonster()
        {
            for (int i = 0; i < NumberOfRows; i++)
            {
                var builder = new FlatBufferBuilder(16);
                var str1 = builder.CreateString("MonsterName");
                Monster.StartMonster(builder);
                Monster.AddName(builder, str1);
                Monster.AddHp(builder, 600);
                Monster.AddMana(builder, 1024);
                Monster.AddColor(builder, Color.Blue);
                Monster.AddTestbool(builder, true);
                Monster.AddTestf(builder, 0.3f);
                Monster.AddTestf2(builder, 0.2f);
                Monster.AddTestf3(builder, 0.1f);

                var monster1 = Monster.EndMonster(builder);
                Monster.FinishMonsterBuffer(builder, monster1);
            }
        }

        [Benchmark]
        public void TestTables()
        {
            FlatBufferBuilder builder = new FlatBufferBuilder(1024 * 1024 * 32);
            for (int x = 0; x < 500000; ++x)
            {
                var offset = builder.CreateString("T");
                builder.StartObject(4);
                builder.AddDouble(3.2);
                builder.AddDouble(4.2);
                builder.AddDouble(5.2);
                builder.AddOffset(offset.Value);
                builder.EndObject();
            }
        }
    }
}
