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

import from "../lobster/"
import monster_test_generated

def check_read_buffer(buf):
    // CheckReadBuffer checks that the given buffer is evaluated correctly as the example Monster.
    let monster = MyGame_Example_GetRootAsMonster(buf)

    assert monster.hp == 80
    assert monster.mana == 150
    assert monster.name == "MyMonster"

    let vec = monster.pos
    assert vec
    assert vec.x == 1.0
    assert vec.y == 2.0
    assert vec.z == 3.0
    assert vec.test1 == 3.0
    assert vec.test2 == 2

    let t = vec.test3
    assert t
    assert t.a == 5
    assert t.b == 6

    assert monster.test_type == MyGame_Example_Any_Monster
    assert monster.test_as_Monster.name == "Fred"

    assert monster.inventory_length == 5
    assert sum(map(monster.inventory_length) i: monster.inventory(i)) == 10

    for(5) i:
        assert monster.vector_of_longs(i) == pow(10, i * 2)

    assert equal([-1.7976931348623157e+308, 0, 1.7976931348623157e+308],
                 (map(monster.vector_of_doubles_length) i: monster.vector_of_doubles(i)))

    assert monster.test4_length == 2
    let test0 = monster.test4(0)
    let test1 = monster.test4(1)
    assert test0.a + test0.b + test1.a + test1.b == 100

    assert monster.testarrayofstring_length == 2
    assert monster.testarrayofstring(0) == "test1"
    assert monster.testarrayofstring(1) == "test2"

    assert monster.testarrayoftables_length == 0
    assert monster.testnestedflatbuffer_length == 0
    assert not monster.testempty()

def make_monster_from_generated_code():
    // Use generated code to build the example Monster.
    let b = flatbuffers_builder {}

    let name = b.CreateString("MyMonster")
    let fred = b.CreateString("Fred")

    let inv = b.MyGame_Example_MonsterCreateInventoryVector([ 0, 1, 2, 3, 4 ])

    let mon2 = MyGame_Example_MonsterBuilder { b }
        .start()
        .add_name(fred)
        .end()

    b.MyGame_Example_MonsterStartTest4Vector(2)
    b.MyGame_Example_CreateTest(10, 20)
    b.MyGame_Example_CreateTest(30, 40)
    let test4 = b.EndVector(2)

    let test_array_of_string = b.MyGame_Example_MonsterCreateTestarrayofstringVector(
                                   [ b.CreateString("test1"), b.CreateString("test2") ])

    let vector_of_longs = b.MyGame_Example_MonsterCreateVectorOfLongsVector(
                              [ 1, 100, 10000, 1000000, 100000000 ])

    let vector_of_doubles = b.MyGame_Example_MonsterCreateVectorOfDoublesVector(
                                [ -1.7976931348623157e+308, 0, 1.7976931348623157e+308 ])

    let mon = MyGame_Example_MonsterBuilder { b }
        .start()
        .add_pos(b.MyGame_Example_CreateVec3(1.0, 2.0, 3.0, 3.0,
                 MyGame_Example_Color_Green, 5, 6))
        .add_hp(80)
        .add_name(name)
        .add_inventory(inv)
        .add_test_type(MyGame_Example_Any_Monster)
        .add_test(mon2)
        .add_test4(test4)
        .add_testarrayofstring(test_array_of_string)
        .add_vector_of_longs(vector_of_longs)
        .add_vector_of_doubles(vector_of_doubles)
        .end()

    b.Finish(mon)

    return b.SizedCopy()

// Verify that the canonical flatbuffer file (produced by the C++ implementation)
// is readable by the generated Lobster code.
let fb2 = read_file("monsterdata_test.mon")
assert fb2
check_read_buffer(fb2)

// Verify that using the generated Lobster code builds a buffer without
// returning errors, and is interpreted correctly.
let fb1 = make_monster_from_generated_code()
check_read_buffer(fb1)
// Write the result to file for no good reason.
write_file("monsterdata_lobster_wire.mon", fb1)

// Test converting the buffer to JSON and parsing the JSON back again.
let schema = read_file("monster_test.fbs")
assert schema
let includedirs = [ "include_test" ]
// Convert binary to JSON:
let json, err1 = flatbuffers_binary_to_json(schema, fb1, includedirs)
assert not err1
// Parse JSON back to binary:
let fb3, err2 = flatbuffers_json_to_binary(schema, json, includedirs)
assert not err2
// Check the resulting binary again (full roundtrip test):
check_read_buffer(fb3)

print "Lobster test succesful!"