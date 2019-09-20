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
import monster_generated

// Example of how to use FlatBuffers to create and read binary buffers.

// Create a builder.
let b = flatbuffers_builder {}

// Create some weapons for our monster.
let weapon_names = [ "Sword", "Axe" ]
let weapon_damages = [ 3, 5 ]

let weapon_offsets = map(weapon_names) name, i:
    let ns = b.CreateString(name)
    MyGame_Sample_WeaponBuilder { b }
        .start()
        .add_name(ns)
        .add_damage(weapon_damages[i])
        .end()

let weapons = b.MyGame_Sample_MonsterCreateWeaponsVector(weapon_offsets)

// Name of the monster.
let name = b.CreateString("Orc")

// Inventory.
let inv = b.MyGame_Sample_MonsterCreateInventoryVector(map(10): _)

// Now pack it all together in our root monster object.
let orc = MyGame_Sample_MonsterBuilder { b }
    .start()
    .add_pos(b.MyGame_Sample_CreateVec3(1.0, 2.0, 3.0))
    .add_hp(300)
    .add_name(name)
    .add_inventory(inv)
    .add_color(MyGame_Sample_Color_Red)
    .add_weapons(weapons)
    .add_equipped_type(MyGame_Sample_Equipment_Weapon)
    .add_equipped(weapon_offsets[1])
    .end()

// Finish the buffer!
b.Finish(orc)

// We now have a FlatBuffer that we could store on disk or send over a network.

let buf = b.SizedCopy()

// ...Saving to file or sending over a network code goes here...

// Instead, we are going to access this buffer right away (as if we just
// received it).

// Get the root object accessor.
let monster = MyGame_Sample_GetRootAsMonster(buf)

// Note: We did not set the `mana` field explicitly, so we get a default value.
assert monster.mana == 150
assert monster.hp == 300
assert monster.name == "Orc"
assert monster.color == MyGame_Sample_Color_Red
let pos = monster.pos
assert pos
assert pos.x == 1.0
assert pos.y == 2.0
assert pos.z == 3.0

// Get and test the `inventory` FlatBuffer vector.
for(monster.inventory_length) e, i:
  assert monster.inventory(i) == e

// Get and test the `weapons` FlatBuffer vector of tables.
for(monster.weapons_length) i:
  assert monster.weapons(i).name == weapon_names[i]
  assert monster.weapons(i).damage == weapon_damages[i]

// Get and test the `equipped` FlatBuffer union.
assert monster.equipped_type() == MyGame_Sample_Equipment_Weapon

// Now that we know the union value is a weapon, we can safely call as_Weapon:
let union_weapon = monster.equipped_as_Weapon

assert union_weapon.name == "Axe"
assert union_weapon.damage == 5

print "The FlatBuffer was successfully created and verified!"
