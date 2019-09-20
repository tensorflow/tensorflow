/*
 * Copyright 2015 Google Inc. All rights reserved.
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

// To run, use the `javascript_sample.sh` script.

var assert = require('assert');
var flatbuffers = require('../js/flatbuffers').flatbuffers;
var MyGame = require('./monster_generated').MyGame;

// Example how to use FlatBuffers to create and read binary buffers.
function main() {
  var builder = new flatbuffers.Builder(0);

  // Create some weapons for our Monster ('Sword' and 'Axe').
  var weaponOne = builder.createString('Sword');
  var weaponTwo = builder.createString('Axe');

  MyGame.Sample.Weapon.startWeapon(builder);
  MyGame.Sample.Weapon.addName(builder, weaponOne);
  MyGame.Sample.Weapon.addDamage(builder, 3);
  var sword = MyGame.Sample.Weapon.endWeapon(builder);

  MyGame.Sample.Weapon.startWeapon(builder);
  MyGame.Sample.Weapon.addName(builder, weaponTwo);
  MyGame.Sample.Weapon.addDamage(builder, 5);
  var axe = MyGame.Sample.Weapon.endWeapon(builder);

  // Serialize the FlatBuffer data.
  var name = builder.createString('Orc');

  var treasure = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  var inv = MyGame.Sample.Monster.createInventoryVector(builder, treasure);

  var weaps = [sword, axe];
  var weapons = MyGame.Sample.Monster.createWeaponsVector(builder, weaps);

  var pos = MyGame.Sample.Vec3.createVec3(builder, 1.0, 2.0, 3.0);

  MyGame.Sample.Monster.startMonster(builder);
  MyGame.Sample.Monster.addPos(builder, pos);
  MyGame.Sample.Monster.addHp(builder, 300);
  MyGame.Sample.Monster.addColor(builder, MyGame.Sample.Color.Red)
  MyGame.Sample.Monster.addName(builder, name);
  MyGame.Sample.Monster.addInventory(builder, inv);
  MyGame.Sample.Monster.addWeapons(builder, weapons);
  MyGame.Sample.Monster.addEquippedType(builder, MyGame.Sample.Equipment.Weapon);
  MyGame.Sample.Monster.addEquipped(builder, weaps[1]);
  var orc = MyGame.Sample.Monster.endMonster(builder);

  builder.finish(orc); // You may also call 'MyGame.Example.Monster.finishMonsterBuffer(builder, orc);'.

  // We now have a FlatBuffer that can be stored on disk or sent over a network.

  // ...Code to store to disk or send over a network goes here...

  // Instead, we are going to access it right away, as if we just received it.

  var buf = builder.dataBuffer();

  // Get access to the root:
  var monster = MyGame.Sample.Monster.getRootAsMonster(buf);

  // Note: We did not set the `mana` field explicitly, so we get back the default value.
  assert.equal(monster.mana(), 150);
  assert.equal(monster.hp(), 300);
  assert.equal(monster.name(), 'Orc');
  assert.equal(monster.color(), MyGame.Sample.Color.Red);
  assert.equal(monster.pos().x(), 1.0);
  assert.equal(monster.pos().y(), 2.0);
  assert.equal(monster.pos().z(), 3.0);

  // Get and test the `inventory` FlatBuffer `vector`.
  for (var i = 0; i < monster.inventoryLength(); i++) {
    assert.equal(monster.inventory(i), i);
  }

  // Get and test the `weapons` FlatBuffer `vector` of `table`s.
  var expectedWeaponNames = ['Sword', 'Axe'];
  var expectedWeaponDamages = [3, 5];
  for (var i = 0; i < monster.weaponsLength(); i++) {
    assert.equal(monster.weapons(i).name(), expectedWeaponNames[i]);
    assert.equal(monster.weapons(i).damage(), expectedWeaponDamages[i]);
  }

  // Get and test the `equipped` FlatBuffer `union`.
  assert.equal(monster.equippedType(), MyGame.Sample.Equipment.Weapon);
  assert.equal(monster.equipped(new MyGame.Sample.Weapon()).name(), 'Axe');
  assert.equal(monster.equipped(new MyGame.Sample.Weapon()).damage(), 5);

  console.log('The FlatBuffer was successfully created and verified!');
}

main();
