<?php
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

// To run, use the `php_sample.sh` script.

// It is recommended that you use PSR autoload when using FlatBuffers.
function __autoload($class_name) {
  $class = substr($class_name, strrpos($class_name, "\\") + 1);
  $root_dir = join(DIRECTORY_SEPARATOR, array(dirname(dirname(__FILE__)))); // `flatbuffers` root.
  $paths = array(join(DIRECTORY_SEPARATOR, array($root_dir, "php")),
                 join(DIRECTORY_SEPARATOR, array($root_dir, "samples", "MyGame", "Sample")));
  foreach ($paths as $path) {
    $file = join(DIRECTORY_SEPARATOR, array($path, $class . ".php"));
    if (file_exists($file)) {
      require($file);
      break;
    }
  }
}

// Example how to use FlatBuffers to create and read binary buffers.
function main() {
  $builder = new Google\FlatBuffers\FlatbufferBuilder(0);

  // Create some weapons for our Monster using the `createWeapon()` helper function.
  $weapon_one = $builder->createString("Sword");
  $sword = \MyGame\Sample\Weapon::CreateWeapon($builder, $weapon_one, 3);
  $weapon_two = $builder->createString("Axe");
  $axe = \MyGame\Sample\Weapon::CreateWeapon($builder, $weapon_two, 5);

  // Serialize the FlatBuffer data.
  $name = $builder->createString("Orc");

  $treasure = array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  $inv = \MyGame\Sample\Monster::CreateInventoryVector($builder, $treasure);

  $weaps = array($sword, $axe);
  $weapons = \MyGame\Sample\Monster::CreateWeaponsVector($builder, $weaps);

  $pos = \MyGame\Sample\Vec3::CreateVec3($builder, 1.0, 2.0, 3.0);

  \MyGame\Sample\Monster::StartMonster($builder);
  \MyGame\Sample\Monster::AddPos($builder, $pos);
  \MyGame\Sample\Monster::AddHp($builder, 300);
  \MyGame\Sample\Monster::AddName($builder, $name);
  \MyGame\Sample\Monster::AddInventory($builder, $inv);
  \MyGame\Sample\Monster::AddColor($builder, \MyGame\Sample\Color::Red);
  \MyGame\Sample\Monster::AddWeapons($builder, $weapons);
  \MyGame\Sample\Monster::AddEquippedType($builder, \MyGame\Sample\Equipment::Weapon);
  \MyGame\Sample\Monster::AddEquipped($builder, $weaps[1]);
  $orc = \MyGame\Sample\Monster::EndMonster($builder);

  $builder->finish($orc); // You may also call `\MyGame\Sample\Monster::FinishMonsterBuffer($builder, $orc);`.

  // We now have a FlatBuffer that can be stored on disk or sent over a network.

  // ...Code to store to disk or send over a network goes here...

  // Instead, we are going to access it right away, as if we just received it.

  $buf = $builder->dataBuffer();

  // Get access to the root:
  $monster = \MyGame\Sample\Monster::GetRootAsMonster($buf);

  $success = true; // Tracks if an assert occurred.

  // Note: We did not set the `mana` field explicitly, so we get back the default value.
  $success &= assert($monster->getMana() == 150);
  $success &= assert($monster->getHp() == 300);
  $success &= assert($monster->getName() == "Orc");
  $success &= assert($monster->getColor() == \MyGame\Sample\Color::Red);
  $success &= assert($monster->getPos()->getX() == 1.0);
  $success &= assert($monster->getPos()->getY() == 2.0);
  $success &= assert($monster->getPos()->getZ() == 3.0);

  // Get and test the `inventory` FlatBuffer `vector`.
  for ($i = 0; $i < $monster->getInventoryLength(); $i++) {
    $success &= assert($monster->getInventory($i) == $i);
  }

  // Get and test the `weapons` FlatBuffer `vector` of `table`s.
  $expected_weapon_names = array("Sword", "Axe");
  $expected_weapon_damages = array(3, 5);
  for ($i = 0; $i < $monster->getWeaponsLength(); $i++) {
    $success &= assert($monster->getWeapons($i)->getName() == $expected_weapon_names[$i]);
    $success &= assert($monster->getWeapons($i)->getDamage() == $expected_weapon_damages[$i]);
  }

  // Get and test the `equipped` FlatBuffer `union`.
  $success &= assert($monster->getEquippedType() == \MyGame\Sample\Equipment::Weapon);
  $success &= assert($monster->getEquipped(new \MyGame\Sample\Weapon())->getName() == "Axe");
  $success &= assert($monster->getEquipped(new \MyGame\Sample\Weapon())->getDamage() == 5);

  if ($success) {
    print("The FlatBuffer was successfully created and verified!\n");
  }
}

main();
?>
