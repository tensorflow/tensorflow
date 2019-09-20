/*
 * Copyright 2018 Google Inc. All rights reserved.
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

// import the flatbuffers runtime library
extern crate flatbuffers;

// import the generated code
#[path = "./monster_generated.rs"]
mod monster_generated;
pub use monster_generated::my_game::sample::{get_root_as_monster,
                                             Color, Equipment,
                                             Monster, MonsterArgs,
                                             Vec3,
                                             Weapon, WeaponArgs};


// Example how to use FlatBuffers to create and read binary buffers.

fn main() {
  // Build up a serialized buffer algorithmically.
  // Initialize it with a capacity of 1024 bytes.
  let mut builder = flatbuffers::FlatBufferBuilder::new_with_capacity(1024);

  // Serialize some weapons for the Monster: A 'sword' and an 'axe'.
  let weapon_one_name = builder.create_string("Sword");
  let weapon_two_name = builder.create_string("Axe");

  // Use the `Weapon::create` shortcut to create Weapons with named field
  // arguments.
  let sword = Weapon::create(&mut builder, &WeaponArgs{
      name: Some(weapon_one_name),
      damage: 3,
  });
  let axe = Weapon::create(&mut builder, &WeaponArgs{
      name: Some(weapon_two_name),
      damage: 5,
  });

  // Name of the Monster.
  let name = builder.create_string("Orc");

  // Inventory.
  let inventory = builder.create_vector(&[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

  // Create a FlatBuffer `vector` that contains offsets to the sword and axe
  // we created above.
  let weapons = builder.create_vector(&[sword, axe]);

  // Create the path vector of Vec3 objects:
  //let x = Vec3::new(1.0, 2.0, 3.0);
  //let y = Vec3::new(4.0, 5.0, 6.0);
  //let path = builder.create_vector(&[x, y]);

  // Note that, for convenience, it is also valid to create a vector of
  // references to structs, like this:
  // let path = builder.create_vector(&[&x, &y]);

  // Create the monster using the `Monster::create` helper function. This
  // function accepts a `MonsterArgs` struct, which supplies all of the data
  // needed to build a `Monster`. To supply empty/default fields, just use the
  // Rust built-in `Default::default()` function, as demononstrated below.
  let orc = Monster::create(&mut builder, &MonsterArgs{
      pos: Some(&Vec3::new(1.0f32, 2.0f32, 3.0f32)),
      mana: 150,
      hp: 80,
      name: Some(name),
      inventory: Some(inventory),
      color: Color::Red,
      weapons: Some(weapons),
      equipped_type: Equipment::Weapon,
      equipped: Some(axe.as_union_value()),
      //path: Some(path),
      ..Default::default()
  });

  // Serialize the root of the object, without providing a file identifier.
  builder.finish(orc, None);

  // We now have a FlatBuffer we can store on disk or send over a network.

  // ** file/network code goes here :) **

  // Instead, we're going to access it right away (as if we just received it).
  // This must be called after `finish()`.
  let buf = builder.finished_data(); // Of type `&[u8]`

  // Get access to the root:
  let monster = get_root_as_monster(buf);

  // Get and test some scalar types from the FlatBuffer.
  let hp = monster.hp();
  let mana = monster.mana();
  let name = monster.name();

  assert_eq!(hp, 80);
  assert_eq!(mana, 150);  // default
  assert_eq!(name, Some("Orc"));

  // Get and test a field of the FlatBuffer's `struct`.
  assert!(monster.pos().is_some());
  let pos = monster.pos().unwrap();
  let x = pos.x();
  let y = pos.y();
  let z = pos.z();
  assert_eq!(x, 1.0f32);
  assert_eq!(y, 2.0f32);
  assert_eq!(z, 3.0f32);

  // Get an element from the `inventory` FlatBuffer's `vector`.
  assert!(monster.inventory().is_some());
  let inv = monster.inventory().unwrap();

  // Note that this vector is returned as a slice, because direct access for
  // this type, a u8 vector, is safe on all platforms:
  let third_item = inv[2];
  assert_eq!(third_item, 2);

  // Get and test the `weapons` FlatBuffers's `vector`.
  assert!(monster.weapons().is_some());
  let weps = monster.weapons().unwrap();
  //let weps_len = weps.len();
  let wep2 = weps.get(1);
  let second_weapon_name = wep2.name();
  let second_weapon_damage = wep2.damage();
  assert_eq!(second_weapon_name, Some("Axe"));
  assert_eq!(second_weapon_damage, 5);

  // Get and test the `Equipment` union (`equipped` field).
  assert_eq!(monster.equipped_type(), Equipment::Weapon);
  let equipped = monster.equipped_as_weapon().unwrap();
  let weapon_name = equipped.name();
  let weapon_damage = equipped.damage();
  assert_eq!(weapon_name, Some("Axe"));
  assert_eq!(weapon_damage, 5);

  // Get and test the `path` FlatBuffers's `vector`.
  //assert_eq!(monster.path().unwrap().len(), 2);
  //assert_eq!(monster.path().unwrap()[0].x(), 1.0);
  //assert_eq!(monster.path().unwrap()[1].x(), 4.0);

  println!("The FlatBuffer was successfully created and accessed!");
}
