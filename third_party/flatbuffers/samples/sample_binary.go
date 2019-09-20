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

// To run, use the `go_sample.sh` script.

package main

import (
	sample "MyGame/Sample"
	"fmt"
	flatbuffers "github.com/google/flatbuffers/go"
	"strconv"
)

// Example how to use Flatbuffers to create and read binary buffers.
func main() {
	builder := flatbuffers.NewBuilder(0)

	// Create some weapons for our Monster ("Sword" and "Axe").
	weaponOne := builder.CreateString("Sword")
	weaponTwo := builder.CreateString("Axe")

	sample.WeaponStart(builder)
	sample.WeaponAddName(builder, weaponOne)
	sample.WeaponAddDamage(builder, 3)
	sword := sample.WeaponEnd(builder)

	sample.WeaponStart(builder)
	sample.WeaponAddName(builder, weaponTwo)
	sample.WeaponAddDamage(builder, 5)
	axe := sample.WeaponEnd(builder)

	// Serialize the FlatBuffer data.
	name := builder.CreateString("Orc")

	sample.MonsterStartInventoryVector(builder, 10)
	// Note: Since we prepend the bytes, this loop iterates in reverse.
	for i := 9; i >= 0; i-- {
		builder.PrependByte(byte(i))
	}
	inv := builder.EndVector(10)

	sample.MonsterStartWeaponsVector(builder, 2)
	// Note: Since we prepend the weapons, prepend in reverse order.
	builder.PrependUOffsetT(axe)
	builder.PrependUOffsetT(sword)
	weapons := builder.EndVector(2)

	pos := sample.CreateVec3(builder, 1.0, 2.0, 3.0)

	sample.MonsterStart(builder)
	sample.MonsterAddPos(builder, pos)
	sample.MonsterAddHp(builder, 300)
	sample.MonsterAddName(builder, name)
	sample.MonsterAddInventory(builder, inv)
	sample.MonsterAddColor(builder, sample.ColorRed)
	sample.MonsterAddWeapons(builder, weapons)
	sample.MonsterAddEquippedType(builder, sample.EquipmentWeapon)
	sample.MonsterAddEquipped(builder, axe)
	orc := sample.MonsterEnd(builder)

	builder.Finish(orc)

	// We now have a FlatBuffer that we could store on disk or send over a network.

	// ...Saving to file or sending over a network code goes here...

	// Instead, we are going to access this buffer right away (as if we just received it).

	buf := builder.FinishedBytes()

	// Note: We use `0` for the offset here, since we got the data using the
	// `builder.FinishedBytes()` method. This simulates the data you would store/receive in your
	// FlatBuffer. If you wanted to read from the `builder.Bytes` directly, you would need to
	// pass in the offset of `builder.Head()`, as the builder actually constructs the buffer
	// backwards.
	monster := sample.GetRootAsMonster(buf, 0)

	// Note: We did not set the `mana` field explicitly, so we get the
	// default value.
	assert(monster.Mana() == 150, "`monster.Mana()`", strconv.Itoa(int(monster.Mana())), "150")
	assert(monster.Hp() == 300, "`monster.Hp()`", strconv.Itoa(int(monster.Hp())), "300")
	assert(string(monster.Name()) == "Orc", "`string(monster.Name())`", string(monster.Name()),
		"\"Orc\"")
	assert(monster.Color() == sample.ColorRed, "`monster.Color()`",
		strconv.Itoa(int(monster.Color())), strconv.Itoa(int(sample.ColorRed)))

	// Note: Whenever you access a new object, like in `Pos()`, a new temporary accessor object
	// gets created. If your code is very performance sensitive, you can pass in a pointer to an
	// existing `Vec3` instead of `nil`. This allows you to reuse it across many calls to reduce
	// the amount of object allocation/garbage collection.
	assert(monster.Pos(nil).X() == 1.0, "`monster.Pos(nil).X()`",
		strconv.FormatFloat(float64(monster.Pos(nil).X()), 'f', 1, 32), "1.0")
	assert(monster.Pos(nil).Y() == 2.0, "`monster.Pos(nil).Y()`",
		strconv.FormatFloat(float64(monster.Pos(nil).Y()), 'f', 1, 32), "2.0")
	assert(monster.Pos(nil).Z() == 3.0, "`monster.Pos(nil).Z()`",
		strconv.FormatFloat(float64(monster.Pos(nil).Z()), 'f', 1, 32), "3.0")

	// For vectors, like `Inventory`, they have a method suffixed with 'Length' that can be used
	// to query the length of the vector. You can index the vector by passing an index value
	// into the accessor.
	for i := 0; i < monster.InventoryLength(); i++ {
		assert(monster.Inventory(i) == byte(i), "`monster.Inventory(i)`",
			strconv.Itoa(int(monster.Inventory(i))), strconv.Itoa(int(byte(i))))
	}

	expectedWeaponNames := []string{"Sword", "Axe"}
	expectedWeaponDamages := []int{3, 5}
	weapon := new(sample.Weapon) // We need a `sample.Weapon` to pass into `monster.Weapons()`
	// to capture the output of that function.
	for i := 0; i < monster.WeaponsLength(); i++ {
		if monster.Weapons(weapon, i) {
			assert(string(weapon.Name()) == expectedWeaponNames[i], "`weapon.Name()`",
				string(weapon.Name()), expectedWeaponNames[i])
			assert(int(weapon.Damage()) == expectedWeaponDamages[i],
				"`weapon.Damage()`", strconv.Itoa(int(weapon.Damage())),
				strconv.Itoa(expectedWeaponDamages[i]))
		}
	}

	// For FlatBuffer `union`s, you can get the type of the union, as well as the union
	// data itself.
	assert(monster.EquippedType() == sample.EquipmentWeapon, "`monster.EquippedType()`",
		strconv.Itoa(int(monster.EquippedType())), strconv.Itoa(int(sample.EquipmentWeapon)))

	unionTable := new(flatbuffers.Table)
	if monster.Equipped(unionTable) {
		// An example of how you can appropriately convert the table depending on the
		// FlatBuffer `union` type. You could add `else if` and `else` clauses to handle
		// other FlatBuffer `union` types for this field. (Similarly, this could be
		// done in a switch statement.)
		if monster.EquippedType() == sample.EquipmentWeapon {
			unionWeapon := new(sample.Weapon)
			unionWeapon.Init(unionTable.Bytes, unionTable.Pos)

			assert(string(unionWeapon.Name()) == "Axe", "`unionWeapon.Name()`",
				string(unionWeapon.Name()), "Axe")
			assert(int(unionWeapon.Damage()) == 5, "`unionWeapon.Damage()`",
				strconv.Itoa(int(unionWeapon.Damage())), strconv.Itoa(5))
		}
	}

	fmt.Printf("The FlatBuffer was successfully created and verified!\n")
}

// A helper function to print out if an assertion failed.
func assert(assertPassed bool, codeExecuted string, actualValue string, expectedValue string) {
	if assertPassed == false {
		panic("Assert failed! " + codeExecuted + " (" + actualValue +
			") was not equal to " + expectedValue + ".")
	}
}
