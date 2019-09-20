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

// Run this file with the `java_sample.sh` script.

import MyGame.Sample.Color
import MyGame.Sample.Equipment
import MyGame.Sample.Monster
import MyGame.Sample.Vec3
import MyGame.Sample.Weapon

import com.google.flatbuffers.FlatBufferBuilder

class SampleBinary {

  companion object {
    // Example how to use FlatBuffers to create and read binary buffers.
    @JvmStatic
    fun main(args: Array<String>) {
        val builder = FlatBufferBuilder(0)

        // Create some weapons for our Monster ('Sword' and 'Axe').
        val weaponOneName = builder.createString("Sword")
        val weaponOneDamage: Short = 3
        val weaponTwoName = builder.createString("Axe")
        val weaponTwoDamage: Short = 5

        // Use the `createWeapon()` helper function to create the weapons, since we set every field.
        val weaps = IntArray(2)
        weaps[0] = Weapon.createWeapon(builder, weaponOneName, weaponOneDamage)
        weaps[1] = Weapon.createWeapon(builder, weaponTwoName, weaponTwoDamage)

        // Serialize the FlatBuffer data.
        val name = builder.createString("Orc")
        val treasure = byteArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        val inv = Monster.createInventoryVector(builder, treasure)
        val weapons = Monster.createWeaponsVector(builder, weaps)
        val pos = Vec3.createVec3(builder, 1.0f, 2.0f, 3.0f)

        Monster.startMonster(builder)
        Monster.addPos(builder, pos)
        Monster.addName(builder, name)
        Monster.addColor(builder, Color.Red)
        Monster.addHp(builder, 300.toShort())
        Monster.addInventory(builder, inv)
        Monster.addWeapons(builder, weapons)
        Monster.addEquippedType(builder, Equipment.Weapon)
        Monster.addEquipped(builder, weaps[1])
        val orc = Monster.endMonster(builder)

        builder.finish(orc) // You could also call `Monster.finishMonsterBuffer(builder, orc);`.

        // We now have a FlatBuffer that can be stored on disk or sent over a network.

        // ...Code to store to disk or send over a network goes here...

        // Instead, we are going to access it right away, as if we just received it.

        val buf = builder.dataBuffer()

        // Get access to the root:
        val monster = Monster.getRootAsMonster(buf)

        // Note: We did not set the `mana` field explicitly, so we get back the default value.
        assert(monster.mana == 150.toShort())
        assert(monster.hp == 300.toShort())
        assert(monster.name.equals("Orc"))
        assert(monster.color == Color.Red)
        assert(monster.pos!!.x == 1.0f)
        assert(monster.pos!!.y == 2.0f)
        assert(monster.pos!!.z == 3.0f)

        // Get and test the `inventory` FlatBuffer `vector`.
        for (i in 0 until monster.inventoryLength) {
            assert(monster.inventory(i) == i.toByte().toInt())
        }

        // Get and test the `weapons` FlatBuffer `vector` of `table`s.
        val expectedWeaponNames = arrayOf("Sword", "Axe")
        val expectedWeaponDamages = intArrayOf(3, 5)
        for (i in 0 until monster.weaponsLength) {
            assert(monster.weapons(i)!!.name.equals(expectedWeaponNames[i]))
            assert(monster.weapons(i)!!.damage.toInt() == expectedWeaponDamages[i])
        }

        // Get and test the `equipped` FlatBuffer `union`.
        assert(monster.equippedType == Equipment.Weapon)
        val equipped = monster.equipped(Weapon()) as Weapon?
        assert(equipped!!.name.equals("Axe"))
        assert(equipped.damage == 5.toShort())

        println("The FlatBuffer was successfully created and verified!")
    }
  }
}
