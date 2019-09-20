-- need to update the Lua path to point to the local flatbuffers implementation
package.path = string.format("../lua/?.lua;%s",package.path)
package.path = string.format("./lua/?.lua;%s",package.path)

-- require the library
local flatbuffers = require("flatbuffers")

local binaryArray = flatbuffers.binaryArray-- for hex dump utility

-- require the files generated from the schema
local weapon = require("MyGame.Sample.Weapon")
local monster = require("MyGame.Sample.Monster")
local vec3 = require("MyGame.Sample.Vec3")
local color = require("MyGame.Sample.Color")
local equipment = require("MyGame.Sample.Equipment")

-- get access to the builder, providing an array of size 1024
local builder = flatbuffers.Builder(1024)

local weaponOne = builder:CreateString("Sword")
local weaponTwo = builder:CreateString("Axe")

-- Create the first 'Weapon'
weapon.Start(builder)
weapon.AddName(builder, weaponOne)
weapon.AddDamage(builder, 3)
local sword = weapon.End(builder)

-- Create the second 'Weapon'
weapon.Start(builder)
weapon.AddName(builder, weaponTwo)
weapon.AddDamage(builder, 5)
local axe = weapon.End(builder)

-- Serialize a name for our mosnter, called 'orc'
local name = builder:CreateString("Orc")

-- Create a `vector` representing the inventory of the Orc. Each number
-- could correspond to an item that can be claimed after he is slain.
-- Note: Since we prepend the bytes, this loop iterates in reverse.
monster.StartInventoryVector(builder, 10)
for i=10,1,-1 do
    builder:PrependByte(i)
end
local inv = builder:EndVector(10)

-- Create a FlatBuffer vector and prepend the weapons.
-- Note: Since we prepend the data, prepend them in reverse order.
monster.StartWeaponsVector(builder, 2)
builder:PrependUOffsetTRelative(axe)
builder:PrependUOffsetTRelative(sword)
local weapons = builder:EndVector(2)

-- Create our monster by using Start() andEnd()
monster.Start(builder)
monster.AddPos(builder, vec3.CreateVec3(builder, 1.0, 2.0, 3.0))
monster.AddHp(builder, 300)
monster.AddName(builder, name)
monster.AddInventory(builder, inv)
monster.AddColor(builder, color.Red)
monster.AddWeapons(builder, weapons)
monster.AddEquippedType(builder, equipment.Weapon)
monster.AddEquipped(builder, axe)
local orc = monster.End(builder)

-- Call 'Finish()' to instruct the builder that this monster is complete.
builder:Finish(orc)

-- Get the flatbuffer as a string containing the binary data
local bufAsString = builder:Output()

-- Convert the string representation into binary array Lua structure
local buf = flatbuffers.binaryArray.New(bufAsString)

-- Get an accessor to the root object insert the buffer
local mon = monster.GetRootAsMonster(buf, 0)

assert(mon:Mana() == 150)
assert(mon:Hp() == 300)
assert(mon:Name() == "Orc")
assert(mon:Color() == color.Red)
assert(mon:Pos():X() == 1.0)
assert(mon:Pos():Y() == 2.0)
assert(mon:Pos():Z() == 3.0)

for i=1,mon:InventoryLength() do
    assert(mon:Inventory(i) == i)
end

local expected = { 
    {w = 'Sword', d = 3}, 
    {w = 'Axe', d = 5}
}

for i=1,mon:WeaponsLength() do
   assert(mon:Weapons(i):Name() == expected[i].w)
   assert(mon:Weapons(i):Damage() == expected[i].d)
end

assert(mon:EquippedType() == equipment.Weapon)

local unionWeapon = weapon.New()
unionWeapon:Init(mon:Equipped().bytes,mon:Equipped().pos)
assert(unionWeapon:Name() == "Axe")
assert(unionWeapon:Damage() == 5)

print("The Lua FlatBuffer example was successfully created and verified!")