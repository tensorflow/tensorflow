Use in Lua    {#flatbuffers_guide_use_lua}
=============

## Before you get started

Before diving into the FlatBuffers usage in Lua, it should be noted that the
[Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide to general
FlatBuffers usage in all of the supported languages (including Lua). This
page is designed to cover the nuances of FlatBuffers usage, specific to
Lua.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## FlatBuffers Lua library code location

The code for the FlatBuffers Lua library can be found at
`flatbuffers/lua`. You can browse the library code on the
[FlatBuffers GitHub page](https://github.com/google/flatbuffers/tree/master/lua).

## Testing the FlatBuffers Lua library

The code to test the Lua library can be found at `flatbuffers/tests`.
The test code itself is located in [luatest.lua](https://github.com/google/
flatbuffers/blob/master/tests/luatest.lua).

To run the tests, use the [LuaTest.sh](https://github.com/google/flatbuffers/
blob/master/tests/LuaTest.sh) shell script.

*Note: This script requires [Lua 5.3](https://www.lua.org/) to be
installed.*

## Using the FlatBuffers Lua library

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in Lua.*

There is support for both reading and writing FlatBuffers in Lua.

To use FlatBuffers in your own code, first generate Lua classes from your
schema with the `--lua` option to `flatc`. Then you can include both
FlatBuffers and the generated code to read or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in Lua:
First, require the module and the generated code. Then read a FlatBuffer binary
file into a `string`, which you pass to the `GetRootAsMonster` function:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.lua}
    -- require the library
    local flatbuffers = require("flatbuffers")
    
    -- require the generated code
    local monster = require("MyGame.Sample.Monster")

    -- read the flatbuffer from a file into a string
    local f = io.open('monster.dat', 'rb')
    local buf = f:read('*a')
    f:close()

    -- parse the flatbuffer to get an instance to the root monster
    local monster1 = monster.GetRootAsMonster(buf, 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can access values like this:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.lua}
    -- use the : notation to access member data
    local hp = monster1:Hp()
    local pos = monster1:Pos()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


## Text Parsing

There currently is no support for parsing text (Schema's and JSON) directly
from Lua, though you could use the C++ parser through SWIG or ctypes. Please
see the C++ documentation for more on text parsing.

<br>
