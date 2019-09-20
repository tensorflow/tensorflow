Use in Lobster    {#flatbuffers_guide_use_lobster}
==============

## Before you get started

Before diving into the FlatBuffers usage in Lobster, it should be noted that the
[Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide to general
FlatBuffers usage in all of the supported languages (including Lobster). This
page is designed to cover the nuances of FlatBuffers usage, specific to
Lobster.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## FlatBuffers Lobster library code location

The code for the FlatBuffers Lobster library can be found at
`flatbuffers/lobster`. You can browse the library code on the
[FlatBuffers GitHub page](https://github.com/google/flatbuffers/tree/master/
lobster).

## Testing the FlatBuffers Lobster library

The code to test the Lobster library can be found at `flatbuffers/tests`.
The test code itself is located in [lobstertest.lobster](https://github.com/google/
flatbuffers/blob/master/tests/lobstertest.lobster).

To run the tests, run `lobster lobstertest.lobster`. To obtain Lobster itself,
go to the [Lobster homepage](http://strlen.com/lobster) or
[github](https://github.com/aardappel/lobster) to learn how to build it for your
platform.

## Using the FlatBuffers Lobster library

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in Lobster.*

There is support for both reading and writing FlatBuffers in Lobster.

To use FlatBuffers in your own code, first generate Lobster classes from your
schema with the `--lobster` option to `flatc`. Then you can include both
FlatBuffers and the generated code to read or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in Lobster:
First, import the library and the generated code. Then read a FlatBuffer binary
file into a string, which you pass to the `GetRootAsMonster` function:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.lobster}
    include "monster_generated.lobster"

    let fb = read_file("monsterdata_test.mon")
    assert fb
    let monster = MyGame_Example_GetRootAsMonster(fb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can access values like this:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.lobster}
    let hp = monster.hp
    let pos = monster.pos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you can see, even though `hp` and `pos` are functions that access FlatBuffer
data in-place in the string buffer, they appear as field accesses.

## Speed

Using FlatBuffers in Lobster should be relatively fast, as the implementation
makes use of native support for writing binary values, and access of vtables.
Both generated code and the runtime library are therefore small and fast.

Actual speed will depend on wether you use Lobster as bytecode VM or compiled to
C++.

## Text Parsing

Lobster has full support for parsing JSON into FlatBuffers, or generating
JSON from FlatBuffers. See `samples/sample_test.lobster` for an example.

This uses the C++ parser and generator underneath, so should be both fast and
conformant.

<br>
