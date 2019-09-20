Use in TypeScript    {#flatbuffers_guide_use_typescript}
=================

## Before you get started

Before diving into the FlatBuffers usage in TypeScript, it should be noted that
the [Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide to
general FlatBuffers usage in all of the supported languages
(including TypeScript). This page is specifically designed to cover the nuances
of FlatBuffers usage in TypeScript.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## FlatBuffers TypeScript library code location

The code for the FlatBuffers TypeScript library can be found at
`flatbuffers/js` with typings available at `@types/flatbuffers`.

## Testing the FlatBuffers TypeScript library

To run the tests, use the [TypeScriptTest.sh](https://github.com/google/
flatbuffers/blob/master/tests/TypeScriptTest.sh) shell script.

*Note: The TypeScript test file requires [Node.js](https://nodejs.org/en/).*

## Using the FlatBuffers TypeScript libary

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in TypeScript.*

FlatBuffers supports both reading and writing FlatBuffers in TypeScript.

To use FlatBuffers in your own code, first generate TypeScript classes from your
schema with the `--ts` option to `flatc`. Then you can include both FlatBuffers
and the generated code to read or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in TypeScript:
First, include the library and generated code. Then read the file into an
`Uint8Array`. Make a `flatbuffers.ByteBuffer` out of the `Uint8Array`, and pass
the ByteBuffer to the `getRootAsMonster` function.

~~~{.ts}
  // note: import flatbuffers with your desired import method

  import { MyGame } from './monster_generated';

  let data = new Uint8Array(fs.readFileSync('monster.dat'));
  let buf = new flatbuffers.ByteBuffer(data);

  let monster = MyGame.Example.Monster.getRootAsMonster(buf);
~~~

Now you can access values like this:

~~~{.ts}
  let hp = monster.hp();
  let pos = monster.pos();
~~~

## Text parsing FlatBuffers in TypeScript

There currently is no support for parsing text (Schema's and JSON) directly
from TypeScript.
