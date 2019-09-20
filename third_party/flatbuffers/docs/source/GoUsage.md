Use in Go    {#flatbuffers_guide_use_go}
=========

## Before you get started

Before diving into the FlatBuffers usage in Go, it should be noted that
the [Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide
to general FlatBuffers usage in all of the supported languages (including Go).
This page is designed to cover the nuances of FlatBuffers usage, specific to
Go.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## FlatBuffers Go library code location

The code for the FlatBuffers Go library can be found at
`flatbuffers/go`. You can browse the library code on the [FlatBuffers
GitHub page](https://github.com/google/flatbuffers/tree/master/go).

## Testing the FlatBuffers Go library

The code to test the Go library can be found at `flatbuffers/tests`.
The test code itself is located in [go_test.go](https://github.com/google/
flatbuffers/blob/master/tests/go_test.go).

To run the tests, use the [GoTest.sh](https://github.com/google/flatbuffers/
blob/master/tests/GoTest.sh) shell script.

*Note: The shell script requires [Go](https://golang.org/doc/install) to
be installed.*

## Using the FlatBuffers Go library

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in Go.*

FlatBuffers supports reading and writing binary FlatBuffers in Go.

To use FlatBuffers in your own code, first generate Go classes from your
schema with the `--go` option to `flatc`. Then you can include both FlatBuffers
and the generated code to read or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in Go: First,
include the library and generated code. Then read a FlatBuffer binary file into
a `[]byte`, which you pass to the `GetRootAsMonster` function:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.go}
    import (
       example "MyGame/Example"
       flatbuffers "github.com/google/flatbuffers/go"

       io/ioutil
    )

    buf, err := ioutil.ReadFile("monster.dat")
    // handle err
    monster := example.GetRootAsMonster(buf, 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can access values like this:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.go}
    hp := monster.Hp()
    pos := monster.Pos(nil)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


In some cases it's necessary to modify values in an existing FlatBuffer in place (without creating a copy). For this reason, scalar fields of a Flatbuffer table or struct can be mutated.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.go}
    monster := example.GetRootAsMonster(buf, 0)

    // Set table field.
    if ok := monster.MutateHp(10); !ok {
      panic("failed to mutate Hp")
    }

    // Set struct field.
    monster.Pos().MutateZ(4)

    // This mutation will fail because the mana field is not available in
    // the buffer. It should be set when creating the buffer.
    if ok := monster.MutateMana(20); !ok {
      panic("failed to mutate Hp")
    }
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The term `mutate` is used instead of `set` to indicate that this is a special use case. All mutate functions return a boolean value which is false if the field we're trying to mutate is not available in the buffer.

## Text Parsing

There currently is no support for parsing text (Schema's and JSON) directly
from Go, though you could use the C++ parser through cgo. Please see the
C++ documentation for more on text parsing.

<br>
