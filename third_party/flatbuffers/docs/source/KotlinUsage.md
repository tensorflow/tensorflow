Use in Kotlin    {#flatbuffers_guide_use_kotlin}
==============

## Before you get started

Before diving into the FlatBuffers usage in Kotlin, it should be noted that
the [Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide to
general FlatBuffers usage in all of the supported languages (including K).

This page is designed to cover the nuances of FlatBuffers usage, specific to Kotlin.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## Kotlin and FlatBuffers Java code location

Code generated for Kotlin currently uses the flatbuffers java runtime library. That means that Kotlin generated code can only have Java virtual machine as target architecture (which includes Android). Kotlin Native and Kotlin.js are currently not supported.

The code for the FlatBuffers Java library can be found at
`flatbuffers/java/com/google/flatbuffers`. You can browse the library on the
[FlatBuffers GitHub page](https://github.com/google/flatbuffers/tree/master/
java/com/google/flatbuffers).

## Testing FlatBuffers Kotlin

The test code for Java is located in [KotlinTest.java](https://github.com/google
/flatbuffers/blob/master/tests/KotlinTest.kt).

To run the tests, use  [KotlinTest.sh](https://github.com/google/
flatbuffers/blob/master/tests/KotlinTest.sh) shell script.

*Note: These scripts require that [Kotlin](https://kotlinlang.org/) is installed.*

## Using the FlatBuffers Kotlin library

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in Kotlin.*

FlatBuffers supports reading and writing binary FlatBuffers in Kotlin.

To use FlatBuffers in your own code, first generate Java classes from your
schema with the `--kotlin` option to `flatc`.
Then you can include both FlatBuffers and the generated code to read
or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in Kotlin:
First, import the library and generated code. Then, you read a FlatBuffer binary
file into a `ByteArray`.  You then turn the `ByteArray` into a `ByteBuffer`, which you
pass to the `getRootAsMyRootType` function:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.kt}
    import MyGame.Example.*
    import com.google.flatbuffers.FlatBufferBuilder

    // This snippet ignores exceptions for brevity.
    val data = RandomAccessFile(File("monsterdata_test.mon"), "r").use {
        val temp = ByteArray(it.length().toInt())
        it.readFully(temp)
        temp
    }

    val bb = ByteBuffer.wrap(data)
    val monster = Monster.getRootAsMonster(bb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can access the data from the `Monster monster`:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.kt}
    val hp = monster.hp
    val pos = monster.pos!!;
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



## Differences between Kotlin and Java code

Kotlin generated code was designed to be as close as possible to the java counterpart, as for now, we only support kotlin on java virtual machine. So the differences in implementation and usage are basically the ones introduced by the Kotlin language itself. You can find more in-depth information [here](https://kotlinlang.org/docs/reference/comparison-to-java.html).

The most obvious ones are:

* Fields as accessed as Kotlin [properties](https://kotlinlang.org/docs/reference/properties.html)
* Static methods are accessed in [companion object](https://kotlinlang.org/docs/reference/classes.html#companion-objects)