Use in Java/C#    {#flatbuffers_guide_use_java_c-sharp}
==============

## Before you get started

Before diving into the FlatBuffers usage in Java or C#, it should be noted that
the [Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide to
general FlatBuffers usage in all of the supported languages (including both Java
and C#). This page is designed to cover the nuances of FlatBuffers usage,
specific to Java and C#.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## FlatBuffers Java and C-sharp code location

#### Java

The code for the FlatBuffers Java library can be found at
`flatbuffers/java/com/google/flatbuffers`. You can browse the library on the
[FlatBuffers GitHub page](https://github.com/google/flatbuffers/tree/master/
java/com/google/flatbuffers).

#### C-sharp

The code for the FlatBuffers C# library can be found at
`flatbuffers/net/FlatBuffers`. You can browse the library on the
[FlatBuffers GitHub page](https://github.com/google/flatbuffers/tree/master/net/
FlatBuffers).

## Testing the FlatBuffers Java and C-sharp libraries

The code to test the libraries can be found at `flatbuffers/tests`.

#### Java

The test code for Java is located in [JavaTest.java](https://github.com/google
/flatbuffers/blob/master/tests/JavaTest.java).

To run the tests, use either [JavaTest.sh](https://github.com/google/
flatbuffers/blob/master/tests/JavaTest.sh) or [JavaTest.bat](https://github.com/
google/flatbuffers/blob/master/tests/JavaTest.bat), depending on your operating
system.

*Note: These scripts require that [Java](https://www.oracle.com/java/index.html)
is installed.*

#### C-sharp

The test code for C# is located in the [FlatBuffers.Test](https://github.com/
google/flatbuffers/tree/master/tests/FlatBuffers.Test) subfolder. To run the
tests, open `FlatBuffers.Test.csproj` in [Visual Studio](
https://www.visualstudio.com), and compile/run the project.

Optionally, you can run this using [Mono](http://www.mono-project.com/) instead.
Once you have installed `Mono`, you can run the tests from the command line
by running the following commands from inside the `FlatBuffers.Test` folder:

~~~{.sh}
  mcs *.cs ../MyGame/Example/*.cs ../../net/FlatBuffers/*.cs
  mono Assert.exe
~~~

## Using the FlatBuffers Java (and C#) library

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in Java or C#.*

FlatBuffers supports reading and writing binary FlatBuffers in Java and C#.

To use FlatBuffers in your own code, first generate Java classes from your
schema with the `--java` option to `flatc`. (Or for C# with `--csharp`).
Then you can include both FlatBuffers and the generated code to read
or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in Java:
First, import the library and generated code. Then, you read a FlatBuffer binary
file into a `byte[]`.  You then turn the `byte[]` into a `ByteBuffer`, which you
pass to the `getRootAsMyRootType` function:

*Note: The code here is written from the perspective of Java. Code for both
languages is both generated and used in nearly the exact same way, with only
minor differences. These differences are
[explained in a section below](#differences_in_c-sharp).*

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.java}
    import MyGame.Example.*;
    import com.google.flatbuffers.FlatBufferBuilder;

    // This snippet ignores exceptions for brevity.
    File file = new File("monsterdata_test.mon");
    RandomAccessFile f = new RandomAccessFile(file, "r");
    byte[] data = new byte[(int)f.length()];
    f.readFully(data);
    f.close();

    ByteBuffer bb = ByteBuffer.wrap(data);
    Monster monster = Monster.getRootAsMonster(bb);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can access the data from the `Monster monster`:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.java}
    short hp = monster.hp();
    Vec3 pos = monster.pos();
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<a name="differences_in_c-sharp">
#### Differences in C-sharp
</a>

C# code works almost identically to Java, with only a few minor differences.
You can see an example of C# code in
`tests/FlatBuffers.Test/FlatBuffersExampleTests.cs` or
`samples/SampleBinary.cs`.

First of all, naming follows standard C# style with `PascalCasing` identifiers,
e.g. `GetRootAsMyRootType`. Also, values (except vectors and unions) are
available as properties instead of parameterless accessor methods as in Java.
The performance-enhancing methods to which you can pass an already created
object are prefixed with `Get`, e.g.:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cs}
    // property
    var pos = monster.Pos;

    // method filling a preconstructed object
    var preconstructedPos = new Vec3();
    monster.GetPos(preconstructedPos);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Storing dictionaries in a FlatBuffer

FlatBuffers doesn't support dictionaries natively, but there is support to
emulate their behavior with vectors and binary search, which means you
can have fast lookups directly from a FlatBuffer without having to unpack
your data into a `Dictionary` or similar.

To use it:
-   Designate one of the fields in a table as the "key" field. You do this
    by setting the `key` attribute on this field, e.g.
    `name:string (key)`.
    You may only have one key field, and it must be of string or scalar type.
-   Write out tables of this type as usual, collect their offsets in an
    array.
-   Instead of calling standard generated method,
    e.g.: `Monster.createTestarrayoftablesVector`,
    call `CreateSortedVectorOfMonster` in C# or
    `createSortedVectorOfTables` (from the `FlatBufferBuilder` object) in Java,
    which will first sort all offsets such that the tables they refer to
    are sorted by the key field, then serialize it.
-   Now when you're accessing the FlatBuffer, you can use
    the `ByKey` accessor to access elements of the vector, e.g.:
    `monster.testarrayoftablesByKey("Frodo")` in Java or
    `monster.TestarrayoftablesByKey("Frodo")` in C#,
    which returns an object of the corresponding table type,
    or `null` if not found.
    `ByKey` performs a binary search, so should have a similar
    speed to `Dictionary`, though may be faster because of better caching.
    `ByKey` only works if the vector has been sorted, it will
    likely not find elements if it hasn't been sorted.

## Text parsing

There currently is no support for parsing text (Schema's and JSON) directly
from Java or C#, though you could use the C++ parser through native call
interfaces available to each language. Please see the
C++ documentation for more on text parsing.

<br>
