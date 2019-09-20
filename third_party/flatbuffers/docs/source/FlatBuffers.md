FlatBuffers    {#flatbuffers_index}
===========

# Overview {#flatbuffers_overview}

[FlatBuffers](@ref flatbuffers_overview) is an efficient cross platform
serialization library for C++, C#, C, Go, Java, Kotlin, JavaScript, Lobster, Lua, TypeScript, PHP, Python, and Rust.
It was originally created at Google for game development and other
performance-critical applications.

It is available as Open Source on [GitHub](http://github.com/google/flatbuffers)
under the Apache license, v2 (see LICENSE.txt).

## Why use FlatBuffers?

-   **Access to serialized data without parsing/unpacking** - What sets
    FlatBuffers apart is that it represents hierarchical data in a flat
    binary buffer in such a way that it can still be accessed directly
    without parsing/unpacking, while also still supporting data
    structure evolution (forwards/backwards compatibility).

-   **Memory efficiency and speed** - The only memory needed to access
    your data is that of the buffer. It requires 0 additional allocations
    (in C++, other languages may vary). FlatBuffers is also very
    suitable for use with mmap (or streaming), requiring only part of the
    buffer to be in memory. Access is close to the speed of raw
    struct access with only one extra indirection (a kind of vtable) to
    allow for format evolution and optional fields. It is aimed at
    projects where spending time and space (many memory allocations) to
    be able to access or construct serialized data is undesirable, such
    as in games or any other performance sensitive applications. See the
    [benchmarks](@ref flatbuffers_benchmarks) for details.

-   **Flexible** - Optional fields means not only do you get great
    forwards and backwards compatibility (increasingly important for
    long-lived games: don't have to update all data with each new
    version!). It also means you have a lot of choice in what data you
    write and what data you don't, and how you design data structures.

-   **Tiny code footprint** - Small amounts of generated code, and just
    a single small header as the minimum dependency, which is very easy
    to integrate. Again, see the benchmark section for details.

-   **Strongly typed** - Errors happen at compile time rather than
    manually having to write repetitive and error prone run-time checks.
    Useful code can be generated for you.

-   **Convenient to use** - Generated C++ code allows for terse access
    & construction code. Then there's optional functionality for parsing
    schemas and JSON-like text representations at runtime efficiently if
    needed (faster and more memory efficient than other JSON
    parsers).

    Java, Kotlin and Go code supports object-reuse. C# has efficient struct based
    accessors.

-   **Cross platform code with no dependencies** - C++ code will work
    with any recent gcc/clang and VS2010. Comes with build files for the tests &
    samples (Android .mk files, and cmake for all other platforms).

### Why not use Protocol Buffers, or .. ?

Protocol Buffers is indeed relatively similar to FlatBuffers,
with the primary difference being that FlatBuffers does not need a parsing/
unpacking step to a secondary representation before you can
access data, often coupled with per-object memory allocation. The code
is an order of magnitude bigger, too. Protocol Buffers has neither optional
text import/export nor schema language features like unions.

### But all the cool kids use JSON!

JSON is very readable (which is why we use it as our optional text
format) and very convenient when used together with dynamically typed
languages (such as JavaScript). When serializing data from statically
typed languages, however, JSON not only has the obvious drawback of runtime
inefficiency, but also forces you to write *more* code to access data
(counterintuitively) due to its dynamic-typing serialization system.
In this context, it is only a better choice for systems that have very
little to no information ahead of time about what data needs to be stored.

If you do need to store data that doesn't fit a schema, FlatBuffers also
offers a schema-less (self-describing) version!

Read more about the "why" of FlatBuffers in the
[white paper](@ref flatbuffers_white_paper).

### Who uses FlatBuffers?
-   [Cocos2d-x](http://www.cocos2d-x.org/), the #1 open source mobile game
    engine, uses it to serialize all their
    [game data](http://www.cocos2d-x.org/reference/native-cpp/V3.5/d7/d2d/namespaceflatbuffers.html).
-   [Facebook](http://facebook.com/) uses it for client-server communication in
    their Android app. They have a nice
    [article](https://code.facebook.com/posts/872547912839369/improving-facebook-s-performance-on-android-with-flatbuffers/)
    explaining how it speeds up loading their posts.
-   [Fun Propulsion Labs](https://developers.google.com/games/#Tools)
    at Google uses it extensively in all their libraries and games.

## Usage in brief

This section is a quick rundown of how to use this system. Subsequent
sections provide a more in-depth usage guide.

-   Write a schema file that allows you to define the data structures
    you may want to serialize. Fields can have a scalar type
    (ints/floats of all sizes), or they can be a: string; array of any type;
    reference to yet another object; or, a set of possible objects (unions).
    Fields are optional and have defaults, so they don't need to be
    present for every object instance.

-   Use `flatc` (the FlatBuffer compiler) to generate a C++ header (or
    Java/Kotlin/C#/Go/Python.. classes) with helper classes to access and construct
    serialized data. This header (say `mydata_generated.h`) only depends on
    `flatbuffers.h`, which defines the core functionality.

-   Use the `FlatBufferBuilder` class to construct a flat binary buffer.
    The generated functions allow you to add objects to this
    buffer recursively, often as simply as making a single function call.

-   Store or send your buffer somewhere!

-   When reading it back, you can obtain the pointer to the root object
    from the binary buffer, and from there traverse it conveniently
    in-place with `object->field()`.

## In-depth documentation

-   How to [build the compiler](@ref flatbuffers_guide_building) and samples on
    various platforms.
-   How to [use the compiler](@ref flatbuffers_guide_using_schema_compiler).
-   How to [write a schema](@ref flatbuffers_guide_writing_schema).
-   How to [use the generated C++ code](@ref flatbuffers_guide_use_cpp) in your
    own programs.
-   How to [use the generated Java/C# code](@ref flatbuffers_guide_use_java_c-sharp)
    in your own programs.
-   How to [use the generated Kotlin code](@ref flatbuffers_guide_use_kotlin)
        in your own programs.
-   How to [use the generated Go code](@ref flatbuffers_guide_use_go) in your
    own programs.
-   How to [use the generated Lua code](@ref flatbuffers_guide_use_lua) in your
    own programs.
-   How to [use the generated JavaScript code](@ref flatbuffers_guide_use_javascript) in your
    own programs.
-   How to [use the generated TypeScript code](@ref flatbuffers_guide_use_typescript) in your
    own programs.
-   How to [use FlatBuffers in C with `flatcc`](@ref flatbuffers_guide_use_c) in your
    own programs.
-   How to [use the generated Lobster code](@ref flatbuffers_guide_use_lobster) in your
    own programs.
-   How to [use the generated Rust code](@ref flatbuffers_guide_use_rust) in your
    own programs.
-   [Support matrix](@ref flatbuffers_support) for platforms/languages/features.
-   Some [benchmarks](@ref flatbuffers_benchmarks) showing the advantage of
    using FlatBuffers.
-   A [white paper](@ref flatbuffers_white_paper) explaining the "why" of
    FlatBuffers.
-   How to use the [schema-less](@ref flexbuffers) version of
    FlatBuffers.
-   A description of the [internals](@ref flatbuffers_internals) of FlatBuffers.
-   A formal [grammar](@ref flatbuffers_grammar) of the schema language.

## Online resources

-   [GitHub repository](http://github.com/google/flatbuffers)
-   [Landing page](http://google.github.io/flatbuffers)
-   [FlatBuffers Google Group](https://groups.google.com/forum/#!forum/flatbuffers)
-   [FlatBuffers Issues Tracker](http://github.com/google/flatbuffers/issues)
-   Independent implementations & tools:
    - [FlatCC](https://github.com/dvidelabs/flatcc) Alternative FlatBuffers
      parser, code generator and runtime all in C.
-   Videos:
    - Colt's [DevByte](https://www.youtube.com/watch?v=iQTxMkSJ1dQ).
    - GDC 2015 [Lightning Talk](https://www.youtube.com/watch?v=olmL1fUnQAQ).
    - FlatBuffers for [Go](https://www.youtube.com/watch?v=-BPVId_lA5w).
    - Evolution of FlatBuffers
      [visualization](https://www.youtube.com/watch?v=a0QE0xS8rKM).
-   Useful documentation created by others:
    - [FlatBuffers in Go](https://rwinslow.com/tags/flatbuffers/)
    - [FlatBuffers in Android](http://frogermcs.github.io/flatbuffers-in-android-introdution/)
    - [Parsing JSON to FlatBuffers in Java](http://frogermcs.github.io/json-parsing-with-flatbuffers-in-android/)
    - [FlatBuffers in Unity](http://exiin.com/blog/flatbuffers-for-unity-sample-code/)
