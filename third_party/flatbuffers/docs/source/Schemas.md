Writing a schema    {#flatbuffers_guide_writing_schema}
================

The syntax of the schema language (aka IDL, [Interface Definition Language][])
should look quite familiar to users of any of the C family of
languages, and also to users of other IDLs. Let's look at an example
first:

    // example IDL file

    namespace MyGame;

    attribute "priority";

    enum Color : byte { Red = 1, Green, Blue }

    union Any { Monster, Weapon, Pickup }

    struct Vec3 {
      x:float;
      y:float;
      z:float;
    }

    table Monster {
      pos:Vec3;
      mana:short = 150;
      hp:short = 100;
      name:string;
      friendly:bool = false (deprecated, priority: 1);
      inventory:[ubyte];
      color:Color = Blue;
      test:Any;
    }

    root_type Monster;

(`Weapon` & `Pickup` not defined as part of this example).

### Tables

Tables are the main way of defining objects in FlatBuffers, and consist
of a name (here `Monster`) and a list of fields. Each field has a name,
a type, and optionally a default value (if omitted, it defaults to `0` /
`NULL`).

Each field is optional: It does not have to appear in the wire
representation, and you can choose to omit fields for each individual
object. As a result, you have the flexibility to add fields without fear of
bloating your data. This design is also FlatBuffer's mechanism for forward
and backwards compatibility. Note that:

-   You can add new fields in the schema ONLY at the end of a table
    definition. Older data will still
    read correctly, and give you the default value when read. Older code
    will simply ignore the new field.
    If you want to have flexibility to use any order for fields in your
    schema, you can manually assign ids (much like Protocol Buffers),
    see the `id` attribute below.

-   You cannot delete fields you don't use anymore from the schema,
    but you can simply
    stop writing them into your data for almost the same effect.
    Additionally you can mark them as `deprecated` as in the example
    above, which will prevent the generation of accessors in the
    generated C++, as a way to enforce the field not being used any more.
    (careful: this may break code!).

-   You may change field names and table names, if you're ok with your
    code breaking until you've renamed them there too.

See "Schema evolution examples" below for more on this
topic.

### Structs

Similar to a table, only now none of the fields are optional (so no defaults
either), and fields may not be added or be deprecated. Structs may only contain
scalars or other structs. Use this for
simple objects where you are very sure no changes will ever be made
(as quite clear in the example `Vec3`). Structs use less memory than
tables and are even faster to access (they are always stored in-line in their
parent object, and use no virtual table).

### Types

Built-in scalar types are

-   8 bit: `byte` (`int8`), `ubyte` (`uint8`), `bool`

-   16 bit: `short` (`int16`), `ushort` (`uint16`)

-   32 bit: `int` (`int32`), `uint` (`uint32`), `float` (`float32`)

-   64 bit: `long` (`int64`), `ulong` (`uint64`), `double` (`float64`)

The type names in parentheses are alias names such that for example
`uint8` can be used in place of `ubyte`, and `int32` can be used in
place of `int` without affecting code generation.

Built-in non-scalar types:

-   Vector of any other type (denoted with `[type]`). Nesting vectors
    is not supported, instead you can wrap the inner vector in a table.

-   `string`, which may only hold UTF-8 or 7-bit ASCII. For other text encodings
    or general binary data use vectors (`[byte]` or `[ubyte]`) instead.

-   References to other tables or structs, enums or unions (see
    below).

You can't change types of fields once they're used, with the exception
of same-size data where a `reinterpret_cast` would give you a desirable result,
e.g. you could change a `uint` to an `int` if no values in current data use the
high bit yet.

### Arrays

Arrays are a convenience short-hand for a fixed-length collection of elements.
Arrays can be used to replace the following schema:

    struct Vec3 {
        x:float;
        y:float;
        z:float;
    }

with the following schema:

    struct Vec3 {
        v:[float:3];
    }

Both representations are binary equivalent.

Arrays are currently only supported in a `struct`.

### (Default) Values

Values are a sequence of digits. Values may be optionally followed by a decimal
point (`.`) and more digits, for float constants, or optionally prefixed by
a `-`. Floats may also be in scientific notation; optionally ending with an `e`
or `E`, followed by a `+` or `-` and more digits.

Only scalar values can have defaults, non-scalar (string/vector/table) fields
default to `NULL` when not present.

You generally do not want to change default values after they're initially
defined. Fields that have the default value are not actually stored in the
serialized data (see also Gotchas below) but are generated in code,
so when you change the default, you'd
now get a different value than from code generated from an older version of
the schema. There are situations, however, where this may be
desirable, especially if you can ensure a simultaneous rebuild of
all code.

### Enums

Define a sequence of named constants, each with a given value, or
increasing by one from the previous one. The default first value
is `0`. As you can see in the enum declaration, you specify the underlying
integral type of the enum with `:` (in this case `byte`), which then determines
the type of any fields declared with this enum type.

Only integer types are allowed, i.e. `byte`, `ubyte`, `short` `ushort`, `int`,
`uint`, `long` and `ulong`.

Typically, enum values should only ever be added, never removed (there is no
deprecation for enums). This requires code to handle forwards compatibility
itself, by handling unknown enum values.

### Unions

Unions share a lot of properties with enums, but instead of new names
for constants, you use names of tables. You can then declare
a union field, which can hold a reference to any of those types, and
additionally a field with the suffix `_type` is generated that holds
the corresponding enum value, allowing you to know which type to cast
to at runtime.

It's possible to give an alias name to a type union. This way a type can even be
used to mean different things depending on the name used:

    table PointPosition { x:uint; y:uint; }
    table MarkerPosition {}
    union Position {
      Start:MarkerPosition,
      Point:PointPosition,
      Finish:MarkerPosition
    }

Unions contain a special `NONE` marker to denote that no value is stored so that
name cannot be used as an alias.

Unions are a good way to be able to send multiple message types as a FlatBuffer.
Note that because a union field is really two fields, it must always be
part of a table, it cannot be the root of a FlatBuffer by itself.

If you have a need to distinguish between different FlatBuffers in a more
open-ended way, for example for use as files, see the file identification
feature below.

There is an experimental support only in C++ for a vector of unions
(and types). In the example IDL file above, use [Any] to add a
vector of Any to Monster table.

### Namespaces

These will generate the corresponding namespace in C++ for all helper
code, and packages in Java. You can use `.` to specify nested namespaces /
packages.

### Includes

You can include other schemas files in your current one, e.g.:

    include "mydefinitions.fbs";

This makes it easier to refer to types defined elsewhere. `include`
automatically ensures each file is parsed just once, even when referred to
more than once.

When using the `flatc` compiler to generate code for schema definitions,
only definitions in the current file will be generated, not those from the
included files (those you still generate separately).

### Root type

This declares what you consider to be the root table (or struct) of the
serialized data. This is particularly important for parsing JSON data,
which doesn't include object type information.

### File identification and extension

Typically, a FlatBuffer binary buffer is not self-describing, i.e. it
needs you to know its schema to parse it correctly. But if you
want to use a FlatBuffer as a file format, it would be convenient
to be able to have a "magic number" in there, like most file formats
have, to be able to do a sanity check to see if you're reading the
kind of file you're expecting.

Now, you can always prefix a FlatBuffer with your own file header,
but FlatBuffers has a built-in way to add an identifier to a
FlatBuffer that takes up minimal space, and keeps the buffer
compatible with buffers that don't have such an identifier.

You can specify in a schema, similar to `root_type`, that you intend
for this type of FlatBuffer to be used as a file format:

    file_identifier "MYFI";

Identifiers must always be exactly 4 characters long. These 4 characters
will end up as bytes at offsets 4-7 (inclusive) in the buffer.

For any schema that has such an identifier, `flatc` will automatically
add the identifier to any binaries it generates (with `-b`),
and generated calls like `FinishMonsterBuffer` also add the identifier.
If you have specified an identifier and wish to generate a buffer
without one, you can always still do so by calling
`FlatBufferBuilder::Finish` explicitly.

After loading a buffer, you can use a call like
`MonsterBufferHasIdentifier` to check if the identifier is present.

Note that this is best for open-ended uses such as files. If you simply wanted
to send one of a set of possible messages over a network for example, you'd
be better off with a union.

Additionally, by default `flatc` will output binary files as `.bin`.
This declaration in the schema will change that to whatever you want:

    file_extension "ext";

### RPC interface declarations

You can declare RPC calls in a schema, that define a set of functions
that take a FlatBuffer as an argument (the request) and return a FlatBuffer
as the response (both of which must be table types):

    rpc_service MonsterStorage {
      Store(Monster):StoreResponse;
      Retrieve(MonsterId):Monster;
    }

What code this produces and how it is used depends on language and RPC system
used, there is preliminary support for GRPC through the `--grpc` code generator,
see `grpc/tests` for an example.

### Comments & documentation

May be written as in most C-based languages. Additionally, a triple
comment (`///`) on a line by itself signals that a comment is documentation
for whatever is declared on the line after it
(table/struct/field/enum/union/element), and the comment is output
in the corresponding C++ code. Multiple such lines per item are allowed.

### Attributes

Attributes may be attached to a declaration, behind a field, or after
the name of a table/struct/enum/union. These may either have a value or
not. Some attributes like `deprecated` are understood by the compiler;
user defined ones need to be declared with the attribute declaration
(like `priority` in the example above), and are
available to query if you parse the schema at runtime.
This is useful if you write your own code generators/editors etc., and
you wish to add additional information specific to your tool (such as a
help text).

Current understood attributes:

-   `id: n` (on a table field): manually set the field identifier to `n`.
    If you use this attribute, you must use it on ALL fields of this table,
    and the numbers must be a contiguous range from 0 onwards.
    Additionally, since a union type effectively adds two fields, its
    id must be that of the second field (the first field is the type
    field and not explicitly declared in the schema).
    For example, if the last field before the union field had id 6,
    the union field should have id 8, and the unions type field will
    implicitly be 7.
    IDs allow the fields to be placed in any order in the schema.
    When a new field is added to the schema it must use the next available ID.
-   `deprecated` (on a field): do not generate accessors for this field
    anymore, code should stop using this data. Old data may still contain this
    field, but it won't be accessible anymore by newer code. Note that if you
    deprecate a field that was previous required, old code may fail to validate
    new data (when using the optional verifier).
-   `required` (on a non-scalar table field): this field must always be set.
    By default, all fields are optional, i.e. may be left out. This is
    desirable, as it helps with forwards/backwards compatibility, and
    flexibility of data structures. It is also a burden on the reading code,
    since for non-scalar fields it requires you to check against NULL and
    take appropriate action. By specifying this field, you force code that
    constructs FlatBuffers to ensure this field is initialized, so the reading
    code may access it directly, without checking for NULL. If the constructing
    code does not initialize this field, they will get an assert, and also
    the verifier will fail on buffers that have missing required fields. Note
    that if you add this attribute to an existing field, this will only be
    valid if existing data always contains this field / existing code always
    writes this field.
-   `force_align: size` (on a struct): force the alignment of this struct
    to be something higher than what it is naturally aligned to. Causes
    these structs to be aligned to that amount inside a buffer, IF that
    buffer is allocated with that alignment (which is not necessarily
    the case for buffers accessed directly inside a `FlatBufferBuilder`).
    Note: currently not guaranteed to have an effect when used with
    `--object-api`, since that may allocate objects at alignments less than
    what you specify with `force_align`.
-   `bit_flags` (on an unsigned enum): the values of this field indicate bits,
    meaning that any unsigned value N specified in the schema will end up
    representing 1<<N, or if you don't specify values at all, you'll get
    the sequence 1, 2, 4, 8, ...
-   `nested_flatbuffer: "table_name"` (on a field): this indicates that the field
    (which must be a vector of ubyte) contains flatbuffer data, for which the
    root type is given by `table_name`. The generated code will then produce
    a convenient accessor for the nested FlatBuffer.
-   `flexbuffer` (on a field): this indicates that the field
    (which must be a vector of ubyte) contains flexbuffer data. The generated
    code will then produce a convenient accessor for the FlexBuffer root.
-   `key` (on a field): this field is meant to be used as a key when sorting
    a vector of the type of table it sits in. Can be used for in-place
    binary search.
-   `hash` (on a field). This is an (un)signed 32/64 bit integer field, whose
    value during JSON parsing is allowed to be a string, which will then be
    stored as its hash. The value of attribute is the hashing algorithm to
    use, one of `fnv1_32` `fnv1_64` `fnv1a_32` `fnv1a_64`.
-   `original_order` (on a table): since elements in a table do not need
    to be stored in any particular order, they are often optimized for
    space by sorting them to size. This attribute stops that from happening.
    There should generally not be any reason to use this flag.
-   'native_*'.  Several attributes have been added to support the [C++ object
    Based API](@ref flatbuffers_cpp_object_based_api).  All such attributes
    are prefixed with the term "native_".


## JSON Parsing

The same parser that parses the schema declarations above is also able
to parse JSON objects that conform to this schema. So, unlike other JSON
parsers, this parser is strongly typed, and parses directly into a FlatBuffer
(see the compiler documentation on how to do this from the command line, or
the C++ documentation on how to do this at runtime).

Besides needing a schema, there are a few other changes to how it parses
JSON:

-   It accepts field names with and without quotes, like many JSON parsers
    already do. It outputs them without quotes as well, though can be made
    to output them using the `strict_json` flag.
-   If a field has an enum type, the parser will recognize symbolic enum
    values (with or without quotes) instead of numbers, e.g.
    `field: EnumVal`. If a field is of integral type, you can still use
    symbolic names, but values need to be prefixed with their type and
    need to be quoted, e.g. `field: "Enum.EnumVal"`. For enums
    representing flags, you may place multiple inside a string
    separated by spaces to OR them, e.g.
    `field: "EnumVal1 EnumVal2"` or `field: "Enum.EnumVal1 Enum.EnumVal2"`.
-   Similarly, for unions, these need to specified with two fields much like
    you do when serializing from code. E.g. for a field `foo`, you must
    add a field `foo_type: FooOne` right before the `foo` field, where
    `FooOne` would be the table out of the union you want to use.
-   A field that has the value `null` (e.g. `field: null`) is intended to
    have the default value for that field (thus has the same effect as if
    that field wasn't specified at all).
-   It has some built in conversion functions, so you can write for example
    `rad(180)` where ever you'd normally write `3.14159`.
    Currently supports the following functions: `rad`, `deg`, `cos`, `sin`,
    `tan`, `acos`, `asin`, `atan`.

When parsing JSON, it recognizes the following escape codes in strings:

-   `\n` - linefeed.
-   `\t` - tab.
-   `\r` - carriage return.
-   `\b` - backspace.
-   `\f` - form feed.
-   `\"` - double quote.
-   `\\` - backslash.
-   `\/` - forward slash.
-   `\uXXXX` - 16-bit unicode code point, converted to the equivalent UTF-8
    representation.
-   `\xXX` - 8-bit binary hexadecimal number XX. This is the only one that is
     not in the JSON spec (see http://json.org/), but is needed to be able to
     encode arbitrary binary in strings to text and back without losing
     information (e.g. the byte 0xFF can't be represented in standard JSON).

It also generates these escape codes back again when generating JSON from a
binary representation.

When parsing numbers, the parser is more flexible than JSON.
A format of numeric literals is more close to the C/C++.
According to the [grammar](@ref flatbuffers_grammar), it accepts the following
numerical literals:

-   An integer literal can have any number of leading zero `0` digits.
    Unlike C/C++, the parser ignores a leading zero, not interpreting it as the
    beginning of the octal number.
    The numbers `[081, -00094]` are equal to `[81, -94]`  decimal integers.
-   The parser accepts unsigned and signed hexadecimal integer numbers.
    For example: `[0x123, +0x45, -0x67]` are equal to `[291, 69, -103]` decimals.
-   The format of float-point numbers is fully compatible with C/C++ format.
    If a modern C++ compiler is used the parser accepts hexadecimal and special
    floating-point literals as well:
    `[-1.0, 2., .3e0, 3.e4, 0x21.34p-5, -inf, nan]`.

    The following conventions for floating-point numbers are used:
    - The exponent suffix of hexadecimal floating-point number is mandatory.
    - Parsed `NaN` converted to unsigned IEEE-754 `quiet-NaN` value.

    Extended floating-point support was tested with:
    - x64 Windows: `MSVC2015` and higher.
    - x64 Linux: `LLVM 6.0`, `GCC 4.9` and higher.

    For details, see [Use in C++](@ref flatbuffers_guide_use_cpp) section.

-   For compatibility with a JSON lint tool all numeric literals of scalar
    fields can be wrapped to quoted string:
    `"1", "2.0", "0x48A", "0x0C.0Ep-1", "-inf", "true"`.

## Guidelines

### Efficiency

FlatBuffers is all about efficiency, but to realize that efficiency you
require an efficient schema. There are usually multiple choices on
how to represent data that have vastly different size characteristics.

It is very common nowadays to represent any kind of data as dictionaries
(as in e.g. JSON), because of its flexibility and extensibility. While
it is possible to emulate this in FlatBuffers (as a vector
of tables with key and value(s)), this is a bad match for a strongly
typed system like FlatBuffers, leading to relatively large binaries.
FlatBuffer tables are more flexible than classes/structs in most systems,
since having a large number of fields only few of which are actually
used is still efficient. You should thus try to organize your data
as much as possible such that you can use tables where you might be
tempted to use a dictionary.

Similarly, strings as values should only be used when they are
truely open-ended. If you can, always use an enum instead.

FlatBuffers doesn't have inheritance, so the way to represent a set
of related data structures is a union. Unions do have a cost however,
so an alternative to a union is to have a single table that has
all the fields of all the data structures you are trying to
represent, if they are relatively similar / share many fields.
Again, this is efficient because optional fields are cheap.

FlatBuffers supports the full range of integer sizes, so try to pick
the smallest size needed, rather than defaulting to int/long.

Remember that you can share data (refer to the same string/table
within a buffer), so factoring out repeating data into its own
data structure may be worth it.

### Style guide

Identifiers in a schema are meant to translate to many different programming
languages, so using the style of your "main" language is generally a bad idea.

For this reason, below is a suggested style guide to adhere to, to keep schemas
consistent for interoperation regardless of the target language.

Where possible, the code generators for specific languages will generate
identifiers that adhere to the language style, based on the schema identifiers.

- Table, struct, enum and rpc names (types): UpperCamelCase.
- Table and struct field names: snake_case. This is translated to lowerCamelCase
  automatically for some languages, e.g. Java.
- Enum values: UpperCamelCase.
- namespaces: UpperCamelCase.

Formatting (this is less important, but still worth adhering to):

- Opening brace: on the same line as the start of the declaration.
- Spacing: Indent by 2 spaces. None around `:` for types, on both sides for `=`.

For an example, see the schema at the top of this file.

## Gotchas

### Schemas and version control

FlatBuffers relies on new field declarations being added at the end, and earlier
declarations to not be removed, but be marked deprecated when needed. We think
this is an improvement over the manual number assignment that happens in
Protocol Buffers (and which is still an option using the `id` attribute
mentioned above).

One place where this is possibly problematic however is source control. If user
A adds a field, generates new binary data with this new schema, then tries to
commit both to source control after user B already committed a new field also,
and just auto-merges the schema, the binary files are now invalid compared to
the new schema.

The solution of course is that you should not be generating binary data before
your schema changes have been committed, ensuring consistency with the rest of
the world. If this is not practical for you, use explicit field ids, which
should always generate a merge conflict if two people try to allocate the same
id.

### Schema evolution examples

Some examples to clarify what happens as you change a schema:

If we have the following original schema:

    table { a:int; b:int; }

And we extend it:

    table { a:int; b:int; c:int; }

This is ok. Code compiled with the old schema reading data generated with the
new one will simply ignore the presence of the new field. Code compiled with the
new schema reading old data will get the default value for `c` (which is 0
in this case, since it is not specified).

    table { a:int (deprecated); b:int; }

This is also ok. Code compiled with the old schema reading newer data will now
always get the default value for `a` since it is not present. Code compiled
with the new schema now cannot read nor write `a` anymore (any existing code
that tries to do so will result in compile errors), but can still read
old data (they will ignore the field).

    table { c:int a:int; b:int; }

This is NOT ok, as this makes the schemas incompatible. Old code reading newer
data will interpret `c` as if it was `a`, and new code reading old data
accessing `a` will instead receive `b`.

    table { c:int (id: 2); a:int (id: 0); b:int (id: 1); }

This is ok. If your intent was to order/group fields in a way that makes sense
semantically, you can do so using explicit id assignment. Now we are compatible
with the original schema, and the fields can be ordered in any way, as long as
we keep the sequence of ids.

    table { b:int; }

NOT ok. We can only remove a field by deprecation, regardless of wether we use
explicit ids or not.

    table { a:uint; b:uint; }

This is MAYBE ok, and only in the case where the type change is the same size,
like here. If old data never contained any negative numbers, this will be
safe to do.

    table { a:int = 1; b:int = 2; }

Generally NOT ok. Any older data written that had 0 values were not written to
the buffer, and rely on the default value to be recreated. These will now have
those values appear to `1` and `2` instead. There may be cases in which this
is ok, but care must be taken.

    table { aa:int; bb:int; }

Occasionally ok. You've renamed fields, which will break all code (and JSON
files!) that use this schema, but as long as the change is obvious, this is not
incompatible with the actual binary buffers, since those only ever address
fields by id/offset.
<br>

### Testing whether a field is present in a table

Most serialization formats (e.g. JSON or Protocol Buffers) make it very
explicit in the format whether a field is present in an object or not,
allowing you to use this as "extra" information.

In FlatBuffers, this also holds for everything except scalar values.

FlatBuffers by default will not write fields that are equal to the default
value (for scalars), sometimes resulting in a significant space savings.

However, this also means testing whether a field is "present" is somewhat
meaningless, since it does not tell you if the field was actually written by
calling `add_field` style calls, unless you're only interested in this
information for non-default values.

Some `FlatBufferBuilder` implementations have an option called `force_defaults`
that circumvents this behavior, and writes fields even if they are equal to
the default. You can then use `IsFieldPresent` to query this.

Another option that works in all languages is to wrap a scalar field in a
struct. This way it will return null if it is not present. The cool thing
is that structs don't take up any more space than the scalar they represent.

   [Interface Definition Language]: https://en.wikipedia.org/wiki/Interface_description_language
