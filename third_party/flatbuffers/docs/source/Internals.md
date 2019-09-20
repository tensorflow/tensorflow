FlatBuffer Internals    {#flatbuffers_internals}
====================

This section is entirely optional for the use of FlatBuffers. In normal
usage, you should never need the information contained herein. If you're
interested however, it should give you more of an appreciation of why
FlatBuffers is both efficient and convenient.

### Format components

A FlatBuffer is a binary file and in-memory format consisting mostly of
scalars of various sizes, all aligned to their own size. Each scalar is
also always represented in little-endian format, as this corresponds to
all commonly used CPUs today. FlatBuffers will also work on big-endian
machines, but will be slightly slower because of additional
byte-swap intrinsics.

It is assumed that the following conditions are met, to ensure
cross-platform interoperability:
- The binary `IEEE-754` format is used for floating-point numbers.
- The `two's complemented` representation is used for signed integers.
- The endianness is the same for floating-point numbers as for integers.

On purpose, the format leaves a lot of details about where exactly
things live in memory undefined, e.g. fields in a table can have any
order, and objects to some extent can be stored in many orders. This is
because the format doesn't need this information to be efficient, and it
leaves room for optimization and extension (for example, fields can be
packed in a way that is most compact). Instead, the format is defined in
terms of offsets and adjacency only. This may mean two different
implementations may produce different binaries given the same input
values, and this is perfectly valid.

### Format identification

The format also doesn't contain information for format identification
and versioning, which is also by design. FlatBuffers is a statically typed
system, meaning the user of a buffer needs to know what kind of buffer
it is. FlatBuffers can of course be wrapped inside other containers
where needed, or you can use its union feature to dynamically identify
multiple possible sub-objects stored. Additionally, it can be used
together with the schema parser if full reflective capabilities are
desired.

Versioning is something that is intrinsically part of the format (the
optionality / extensibility of fields), so the format itself does not
need a version number (it's a meta-format, in a sense). We're hoping
that this format can accommodate all data needed. If format breaking
changes are ever necessary, it would become a new kind of format rather
than just a variation.

### Offsets

The most important and generic offset type (see `flatbuffers.h`) is
`uoffset_t`, which is currently always a `uint32_t`, and is used to
refer to all tables/unions/strings/vectors (these are never stored
in-line). 32bit is
intentional, since we want to keep the format binary compatible between
32 and 64bit systems, and a 64bit offset would bloat the size for almost
all uses. A version of this format with 64bit (or 16bit) offsets is easy to set
when needed. Unsigned means they can only point in one direction, which
typically is forward (towards a higher memory location). Any backwards
offsets will be explicitly marked as such.

The format starts with an `uoffset_t` to the root object in the buffer.

We have two kinds of objects, structs and tables.

### Structs

These are the simplest, and as mentioned, intended for simple data that
benefits from being extra efficient and doesn't need versioning /
extensibility. They are always stored inline in their parent (a struct,
table, or vector) for maximum compactness. Structs define a consistent
memory layout where all components are aligned to their size, and
structs aligned to their largest scalar member. This is done independent
of the alignment rules of the underlying compiler to guarantee a cross
platform compatible layout. This layout is then enforced in the generated
code.

### Tables

Unlike structs, these are not stored in inline in their parent, but are
referred to by offset.

They start with an `soffset_t` to a vtable. This is a signed version of
`uoffset_t`, since vtables may be stored anywhere relative to the object.
This offset is substracted (not added) from the object start to arrive at
the vtable start. This offset is followed by all the
fields as aligned scalars (or offsets). Unlike structs, not all fields
need to be present. There is no set order and layout.

To be able to access fields regardless of these uncertainties, we go
through a vtable of offsets. Vtables are shared between any objects that
happen to have the same vtable values.

The elements of a vtable are all of type `voffset_t`, which is
a `uint16_t`. The first element is the size of the vtable in bytes,
including the size element. The second one is the size of the object, in bytes
(including the vtable offset). This size could be used for streaming, to know
how many bytes to read to be able to access all *inline* fields of the object.
The remaining elements are the N offsets, where N is the amount of fields
declared in the schema when the code that constructed this buffer was
compiled (thus, the size of the table is N + 2).

All accessor functions in the generated code for tables contain the
offset into this table as a constant. This offset is checked against the
first field (the number of elements), to protect against newer code
reading older data. If this offset is out of range, or the vtable entry
is 0, that means the field is not present in this object, and the
default value is return. Otherwise, the entry is used as offset to the
field to be read.

### Strings and Vectors

Strings are simply a vector of bytes, and are always
null-terminated. Vectors are stored as contiguous aligned scalar
elements prefixed by a 32bit element count (not including any
null termination). Neither is stored inline in their parent, but are referred to
by offset.

### Construction

The current implementation constructs these buffers backwards (starting
at the highest memory address of the buffer), since
that significantly reduces the amount of bookkeeping and simplifies the
construction API.

### Code example

Here's an example of the code that gets generated for the `samples/monster.fbs`.
What follows is the entire file, broken up by comments:

    // automatically generated, do not modify

    #include "flatbuffers/flatbuffers.h"

    namespace MyGame {
    namespace Sample {

Nested namespace support.

    enum {
      Color_Red = 0,
      Color_Green = 1,
      Color_Blue = 2,
    };

    inline const char **EnumNamesColor() {
      static const char *names[] = { "Red", "Green", "Blue", nullptr };
      return names;
    }

    inline const char *EnumNameColor(int e) { return EnumNamesColor()[e]; }

Enums and convenient reverse lookup.

    enum {
      Any_NONE = 0,
      Any_Monster = 1,
    };

    inline const char **EnumNamesAny() {
      static const char *names[] = { "NONE", "Monster", nullptr };
      return names;
    }

    inline const char *EnumNameAny(int e) { return EnumNamesAny()[e]; }

Unions share a lot with enums.

    struct Vec3;
    struct Monster;

Predeclare all data types since circular references between types are allowed
(circular references between object are not, though).

    FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(4) Vec3 {
     private:
      float x_;
      float y_;
      float z_;

     public:
      Vec3(float x, float y, float z)
        : x_(flatbuffers::EndianScalar(x)), y_(flatbuffers::EndianScalar(y)), z_(flatbuffers::EndianScalar(z)) {}

      float x() const { return flatbuffers::EndianScalar(x_); }
      float y() const { return flatbuffers::EndianScalar(y_); }
      float z() const { return flatbuffers::EndianScalar(z_); }
    };
    FLATBUFFERS_STRUCT_END(Vec3, 12);

These ugly macros do a couple of things: they turn off any padding the compiler
might normally do, since we add padding manually (though none in this example),
and they enforce alignment chosen by FlatBuffers. This ensures the layout of
this struct will look the same regardless of compiler and platform. Note that
the fields are private: this is because these store little endian scalars
regardless of platform (since this is part of the serialized data).
`EndianScalar` then converts back and forth, which is a no-op on all current
mobile and desktop platforms, and a single machine instruction on the few
remaining big endian platforms.

    struct Monster : private flatbuffers::Table {
      const Vec3 *pos() const { return GetStruct<const Vec3 *>(4); }
      int16_t mana() const { return GetField<int16_t>(6, 150); }
      int16_t hp() const { return GetField<int16_t>(8, 100); }
      const flatbuffers::String *name() const { return GetPointer<const flatbuffers::String *>(10); }
      const flatbuffers::Vector<uint8_t> *inventory() const { return GetPointer<const flatbuffers::Vector<uint8_t> *>(14); }
      int8_t color() const { return GetField<int8_t>(16, 2); }
    };

Tables are a bit more complicated. A table accessor struct is used to point at
the serialized data for a table, which always starts with an offset to its
vtable. It derives from `Table`, which contains the `GetField` helper functions.
GetField takes a vtable offset, and a default value. It will look in the vtable
at that offset. If the offset is out of bounds (data from an older version) or
the vtable entry is 0, the field is not present and the default is returned.
Otherwise, it uses the entry as an offset into the table to locate the field.

    struct MonsterBuilder {
      flatbuffers::FlatBufferBuilder &fbb_;
      flatbuffers::uoffset_t start_;
      void add_pos(const Vec3 *pos) { fbb_.AddStruct(4, pos); }
      void add_mana(int16_t mana) { fbb_.AddElement<int16_t>(6, mana, 150); }
      void add_hp(int16_t hp) { fbb_.AddElement<int16_t>(8, hp, 100); }
      void add_name(flatbuffers::Offset<flatbuffers::String> name) { fbb_.AddOffset(10, name); }
      void add_inventory(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> inventory) { fbb_.AddOffset(14, inventory); }
      void add_color(int8_t color) { fbb_.AddElement<int8_t>(16, color, 2); }
      MonsterBuilder(flatbuffers::FlatBufferBuilder &_fbb) : fbb_(_fbb) { start_ = fbb_.StartTable(); }
      flatbuffers::Offset<Monster> Finish() { return flatbuffers::Offset<Monster>(fbb_.EndTable(start_, 7)); }
    };

`MonsterBuilder` is the base helper struct to construct a table using a
`FlatBufferBuilder`. You can add the fields in any order, and the `Finish`
call will ensure the correct vtable gets generated.

    inline flatbuffers::Offset<Monster> CreateMonster(flatbuffers::FlatBufferBuilder &_fbb,
                                                      const Vec3 *pos, int16_t mana,
                                                      int16_t hp,
                                                      flatbuffers::Offset<flatbuffers::String> name,
                                                      flatbuffers::Offset<flatbuffers::Vector<uint8_t>> inventory,
                                                      int8_t color) {
      MonsterBuilder builder_(_fbb);
      builder_.add_inventory(inventory);
      builder_.add_name(name);
      builder_.add_pos(pos);
      builder_.add_hp(hp);
      builder_.add_mana(mana);
      builder_.add_color(color);
      return builder_.Finish();
    }

`CreateMonster` is a convenience function that calls all functions in
`MonsterBuilder` above for you. Note that if you pass values which are
defaults as arguments, it will not actually construct that field, so
you can probably use this function instead of the builder class in
almost all cases.

    inline const Monster *GetMonster(const void *buf) { return flatbuffers::GetRoot<Monster>(buf); }

This function is only generated for the root table type, to be able to
start traversing a FlatBuffer from a raw buffer pointer.

    }; // namespace MyGame
    }; // namespace Sample

### Encoding example.

Below is a sample encoding for the following JSON corresponding to the above
schema:

    { pos: { x: 1, y: 2, z: 3 }, name: "fred", hp: 50 }

Resulting in this binary buffer:

    // Start of the buffer:
    uint32_t 20  // Offset to the root table.

    // Start of the vtable. Not shared in this example, but could be:
    uint16_t 16 // Size of table, starting from here.
    uint16_t 22 // Size of object inline data.
    uint16_t 4, 0, 20, 16, 0, 0  // Offsets to fields from start of (root) table, 0 for not present.

    // Start of the root table:
    int32_t 16     // Offset to vtable used (default negative direction)
    float 1, 2, 3  // the Vec3 struct, inline.
    uint32_t 8     // Offset to the name string.
    int16_t 50     // hp field.
    int16_t 0      // Padding for alignment.

    // Start of name string:
    uint32_t 4  // Length of string.
    int8_t 'f', 'r', 'e', 'd', 0, 0, 0, 0  // Text + 0 termination + padding.

Note that this not the only possible encoding, since the writer has some
flexibility in which of the children of root object to write first (though in
this case there's only one string), and what order to write the fields in.
Different orders may also cause different alignments to happen.

### Additional reading.

The author of the C language implementation has made a similar
[document](https://github.com/dvidelabs/flatcc/blob/master/doc/binary-format.md#flatbuffers-binary-format)
that may further help clarify the format.

# FlexBuffers

The [schema-less](@ref flexbuffers) version of FlatBuffers have their
own encoding, detailed here.

It shares many properties mentioned above, in that all data is accessed
over offsets, all scalars are aligned to their own size, and
all data is always stored in little endian format.

One difference is that FlexBuffers are built front to back, so children are
stored before parents, and the root of the data starts at the last byte.

Another difference is that scalar data is stored with a variable number of bits
(8/16/32/64). The current width is always determined by the *parent*, i.e. if
the scalar sits in a vector, the vector determines the bit width for all
elements at once. Selecting the minimum bit width for a particular vector is
something the encoder does automatically and thus is typically of no concern
to the user, though being aware of this feature (and not sticking a double in
the same vector as a bunch of byte sized elements) is helpful for efficiency.

Unlike FlatBuffers there is only one kind of offset, and that is an unsigned
integer indicating the number of bytes in a negative direction from the address
of itself (where the offset is stored).

### Vectors

The representation of the vector is at the core of how FlexBuffers works (since
maps are really just a combination of 2 vectors), so it is worth starting there.

As mentioned, a vector is governed by a single bit width (supplied by its
parent). This includes the size field. For example, a vector that stores the
integer values `1, 2, 3` is encoded as follows:

    uint8_t 3, 1, 2, 3, 4, 4, 4

The first `3` is the size field, and is placed before the vector (an offset
from the parent to this vector points to the first element, not the size
field, so the size field is effectively at index -1).
Since this is an untyped vector `SL_VECTOR`, it is followed by 3 type
bytes (one per element of the vector), which are always following the vector,
and are always a uint8_t even if the vector is made up of bigger scalars.

### Types

A type byte is made up of 2 components (see flexbuffers.h for exact values):

* 2 lower bits representing the bit-width of the child (8, 16, 32, 64).
  This is only used if the child is accessed over an offset, such as a child
  vector. It is ignored for inline types.
* 6 bits representing the actual type (see flexbuffers.h).

Thus, in this example `4` means 8 bit child (value 0, unused, since the value is
in-line), type `SL_INT` (value 1).

### Typed Vectors

These are like the Vectors above, but omit the type bytes. The type is instead
determined by the vector type supplied by the parent. Typed vectors are only
available for a subset of types for which these savings can be significant,
namely inline signed/unsigned integers (`TYPE_VECTOR_INT` / `TYPE_VECTOR_UINT`),
floats (`TYPE_VECTOR_FLOAT`), and keys (`TYPE_VECTOR_KEY`, see below).

Additionally, for scalars, there are fixed length vectors of sizes 2 / 3 / 4
that don't store the size (`TYPE_VECTOR_INT2` etc.), for an additional savings
in space when storing common vector or color data.

### Scalars

FlexBuffers supports integers (`TYPE_INT` and `TYPE_UINT`) and floats
(`TYPE_FLOAT`), available in the bit-widths mentioned above. They can be stored
both inline and over an offset (`TYPE_INDIRECT_*`).

The offset version is useful to encode costly 64bit (or even 32bit) quantities
into vectors / maps of smaller sizes, and to share / repeat a value multiple
times.

### Booleans and Nulls

Booleans (`TYPE_BOOL`) and nulls (`TYPE_NULL`) are encoded as inlined unsigned integers.

### Blobs, Strings and Keys.

A blob (`TYPE_BLOB`) is encoded similar to a vector, with one difference: the
elements are always `uint8_t`. The parent bit width only determines the width of
the size field, allowing blobs to be large without the elements being large.

Strings (`TYPE_STRING`) are similar to blobs, except they have an additional 0
termination byte for convenience, and they MUST be UTF-8 encoded (since an
accessor in a language that does not support pointers to UTF-8 data may have to
convert them to a native string type).

A "Key" (`TYPE_KEY`) is similar to a string, but doesn't store the size
field. They're so named because they are used with maps, which don't care
for the size, and can thus be even more compact. Unlike strings, keys cannot
contain bytes of value 0 as part of their data (size can only be determined by
`strlen`), so while you can use them outside the context of maps if you so
desire, you're usually better off with strings.

### Maps

A map (`TYPE_MAP`) is like an (untyped) vector, but with 2 prefixes before the
size field:

| index | field                                                        |
| ----: | :----------------------------------------------------------- |
| -3    | An offset to the keys vector (may be shared between tables). |
| -2    | Byte width of the keys vector.                               |
| -1    | Size (from here on it is compatible with `TYPE_VECTOR`)      |
| 0     | Elements.                                                    |
| Size  | Types.                                                       |

Since a map is otherwise the same as a vector, it can be iterated like
a vector (which is probably faster than lookup by key).

The keys vector is a typed vector of keys. Both the keys and corresponding
values *have* to be stored in sorted order (as determined by `strcmp`), such
that lookups can be made using binary search.

The reason the key vector is a seperate structure from the value vector is
such that it can be shared between multiple value vectors, and also to
allow it to be treated as its own individual vector in code.

An example map { foo: 13, bar: 14 } would be encoded as:

    0 : uint8_t 'b', 'a', 'r', 0
    4 : uint8_t 'f', 'o', 'o', 0
    8 : uint8_t 2      // key vector of size 2
    // key vector offset points here
    9 : uint8_t 9, 6   // offsets to bar_key and foo_key
    11: uint8_t 2, 1   // offset to key vector, and its byte width
    13: uint8_t 2      // value vector of size
    // value vector offset points here
    14: uint8_t 14, 13 // values
    16: uint8_t 4, 4   // types

### The root

As mentioned, the root starts at the end of the buffer.
The last uint8_t is the width in bytes of the root (normally the parent
determines the width, but the root has no parent). The uint8_t before this is
the type of the root, and the bytes before that are the root value (of the
number of bytes specified by the last byte).

So for example, the integer value `13` as root would be:

    uint8_t 13, 4, 1    // Value, type, root byte width.


<br>
