FlexBuffers    {#flexbuffers}
==========

FlatBuffers was designed around schemas, because when you want maximum
performance and data consistency, strong typing is helpful.

There are however times when you want to store data that doesn't fit a
schema, because you can't know ahead of time what all needs to be stored.

For this, FlatBuffers has a dedicated format, called FlexBuffers.
This is a binary format that can be used in conjunction
with FlatBuffers (by storing a part of a buffer in FlexBuffers
format), or also as its own independent serialization format.

While it loses the strong typing, you retain the most unique advantage
FlatBuffers has over other serialization formats (schema-based or not):
FlexBuffers can also be accessed without parsing / copying / object allocation.
This is a huge win in efficiency / memory friendly-ness, and allows unique
use cases such as mmap-ing large amounts of free-form data.

FlexBuffers' design and implementation allows for a very compact encoding,
combining automatic pooling of strings with automatic sizing of containers to
their smallest possible representation (8/16/32/64 bits). Many values and
offsets can be encoded in just 8 bits. While a schema-less representation is
usually more bulky because of the need to be self-descriptive, FlexBuffers
generates smaller binaries for many cases than regular FlatBuffers.

FlexBuffers is still slower than regular FlatBuffers though, so we recommend to
only use it if you need it.


# Usage in C++

Include the header `flexbuffers.h`, which in turn depends on `flatbuffers.h`
and `util.h`.

To create a buffer:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
flexbuffers::Builder fbb;
fbb.Int(13);
fbb.Finish();
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You create any value, followed by `Finish`. Unlike FlatBuffers which requires
the root value to be a table, here any value can be the root, including a lonely
int value.

You can now access the `std::vector<uint8_t>` that contains the encoded value
as `fbb.GetBuffer()`. Write it, send it, or store it in a parent FlatBuffer. In
this case, the buffer is just 3 bytes in size.

To read this value back, you could just say:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
auto root = flexbuffers::GetRoot(my_buffer);
int64_t i = root.AsInt64();
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FlexBuffers stores ints only as big as needed, so it doesn't differentiate
between different sizes of ints. You can ask for the 64 bit version,
regardless of what you put in. In fact, since you demand to read the root
as an int, if you supply a buffer that actually contains a float, or a
string with numbers in it, it will convert it for you on the fly as well,
or return 0 if it can't. If instead you actually want to know what is inside
the buffer before you access it, you can call `root.GetType()` or `root.IsInt()`
etc.

Here's a slightly more complex value you could write instead of `fbb.Int` above:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
fbb.Map([&]() {
  fbb.Vector("vec", [&]() {
    fbb.Int(-100);
    fbb.String("Fred");
    fbb.IndirectFloat(4.0f);
  });
  fbb.UInt("foo", 100);
});
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This stores the equivalent of the JSON value
`{ vec: [ -100, "Fred", 4.0 ], foo: 100 }`. The root is a dictionary that has
just two key-value pairs, with keys `vec` and `foo`. Unlike FlatBuffers, it
actually has to store these keys in the buffer (which it does only once if
you store multiple such objects, by pooling key values), but also unlike
FlatBuffers it has no restriction on the keys (fields) that you use.

The map constructor uses a C++11 Lambda to group its children, but you can
also use more conventional start/end calls if you prefer.

The first value in the map is a vector. You'll notice that unlike FlatBuffers,
you can use mixed types. There is also a `TypedVector` variant that only
allows a single type, and uses a bit less memory.

`IndirectFloat` is an interesting feature that allows you to store values
by offset rather than inline. Though that doesn't make any visible change
to the user, the consequence is that large values (especially doubles or
64 bit ints) that occur more than once can be shared (see ReuseValue).
Another use case is inside of vectors, where the largest element makes
up the size of all elements (e.g. a single double forces all elements to
64bit), so storing a lot of small integers together with a double is more efficient if the double is indirect.

Accessing it:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
auto map = flexbuffers::GetRoot(my_buffer).AsMap();
map.size();  // 2
auto vec = map["vec"].AsVector();
vec.size();  // 3
vec[0].AsInt64();  // -100;
vec[1].AsString().c_str();  // "Fred";
vec[1].AsInt64();  // 0 (Number parsing failed).
vec[2].AsDouble();  // 4.0
vec[2].AsString().IsTheEmptyString();  // true (Wrong Type).
vec[2].AsString().c_str();  // "" (This still works though).
vec[2].ToString().c_str();  // "4" (Or have it converted).
map["foo"].AsUInt8();  // 100
map["unknown"].IsNull();  // true
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Usage in Java

Java implementation follows the C++ one, closely.

For creating the equivalent of the same JSON `{ vec: [ -100, "Fred", 4.0 ], foo: 100 }`,
one could use the following code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.java}
FlexBuffersBuilder builder = new FlexBuffersBuilder(ByteBuffer.allocate(512),
		                                                FlexBuffersBuilder.BUILDER_FLAG_SHARE_KEYS_AND_STRINGS);
int smap = builder.startMap();
int svec = builder.startVector();
builder.putInt(-100);
builder.putString("Fred");
builder.putFloat(4.0);
builder.endVector("vec", svec, false, false);
builder.putInt("foo", 100);
builder.endMap(null, smap);
ByteBuffer bb = builder.finish();
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, to read the data, just:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.java}
FlexBuffers.Map map = FlexBuffers.getRoot(bb).asMap();
map.size();  // 2
FlexBuffers.Vector vec = map.get("vec").asVector();
vec.size();  // 3
vec.get(0).asLong();  // -100;
vec.get(1).asString();  // "Fred";
vec.get(1).asLong();  // 0 (Number parsing failed).
vec.get(2).asFloat();  // 4.0
vec.get(2).asString().isEmpty();  // true (Wrong Type).
vec.get(2).asString();  // "" (This still works though).
vec.get(2).toString();  // "4.0" (Or have it converted).
map.get("foo").asUInt();  // 100
map.get("unknown").isNull();  // true
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Binary encoding

A description of how FlexBuffers are encoded is in the
[internals](@ref flatbuffers_internals) document.


# Nesting inside a FlatBuffer

You can mark a field as containing a FlexBuffer, e.g.

    a:[ubyte] (flexbuffer);

A special accessor will be generated that allows you to access the root value
directly, e.g. `a_flexbuffer_root().AsInt64()`.


# Efficiency tips

* Vectors generally are a lot more efficient than maps, so prefer them over maps
  when possible for small objects. Instead of a map with keys `x`, `y` and `z`,
  use a vector. Better yet, use a typed vector. Or even better, use a fixed
  size typed vector.
* Maps are backwards compatible with vectors, and can be iterated as such.
  You can iterate either just the values (`map.Values()`), or in parallel with
  the keys vector (`map.Keys()`). If you intend
  to access most or all elements, this is faster than looking up each element
  by key, since that involves a binary search of the key vector.
* When possible, don't mix values that require a big bit width (such as double)
  in a large vector of smaller values, since all elements will take on this
  width. Use `IndirectDouble` when this is a possibility. Note that
  integers automatically use the smallest width possible, i.e. if you ask
  to serialize an int64_t whose value is actually small, you will use less
  bits. Doubles are represented as floats whenever possible losslessly, but
  this is only possible for few values.
  Since nested vectors/maps are stored over offsets, they typically don't
  affect the vector width.
* To store large arrays of byte data, use a blob. If you'd use a typed
  vector, the bit width of the size field may make it use more space than
  expected, and may not be compatible with `memcpy`.
  Similarly, large arrays of (u)int16_t may be better off stored as a
  binary blob if their size could exceed 64k elements.
  Construction and use are otherwise similar to strings.
