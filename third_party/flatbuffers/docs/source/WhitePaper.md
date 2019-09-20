FlatBuffers white paper    {#flatbuffers_white_paper}
=======================

This document tries to shed some light on to the "why" of FlatBuffers, a
new serialization library.

## Motivation

Back in the good old days, performance was all about instructions and
cycles. Nowadays, processing units have run so far ahead of the memory
subsystem, that making an efficient application should start and finish
with thinking about memory. How much you use of it. How you lay it out
and access it. How you allocate it. When you copy it.

Serialization is a pervasive activity in a lot programs, and a common
source of memory inefficiency, with lots of temporary data structures
needed to parse and represent data, and inefficient allocation patterns
and locality.

If it would be possible to do serialization with no temporary objects,
no additional allocation, no copying, and good locality, this could be
of great value. The reason serialization systems usually don't manage
this is because it goes counter to forwards/backwards compatability, and
platform specifics like endianness and alignment.

FlatBuffers is what you get if you try anyway.

In particular, FlatBuffers focus is on mobile hardware (where memory
size and memory bandwidth is even more constrained than on desktop
hardware), and applications that have the highest performance needs:
games.

## FlatBuffers

*This is a summary of FlatBuffers functionality, with some rationale.
A more detailed description can be found in the FlatBuffers
documentation.*

### Summary

A FlatBuffer is a binary buffer containing nested objects (structs,
tables, vectors,..) organized using offsets so that the data can be
traversed in-place just like any pointer-based data structure. Unlike
most in-memory data structures however, it uses strict rules of
alignment and endianness (always little) to ensure these buffers are
cross platform. Additionally, for objects that are tables, FlatBuffers
provides forwards/backwards compatibility and general optionality of
fields, to support most forms of format evolution.

You define your object types in a schema, which can then be compiled to
C++ or Java for low to zero overhead reading & writing.
Optionally, JSON data can be dynamically parsed into buffers.

### Tables

Tables are the cornerstone of FlatBuffers, since format evolution is
essential for most applications of serialization. Typically, dealing
with format changes is something that can be done transparently during
the parsing process of most serialization solutions out there.
But a FlatBuffer isn't parsed before it is accessed.

Tables get around this by using an extra indirection to access fields,
through a *vtable*. Each table comes with a vtable (which may be shared
between multiple tables with the same layout), and contains information
where fields for this particular kind of instance of vtable are stored.
The vtable may also indicate that the field is not present (because this
FlatBuffer was written with an older version of the software, of simply
because the information was not necessary for this instance, or deemed
deprecated), in which case a default value is returned.

Tables have a low overhead in memory (since vtables are small and
shared) and in access cost (an extra indirection), but provide great
flexibility. Tables may even cost less memory than the equivalent
struct, since fields do not need to be stored when they are equal to
their default.

FlatBuffers additionally offers "naked" structs, which do not offer
forwards/backwards compatibility, but can be even smaller (useful for
very small objects that are unlikely to change, like e.g. a coordinate
pair or a RGBA color).

### Schemas

While schemas reduce some generality (you can't just read any data
without having its schema), they have a lot of upsides:

-   Most information about the format can be factored into the generated
    code, reducing memory needed to store data, and time to access it.

-   The strong typing of the data definitions means less error
    checking/handling at runtime (less can go wrong).

-   A schema enables us to access a buffer without parsing.

FlatBuffer schemas are fairly similar to those of the incumbent,
Protocol Buffers, and generally should be readable to those familiar
with the C family of languages. We chose to improve upon the features
offered by .proto files in the following ways:

-   Deprecation of fields instead of manual field id assignment.
    Extending an object in a .proto means hunting for a free slot among
    the numbers (preferring lower numbers since they have a more compact
    representation). Besides being inconvenient, it also makes removing
    fields problematic: you either have to keep them, not making it
    obvious that this field shouldn't be read/written anymore, and still
    generating accessors. Or you remove it, but now you risk that
    there's still old data around that uses that field by the time
    someone reuses that field id, with nasty consequences.

-   Differentiating between tables and structs (see above). Effectively
    all table fields are `optional`, and all struct fields are
    `required`.

-   Having a native vector type instead of `repeated`. This gives you a
    length without having to collect all items, and in the case of
    scalars provides for a more compact representation, and one that
    guarantees adjacency.

-   Having a native `union` type instead of using a series of `optional`
    fields, all of which must be checked individually.

-   Being able to define defaults for all scalars, instead of having to
    deal with their optionality at each access.

-   A parser that can deal with both schemas and data definitions (JSON
    compatible) uniformly.

<br>
