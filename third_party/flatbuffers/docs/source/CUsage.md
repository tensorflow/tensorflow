Use in C    {#flatbuffers_guide_use_c}
==========

The C language binding exists in a separate project named [FlatCC](https://github.com/dvidelabs/flatcc).

The `flatcc` C schema compiler can generate code offline as well as
online via a C library. It can also generate buffer verifiers and fast
JSON parsers, printers.

Great care has been taken to ensure compatibily with the main `flatc`
project.

## General Documention

- [Tutorial](@ref flatbuffers_guide_tutorial) - select C as language
  when scrolling down
- [FlatCC Guide](https://github.com/dvidelabs/flatcc#flatcc-flatbuffers-in-c-for-c)
- [The C Builder Interface](https://github.com/dvidelabs/flatcc/blob/master/doc/builder.md#the-builder-interface)
- [The Monster Sample in C](https://github.com/dvidelabs/flatcc/blob/master/samples/monster/monster.c)
- [GitHub](https://github.com/dvidelabs/flatcc)


## Supported Platforms

- Ubuntu (clang / gcc, ninja / gnu make)
- OS-X (clang / gcc, ninja / gnu make)
- Windows MSVC 2010, 2013, 2015

CI builds recent versions of gcc, clang and MSVC on OS-X, Ubuntu, and
Windows, and occasionally older compiler versions. See main project [Status](https://github.com/dvidelabs/flatcc#status).

Other platforms may well work, including Centos, but are not tested
regularly.

The monster sample project was specifically written for C99 in order to
follow the C++ version and for that reason it will not work with MSVC
2010.

## Modular Object Creation

In the tutorial we used the call `Monster_create_as_root` to create the
root buffer object since this is easier in simple use cases. Sometimes
we need more modularity so we can reuse a function to create nested
tables and root tables the same way. For this we need the
`flatcc_builder_buffer_create_call`. It is best to keep `flatcc_builder`
calls isolated at the top driver level, so we get:

<div class="language-c">
~~~{.c}
  ns(Monster_ref_t) create_orc(flatcc_builder_t *B)
  {
    // ... same as in the tutorial.
    return s(Monster_create(B, ...));
  }

  void create_monster_buffer()
  {
      uint8_t *buf;
      size_t size;
      flatcc_builder_t builder, *B;

      // Initialize the builder object.
      B = &builder;
      flatcc_builder_init(B);
      // Only use `buffer_create` without `create/start/end_as_root`.
      flatcc_builder_buffer_create(create_orc(B));
      // Allocate and copy buffer to user memory.
      buf = flatcc_builder_finalize_buffer(B, &size);
      // ... write the buffer to disk or network, or something.

      free(buf);
      flatcc_builder_clear(B);
  }
~~~
</div>

The same principle applies with `start/end` vs `start/end_as_root` in
the top-down approach.


## Top Down Example

The tutorial uses a bottom up approach. In C it is also possible to use
a top-down approach by starting and ending objects nested within each
other. In the tutorial there is no deep nesting, so the difference is
limited, but it shows the idea:

<div class="language-c">
<br>
~~~{.c}
  uint8_t treasure[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  size_t treasure_count = c_vec_len(treasure);
  ns(Weapon_ref_t) axe;

  // NOTE: if we use end_as_root, we MUST also start as root.
  ns(Monster_start_as_root(B));
  ns(Monster_pos_create(B, 1.0f, 2.0f, 3.0f));
  ns(Monster_hp_add(B, 300));
  ns(Monster_mana_add(B, 150));
  // We use create_str instead of add because we have no existing string reference.
  ns(Monster_name_create_str(B, "Orc"));
  // Again we use create because we no existing vector object, only a C-array.
  ns(Monster_inventory_create(B, treasure, treasure_count));
  ns(Monster_color_add(B, ns(Color_Red)));
  if (1) {
      ns(Monster_weapons_start(B));
      ns(Monster_weapons_push_create(B, flatbuffers_string_create_str(B, "Sword"), 3));
      // We reuse the axe object later. Note that we dereference a pointer
      // because push always returns a short-term pointer to the stored element.
      // We could also have created the axe object first and simply pushed it.
      axe = *ns(Monster_weapons_push_create(B, flatbuffers_string_create_str(B, "Axe"), 5));
      ns(Monster_weapons_end(B));
  } else {
      // We can have more control with the table elements added to a vector:
      //
      ns(Monster_weapons_start(B));
      ns(Monster_weapons_push_start(B));
      ns(Weapon_name_create_str(B, "Sword"));
      ns(Weapon_damage_add(B, 3));
      ns(Monster_weapons_push_end(B));
      ns(Monster_weapons_push_start(B));
      ns(Monster_weapons_push_start(B));
      ns(Weapon_name_create_str(B, "Axe"));
      ns(Weapon_damage_add(B, 5));
      axe = *ns(Monster_weapons_push_end(B));
      ns(Monster_weapons_end(B));
  }
  // Unions can get their type by using a type-specific add/create/start method.
  ns(Monster_equipped_Weapon_add(B, axe));

  ns(Monster_end_as_root(B));
~~~
</div>


## Basic Reflection

The C-API does support reading binary schema (.bfbs)
files via code generated from the `reflection.fbs` schema, and an
[example usage](https://github.com/dvidelabs/flatcc/tree/master/samples/reflection)
shows how to use this. The reflection schema files are pre-generated
in the [runtime distribution](https://github.com/dvidelabs/flatcc/tree/master/include/flatcc/reflection).


## Mutations and Reflection

The C-API does not support mutating reflection like C++ does, nor does
the reader interface support mutating scalars (and it is generally
unsafe to do so even after verification).

The generated reader interface supports sorting vectors in-place after
casting them to a mutating type because it is not practical to do so
while building a buffer. This is covered in the builder documentation.  
The reflection example makes use of this feature to look up objects by
name.

It is possible to build new buffers using complex objects from existing
buffers as source. This can be very efficient due to direct copy
semantics without endian conversion or temporary stack allocation.

Scalars, structs and strings can be used as source, as well vectors of
these.

It is currently not possible to use an existing table or vector of table
as source, but it would be possible to add support for this at some
point.


## Namespaces

The `FLATBUFFERS_WRAP_NAMESPACE` approach used in the tutorial is convenient
when each function has a very long namespace prefix. But it isn't always
the best approach. If the namespace is absent, or simple and
informative, we might as well use the prefix directly. The
[reflection example](https://github.com/dvidelabs/flatcc/blob/master/samples/reflection/bfbs2json.c)
mentioned above uses this approach.


## Checking for Present Members

Not all languages support testing if a field is present, but in C we can
elaborate the reader section of the tutorial with tests for this. Recall
that `mana` was set to the default value `150` and therefore shouldn't
be present.

<div class="language-c">
~~~{.c}
  int hp_present = ns(Monster_hp_is_present(monster)); // 1
  int mana_present = ns(Monster_mana_is_present(monster)); // 0
~~~
</div>

## Alternative ways to add a Union

In the tutorial we used a single call to add a union.  Here we show
different ways to accomplish the same thing. The last form is rarely
used, but is the low-level way to do it. It can be used to group small
values together in the table by adding type and data at different
points in time.

<div class="language-c">
~~~{.c}
   ns(Equipment_union_ref_t) equipped = ns(Equipment_as_Weapon(axe));
   ns(Monster_equipped_add(B, equipped));
   // or alternatively
   ns(Monster_equipped_Weapon_add(B, axe);
   // or alternatively
   ns(Monster_equipped_add_type(B, ns(Equipment_Weapon));
   ns(Monster_equipped_add_member(B, axe));
~~~
</div>

## Why not integrate with the `flatc` tool?

[It was considered how the C code generator could be integrated into the
`flatc` tool](https://github.com/dvidelabs/flatcc/issues/1), but it
would either require that the standalone C implementation of the schema
compiler was dropped, or it would lead to excessive code duplication, or
a complicated intermediate representation would have to be invented.
Neither of these alternatives are very attractive, and it isn't a big
deal to use the `flatcc` tool instead of `flatc` given that the
FlatBuffers C runtime library needs to be made available regardless.


