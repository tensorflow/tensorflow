# FlatBuffers for Dart

This package is used to read and write FlatBuffer files in Dart.

Most consumers will want to use the [`flatc`](https://github.com/google/flatbuffers)
compiler to generate Dart code from a FlatBuffers IDL schema.  For example, the
`monster_my_game.sample_generated.dart` was generated with `flatc` from
`monster.fbs` in the example folder. The generated classes can be used to read
or write binary files that are interoperable with other languages and platforms
supported by FlatBuffers, as illustrated in the `example.dart` in the
examples folder.

Additional documentation and examples are available [at the FlatBuffers site](https://google.github.io/flatbuffers/index.html)