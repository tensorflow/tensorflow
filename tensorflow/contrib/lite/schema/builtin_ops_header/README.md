# Builtin Ops Header Generator.

This directory contains a code generator to generate a pure C header for
builtin op definition.

Whenever you add a new builtin op, please execute:

```sh
bazel run \
  //tensorflow/contrib/lite/schema/builtin_ops_header:generate > \
  tensorflow/contrib/lite/builtin_ops.h
```
