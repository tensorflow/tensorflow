# Builtin Ops Header Generator.

This directory contains a code generator to generate a pure C header for
builtin op definition.

Whenever you add a new builtin op, please execute:

```sh
bazel run \
  //tensorflow/lite/schema/builtin_ops_header:generate > \
  tensorflow/lite/builtin_ops.h &&
bazel run \
  //tensorflow/lite/schema/builtin_ops_list:generate > \
  tensorflow/lite/core/shims/builtin_ops_list.inc
```
