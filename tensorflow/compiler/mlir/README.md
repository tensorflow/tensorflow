# MLIR dialects and utilities for TensorFlow, TensorFlow Lite and XLA.

This module contains the MLIR
([Multi-Level Intermediate Representation](https://mlir.llvm.org))
dialects and utilities for

1. TensorFlow
2. XLA
3. TF Lite

See [MLIR's website](https://mlir.llvm.org) for complete documentation.

## Getting started

Building dialects and utilities here follow the standard approach using
`bazel` as the rest of TensorFlow.

### Using local LLVM repo

To develop across MLIR core and TensorFlow, it is useful to override the repo to
use a local version instead of fetching from head. This can be achieved by
setting up your local repository for Bazel build. For this you will need to
create bazel workspace and build files:

```sh
LLVM_SRC=... # this the path to the LLVM local source directory you intend to use.
touch ${LLVM_SRC}/BUILD.bazel ${LLVM_SRC}/WORKSPACE
```

You can then use this overlay to build TensorFlow:

```
bazel build --override_repository="llvm-raw=${LLVM_SRC}" \
  -c opt tensorflow/compiler/mlir:tf-opt
```
