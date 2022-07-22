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

To develop across MLIR core and TensorFlow, it is useful to override the repo
to use a local version instead of fetching from head. This can be achieved by
setting up your local repository for Bazel build. For this you will need a
temporary directory that will be "overlaid" with you LLVM source directory and
the Bazel files:

```sh
LLVM_SRC=... # this the path to the LLVM local source directory you intend to use.
LLVM_BAZEL_OVERLAY=${LLVM_SRC}/bazel # Note: this can be anywhere
mkdir -p ${LLVM_BAZEL_OVERLAY}
# This will symlink your LLVM sources with the BUILD files to be usable by Bazel.
python ${LLVM_SRC}/utils/bazel/overlay_directories.py \
    --src ${LLVM_SRC} \
    --overlay ${LLVM_SRC}/utils/bazel/llvm-project-overlay/ \
    --target ${LLVM_BAZEL_OVERLAY}
touch ${LLVM_BAZEL_OVERLAY}/BUILD.bazel ${LLVM_BAZEL_OVERLAY}/WORKSPACE
# The complete list is "AArch64", "AMDGPU", "ARM", "NVPTX", "PowerPC", "RISCV", "SystemZ", "X86"
echo 'llvm_targets = ["X86"]' > ${LLVM_BAZEL_OVERLAY}/llvm/targets.bzl
```

You can then use this overlay to build TensorFlow:

```
bazel build --override_repository=llvm-project=$LLVM_BAZEL_OVERLAY \
  -c opt tensorflow/compiler/mlir:tf-opt
```
