## 3rd party XLA devices

This directory is intended as a place for 3rd party XLA devices which are _not_
integrated into the public repository.

By adding entries to the BUILD target in this directory, a third party device
can be included as a dependency of the JIT subsystem.

For integration into the unit test system, see the files:

-   tensorflow/compiler/tests/plugin.bzl
-   tensorflow/compiler/xla/tests/plugin.bzl
