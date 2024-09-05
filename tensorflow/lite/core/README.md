This directory contains the "core" part of the TensorFlow Lite runtime library.
The header files in this `tensorflow/lite/core/` directory fall into several
categories.

1.  Public API headers, in the `api` subdirectory `tensorflow/lite/core/api/`

    These are in addition to the other public API headers in `tensorflow/lite/`.

    For example:
    - `tensorflow/lite/core/api/error_reporter.h`
    - `tensorflow/lite/core/api/op_resolver.h`

2.  Private headers that define public API types and functions.
    These headers are each `#include`d from a corresponding public "shim" header
    in `tensorflow/lite/` that forwards to the private header.

    For example:
    - `tensorflow/lite/core/interpreter.h` is a private header file that is
      included from the public "shim" header file `tensorflow/lite/interpeter.h`.

    These private header files should be used as follows: `#include`s from `.cc`
    files in TF Lite itself that are _implementing_ the TF Lite APIs should
    include the "core" TF Lite API headers.  `#include`s from files that are
    just _using_ the regular TF Lite APIs should include the regular public
    headers.

3.  The header file `tensorflow/lite/core/subgraph.h`. This contains
    some experimental APIs.