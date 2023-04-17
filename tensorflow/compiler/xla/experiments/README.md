# XLA Experiments

This folder is intended to serve as a place to collaborate on code related to
the XLA compiler, but will not end up being a part of the compiler itself.

As such, the code here is not necessarily production quality, and should not be
depended on from other parts of the compiler.

Some examples of code appropriate for this folder are:

*   microbenchmarks that allow us to better understand various architectures
*   scripts that help with developing specific features of the compiler, which
    might remain useful after the feature is complete (general tools should
    instead go into the xla/tools directory)
*   experimental code transformations that are not yet integrated into the
    compiler

## Visibility

As a result of the nature of the content in this folder, its build visibility
is intentionally kept private.

If you need something from here elsewhere, the recommended approach is to move
it to a more suitable and production-supported location.