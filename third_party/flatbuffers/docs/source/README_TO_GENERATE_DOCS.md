## Prerequisites

To generate the docs for FlatBuffers from the source files, you
will first need to install two programs.

1. You will need to install `doxygen`. See
   [Download Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html).

2. You will need to install `doxypypy` to format python comments appropriately.
   Install it from [here](https://github.com/Feneric/doxypypy).

*Note: You will need both `doxygen` and `doxypypy` to be in your
[PATH](https://en.wikipedia.org/wiki/PATH_(variable)) environment variable.*

After you have both of those files installed and in your path, you need to
set up the `py_filter` to invoke `doxypypy` from `doxygen`.

Follow the steps
[here](https://github.com/Feneric/doxypypy#invoking-doxypypy-from-doxygen).

## Generating Docs

Run the following commands to generate the docs:

`cd flatbuffers/docs/source`
`doxygen`

The output is placed in `flatbuffers/docs/html`.

*Note: The Go API Reference code must be generated ahead of time. For
instructions on how to regenerated this file, please read the comments
in `GoApi.md`.*
