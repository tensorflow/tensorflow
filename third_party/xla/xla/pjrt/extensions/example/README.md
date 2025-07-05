# Example Extension

This directory provides an example of how to produce a PJRT C API extension.
The example_extension files show how to instantiate an extension in C, and
then, optionally, how to set up your C-API extension for dependency injection
of a CPP type.

The example extension provides a single method, `ExampleMethod`, which takes
an integer as input and prints it to the console. The extension is implemented
in C++ and then wrapped in a C API. This allows the extension to be used by
any language that supports the PJRT C API.

The example extension also demonstrates how to use the PJRT C API's
dependency injection feature. This feature allows the extension to be
instantiated with a C++ object that provides the implementation of the
extension's methods. This object is passed as an optional handle across the C
API. This allows the extension to be easily implemented at the CPP layer.
