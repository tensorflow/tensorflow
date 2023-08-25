This directory contains build macros such as `cc_library_with_tflite`,
`java_library_with_tflite`, etc.

`cc_library_with_tflite` generates a `cc_library` target by default.
The target will not use TF Lite in Play Services.

The intent is that the build macros in this directory could be modified to
optionally redirect to a different implementation of TF Lite C and C++ APIs
(for example, one built into the underlying operating system platform).
