"""These are the same as Bazel's native cc_libraries."""

_cc_binary = native.cc_binary
_cc_import = native.cc_import
_cc_library = native.cc_library
_cc_shared_library = native.cc_shared_library
_cc_test = native.cc_test

cc_binary = _cc_binary
cc_import = _cc_import
cc_library = _cc_library
cc_shared_library = _cc_shared_library
cc_test = _cc_test
