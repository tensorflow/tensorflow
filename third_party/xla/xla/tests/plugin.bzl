"""Additional XLA devices to be included in the unit test suite."""

# Example:
#
# plugins = {
#   "foo": {
#     "deps": [
#       "//tensorflow/compiler/plugin/foo:foo_lib",
#     ],
#     "disabled_manifest": "tensorflow/compiler/plugin/foo/disabled_test_manifest.txt",
#     "copts": [],
#     "tags": [],
#     "args": []
#     "data": [
#       "//tensorflow/compiler/plugin/foo:disabled_test_manifest.txt",
#     ],
#   },
# }

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

plugins = {}
