licenses(["notice"])  # Apache 2.0

#
# In order to include a loadable XLA plugin in the unit test
# runfiles, this file should be copied to 'BUILD' and have the
# data attribute updated to include one or more targets that
# describe the binary and its associated data files.
#
# See also //tensorflow/compiler/tests/plugin_config.tpl
#

package(default_visibility = ["//visibility:public"])

py_library(
    name = "deps",
    data = ["//...."],
)
