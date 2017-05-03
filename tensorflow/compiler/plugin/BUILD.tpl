#
# Copy this file to //tensorflow/compiler/plugin/BUILD and replace
# the actual attribute with the library target of an XLA plugin to
# include in the main TF output
#

package(
    default_visibility = ["//visibility:public"],
)

alias(
    name = "plugin",
    actual = "//tensorflow/compiler/my_plugin:my_plugin_library_target",
)