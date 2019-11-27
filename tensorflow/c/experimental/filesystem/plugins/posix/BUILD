# Experimental posix filesystem plugin.

package(
    licenses = ["notice"],  # Apache 2.0
)

# Although this target results in a shared object that will be loaded at
# runtime, this target must be a `cc_library` instead of a `cc_binary`. Making
# it a `cc_binary` requires `linkshared = True`. In turn, this brings in several
# TensorFlow symbols under `tensorflow::` namespace, for which we have no ABI
# guarantees. Hence, in order to maintain ABI compatibility, this is marked as a
# `cc_library` for now and we will revisit in the future.
# TODO(mihaimaruseac): Determine if `cc_binary` makes more sense (when all
# filesystems are converted and BUILD files are refactored to be modular).
# TODO(b/144585140): The helpers should be separated into a different BUILD target
# but doing that would result in symbols not being visible when loading plugin.
# Revisit this once POSIX filesystem completely lands. See also the other TODO.
# This also has the unfortunate effect that both versions of copy_file get
# compiled, regardless of which one actually gets used!
cc_library(
    name = "posix_filesystem",
    srcs = [
        "posix_filesystem.cc",
        "posix_filesystem_helper.cc",
        "posix_filesystem_helper.h",
        "copy_file.h",
    ] + select({
        "//tensorflow:linux_x86_64": ["copy_file_linux.cc"],
        "//conditions:default": ["copy_file_portable.cc"],
    }),
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
    ],
)
