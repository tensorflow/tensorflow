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
cc_library(
    name = "posix_filesystem",
    srcs = [
        "posix_filesystem.cc",
    ],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
    ],
)
