# TensorFlow -> TOSA Compiler Bridge.
# See:
#   https://developer.mlplatform.org/w/tosa/
#   https://github.com/llvm/llvm-project/blob/master/mlir/docs/Dialects/TOSA.md

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    includes = [
    ],
    packages = [
    ],
)

package_group(
    name = "friends",
    includes = [
        ":internal",
    ],
    packages = [
    ],
)

# This can be removed once the package contains real targets.
# It is merely needed to satisfy Bazel's need to expand exclusions
# to something.
cc_library(
    name = "placeholder",
    srcs = [],
    hdrs = [],
)
