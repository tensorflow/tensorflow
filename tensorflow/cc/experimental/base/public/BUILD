# Experimental C++ APIs for TensorFlow.
# New TF C++ APIs under the tensorflow::cc namespace aim to guarantee ABI stability.
# Users are expected to compile against public c++ headers, and link against
# libtensorflow (https://www.tensorflow.org/install/lang_c).
# We aim to achieve ABI stability in new C++ APIs by only using types
# on the API surface that:
# 1. Have a header-only implementation
# 2. Are std:: types
# 3. Wrap an opaque C type

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # This is intentionally public
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "runtime",
    hdrs = [
        "runtime.h",
    ],
    deps = [
        ":status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
    ],
)

cc_library(
    name = "runtime_builder",
    hdrs = [
        "runtime_builder.h",
    ],
    deps = [
        ":runtime",
        ":status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
    ],
)

cc_library(
    name = "status",
    hdrs = [
        "status.h",
    ],
    deps = [
        "//tensorflow/c:tf_status",
    ],
)

cc_library(
    name = "tensor",
    hdrs = [
        "tensor.h",
    ],
    deps = [
        ":status",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_tensor",
    ],
)

cc_library(
    name = "tensorhandle",
    hdrs = [
        "tensorhandle.h",
    ],
    deps = [
        ":runtime",
        ":status",
        ":tensor",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
    ],
)
