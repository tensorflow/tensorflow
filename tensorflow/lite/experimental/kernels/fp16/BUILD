# Experimental FP16-on-CPU implementation of a few select layers.

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels/internal:tensor",
    ],
)
