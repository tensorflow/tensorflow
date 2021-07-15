package(
    default_visibility = [
        "//tensorflow/cc/experimental/libtf:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tfrt",
    srcs = [
        "tfrt.cc",
    ],
    hdrs = [
        "tfrt.h",
    ],
    deps = [
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/cc/experimental/libtf",
        "//tensorflow/cc/experimental/libtf/runtime",
    ],
)
