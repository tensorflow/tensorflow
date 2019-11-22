package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "resource",
    srcs = [
        "resource_variable.cc",
        "static_hashtable.cc",
    ],
    hdrs = [
        "lookup_interfaces.h",
        "lookup_util.h",
        "resource_base.h",
        "resource_variable.h",
        "static_hashtable.h",
    ],
    deps = [
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
    ],
)
