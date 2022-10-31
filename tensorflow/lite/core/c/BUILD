load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_cc_library_with_c_headers_test",
    "tflite_copts",
)

package(
    licenses = ["notice"],
)

exports_files([
    "c_api.h",
])

# LINT.IfChange(c_api)
tflite_cc_library_with_c_headers_test(
    name = "c_api",
    hdrs = [
        "c_api.h",
    ],
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":c_api_without_op_resolver",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:create_op_resolver_with_builtin_ops",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)
# LINT.ThenChange(:private_c_api)

# LINT.IfChange(private_c_api)
cc_library(
    name = "private_c_api",
    hdrs = [
        "c_api.h",
    ],
    copts = tflite_copts(),
    tags = ["avoid_dep"],
    # This is a private target, its visibility is set to public only to be
    # used by "tflite_custom_c_library".
    # Do not use this target directly and don't consider it as a part of the public API.
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":c_api_without_op_resolver",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:create_op_resolver_with_builtin_ops",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
    ],
)
# LINT.ThenChange(:c_api)

tflite_cc_library_with_c_headers_test(
    name = "c_api_without_op_resolver",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:create_op_resolver_header",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core:headers",
        "//tensorflow/lite/delegates:interpreter_utils",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

tflite_cc_library_with_c_headers_test(
    name = "private_c_api_without_op_resolver",
    hdrs = ["c_api.h"],
    copts = tflite_copts(),
    tags = [
        "allow_undefined_symbols",  # For tflite::CreateOpResolver().
        "avoid_dep",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/c:c_api_types",
    ],
)

# Same as ":c_api_without_op_resolver", but without alwayslink=1.
tflite_cc_library_with_c_headers_test(
    name = "c_api_without_op_resolver_without_alwayslink",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:create_op_resolver_header",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core:headers",
        "//tensorflow/lite/delegates:interpreter_utils",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
        "//tensorflow/lite:testdata/custom_sinh.bin",
    ],
    visibility = [
        "//tensorflow/lite/c:__subpackages__",
    ],
    deps = [
        ":c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates:delegate_test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "selectively_built_c_api_test",
    size = "small",
    srcs = ["c_api_test.cc"],
    copts = tflite_copts(),
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
        "//tensorflow/lite:testdata/custom_sinh.bin",
    ],
    visibility = [
        "//tensorflow/lite/c:__subpackages__",
    ],
    deps = [
        ":c_api_without_op_resolver_without_alwayslink",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/c:selectively_built_c_api_test_lib",
        "//tensorflow/lite/delegates:delegate_test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)
