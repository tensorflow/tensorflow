load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_cc_library_with_c_headers_test",
    "tflite_cc_shared_object",
    "tflite_copts",
    "tflite_copts_warnings",
    "tflite_custom_c_library",
    "tflite_linkopts_no_undefined",
    "tflite_self_contained_libs_test_suite",
)
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite_with_c_headers_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Generates a platform-specific shared library containing the TensorFlow Lite C
# API implementation as define in `c_api.h`. The exact output library name
# is platform dependent:
#   - Linux/Android: `libtensorflowlite_c.so`
#   - Mac: `libtensorflowlite_c.dylib`
#   - Windows: `tensorflowlite_c.dll`
tflite_cc_shared_object(
    name = "tensorflowlite_c",
    linkopts = tflite_linkopts_no_undefined() + select({
        "//tensorflow:ios": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,--version-script,$(location //tensorflow/lite/c:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":exported_symbols.lds",
        ":version_script.lds",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api",
    hdrs = [
        "c_api.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    generate_opaque_delegate_target = True,
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_without_op_resolver",
        "//tensorflow/lite/core/async/c:types",
        "//tensorflow/lite/core/c:c_api",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

tflite_cc_library_with_c_headers_test(
    name = "c_api_for_testing",
    testonly = True,
    srcs = ["c_api_for_testing.cc"],
    hdrs = ["c_api_for_testing.h"],
    deps = [
        ":c_api",
        ":c_api_internal",
        "//tensorflow/lite/core/c:c_api",
    ],
)

# TODO(b/248514738): Deprecate this target in favour
# of //third_party/tensorflow/lite/core/c:c_api_without_op_resolver after all known existing uses of
# those targets have been changed to instead refer to the new target.
# Same as ":c_api", but doesn't link in the default CreateOpResolver implementation.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api_without_op_resolver",
    hdrs = ["c_api.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_types",
        "//tensorflow/lite/core/c:c_api_without_op_resolver",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

# TODO(b/248514738): Deprecate this target in favour
# of //third_party/tensorflow/lite/core/c:c_api_without_op_resolver_without_alwayslink after all
# known existing uses of those targets have been changed to instead refer to the new target.
# Same as ":c_api_without_op_resolver", but without alwayslink=1.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api_without_op_resolver_without_alwayslink",
    hdrs = ["c_api.h"],
    copts = tflite_copts(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = ["//tensorflow/lite/core/c:c_api_without_op_resolver_without_alwayslink"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api_experimental",
    hdrs = [
        "c_api_experimental.h",
        "c_api_opaque.h",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        ":c_api",
    ],
    deps = [
        ":c_api_opaque_internal",
        ":common_internal",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:c_api",
        "//tensorflow/lite/core/c:c_api_experimental",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

# Same as ":c_api_experimental", but without linking in the default CreateOpResolver implementation.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api_experimental_without_op_resolver",
    hdrs = [
        "c_api_experimental.h",
        "c_api_opaque.h",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_opaque_internal",
        ":c_api_without_op_resolver",
        ":common_internal",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:c_api_experimental_without_op_resolver",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:c_api_without_op_resolver",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
    ],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)

# Same as ":c_api_experimental", but without linking in the default CreateOpResolver implementation.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api_experimental_without_op_resolver_without_alwayslink",
    hdrs = [
        "c_api_experimental.h",
        "c_api_opaque.h",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    deps = [
        ":c_api_opaque_internal_without_alwayslink",
        ":c_api_without_op_resolver_without_alwayslink",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:c_api_experimental_without_op_resolver_without_alwayslink",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:c_api_without_op_resolver_without_alwayslink",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    deps = ["//tensorflow/lite/core/c:c_api_types"],
)

cc_library(
    name = "c_api_internal",
    hdrs = ["c_api_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/lite/core/async/c:__subpackages__",
        "//tensorflow/lite/core/c:__subpackages__",
        "//tensorflow/lite/delegates/utils:__pkg__",
    ],
    deps = [
        ":common_internal",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/profiling/telemetry/c:profiler",
    ],
)

filegroup(
    name = "c_api_test_builtin_op_models",
    testonly = 1,
    srcs = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
    ],
)

# TODO(b/248514738): Deprecate this target in favour
# of //third_party/tensorflow/lite/core/c:c_api_test after all known existing uses of those targets
# have been changed to instead refer to the new target.
test_suite(
    name = "c_api_test",
    tests = [
        "//tensorflow/lite/core/c:c_api_test",
    ],
)

tflite_custom_c_library(
    name = "selectively_built_c_api_test_lib",
    testonly = 1,
    experimental = True,
    models = [":c_api_test_builtin_op_models"],
    visibility = [
        "//tensorflow/lite/core/c:__subpackages__",
    ],
)

# TODO(b/248514738): Deprecate this target in favour
# of //third_party/tensorflow/lite/core/c:selectively_built_c_api_test after all known existing uses
# of those targets have been changed to instead refer to the new target.
test_suite(
    name = "selectively_built_c_api_test",
    tests = [
        "//tensorflow/lite/core/c:selectively_built_c_api_test",
    ],
)

test_suite(
    name = "c_api_experimental_test",
    tests = [
        "//tensorflow/lite/core/c:c_api_experimental_test",
    ],
)

cc_test(
    name = "c_api_signature_runner_test",
    size = "small",
    srcs = ["c_api_signature_runner_test.cc"],
    copts = tflite_copts(),
    data = ["//tensorflow/lite:testdata/multi_signatures.bin"],
    deps = [
        ":c_api",
        "//tensorflow/lite/core/c:c_api",
        "//tensorflow/lite/core/c:c_api_experimental",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "common",
    hdrs = [
        "builtin_op_data.h",
        "common.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        ":c_api_types",
    ],
    deps = [
        "//tensorflow/lite:tflite_kernel_use_xnnpack_optional",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ] + select({
        "//tensorflow/lite:tensorflow_profiler_config": [
            "//tensorflow/lite:macros",
            "//tensorflow/lite:tensorflow_profiler_logger_shim",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "common_internal",
    srcs = ["common_internal.cc"],
    hdrs = ["common_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "c_api_opaque_internal",
    srcs = ["c_api_opaque_internal.cc"],
    hdrs = ["c_api_opaque_internal.h"],
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":c_api_types",
        ":c_api_without_op_resolver",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/core/c:common",
    ],
)

# Same as c_api_opaque_internal, but depends on the
# '_without_alwayslink' variant of ':c_api_without_op_resolver'.
cc_library(
    name = "c_api_opaque_internal_without_alwayslink",
    srcs = ["c_api_opaque_internal.cc"],
    hdrs = ["c_api_opaque_internal.h"],
    tags = ["allow_undefined_symbols"],  # For tflite::CreateOpResolver().
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":c_api_types",
        ":c_api_without_op_resolver_without_alwayslink",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/core/c:common",
    ],
)

# For use with library targets that can't use relative paths.
# LINT.IfChange(exported_headers)
exports_files([
    "c_api.h",
    "c_api_experimental.h",
    "c_api_opaque.h",
    "c_api_types.h",
    "common.h",
])

filegroup(
    name = "tensorflowlite_c_api_hdrs_filegroup",
    srcs = [
        "c_api.h",
        "c_api_types.h",
        "common.h",
    ],
)

alias(
    name = "tensorflowlite_c_impl_hdrs_filegroup",
    actual = "//tensorflow/lite/core/c:headers_filegroup",
)
# LINT.ThenChange(../java/BUILD:TFLITE_HEADERS)

# Test the C extension API code.
test_suite(
    name = "common_test",
    tests = [
        "//tensorflow/lite/core/c:common_test",
    ],
)

test_suite(
    name = "builtin_op_data_test",
    tests = [
        "//tensorflow/lite/core/c:builtin_op_data_test",
    ],
)

cc_test(
    name = "c_test",
    size = "small",
    srcs = ["c_test.c"],
    copts = tflite_copts(),
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/multi_signatures.bin",
    ],
    tags = [
        # Testing on Android uses "--gunit_output=xml:test.xml" runtime flag,
        # but this test is C code, not C++, and hence doesn't use GUnit,
        # so doesn't support that flag.
        "tflite_not_portable_android",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        "//tensorflow/lite/core/c:c_api",
        "//tensorflow/lite/core/c:c_api_experimental",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library_with_tflite_with_c_headers_test(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")

tflite_portable_test_suite()
