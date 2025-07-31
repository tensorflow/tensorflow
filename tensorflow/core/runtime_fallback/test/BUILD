load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_test", "tf_cc_test")
# copybara:uncomment load("//third_party/tf_runtime_google/cpp_tests:gen_tests.bzl", "tfrt_cc_test_and_strict_benchmark")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//learning/brain/experimental/tfrt/cpp_tests/tpu_model/...",
        "//tensorflow/core/runtime_fallback/test/...",
        "//tensorflow/core/runtime_fallback/test/gpu/...",
    ],
)

cc_library(
    name = "forwarding_test_kernels",
    srcs = ["forwarding_test_kernels.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/runtime_fallback/kernel:tfrt_op_kernel",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core/platform:errors",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tfrt_forwarding_kernels_alwayslink",
    srcs = [
        "tfrt_forwarding_kernels.cc",
        "tfrt_forwarding_static_registration.cc",
    ],
    includes = [
        "third_party/tf_runtime/include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/framework:types_proto_cc",
        "@tf_runtime//:hostcontext",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:tensor",
        ],
    }),
    alwayslink = 1,
)

# CoreRuntime cpp test driver.
cc_library(
    name = "coreruntime_driver",
    srcs = ["coreruntime_driver.cc"],
    hdrs = ["coreruntime_driver.h"],
    includes = [
        "third_party/tf_runtime/include",
    ],
    deps = [
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/runtime_fallback/util:fallback_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
    ],
)

# copybara:uncomment_begin(internal benchmarking)
# # C++ tests and benchmarks for native op excuted via core runtime execution.
# tfrt_cc_test_and_strict_benchmark(
#     name = "coreruntime",
#     srcs = ["coreruntime_test.cc"],
#     enable_xprof = True,
#     includes = ["third_party/tf_runtime/include"],
#     owners = ["tf-runtime-testing"],
#     tags = [
#         "need_main",
#         "no_gpu",
#         "noguitar",  # b/190188191
#     ],
#     deps = [
#         ":coreruntime_driver",
#         "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
#         "@tf_runtime//:core_runtime_alwayslink",
#         "@tf_runtime//:tensor",
#         "@tf_runtime//backends/cpu:core_runtime_alwayslink",
#         "@tf_runtime//backends/cpu:test_ops_alwayslink",
#         "@tf_runtime//backends/cpu:tf_ops_alwayslink",
#     ],
# )
#
# # C++ tests and benchmarks for runtime fallback.
# tfrt_cc_test_and_strict_benchmark(
#     name = "fallback_coreruntime",
#     srcs = ["fallback_coreruntime_test.cc"],
#     enable_xprof = True,
#     includes = ["third_party/tf_runtime/include"],
#     owners = ["tf-runtime-testing"],
#     tags = [
#         "need_main",
#         "no_gpu",
#     ],
#     deps = [
#         ":coreruntime_driver",
#         # b/176116250: Use "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime" for Andriod build.
#         "//tensorflow/core:all_kernels",
#         "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
#         "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
#         "@tf_runtime//:core_runtime_alwayslink",
#         "@tf_runtime//:tensor",
#         "@tf_runtime//backends/cpu:core_runtime_alwayslink",
#         "@tf_runtime//backends/cpu:test_ops_alwayslink",
#     ],
# )
#
# # C++ tests and benchmarks for runtime fallback.
# tfrt_cc_test_and_strict_benchmark(
#     name = "c_api_tfrt",
#     srcs = ["c_api_tfrt_test.cc"],
#     enable_xprof = True,
#     includes = ["third_party/tf_runtime/include"],
#     owners = ["tf-runtime-testing"],
#     tags = [
#         "need_main",
#         "no_gpu",
#     ],
#     deps = [
#         # b/176116250: Use "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime" for Andriod build.
#         "//tensorflow/c:c_test_util",
#         "//tensorflow/c/eager:c_api",
#         "//tensorflow/c/eager:c_api_experimental",
#         "//tensorflow/c/eager:c_api_internal",
#         "//tensorflow/c/eager:c_api_test_util",
#         "//tensorflow/c/eager:tfe_op_internal",
#         "//tensorflow/c/eager:tfe_tensorhandle_internal",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:lib_internal",
#         "//tensorflow/core:protos_all_cc",
#         "//tensorflow/core:test",
#         "//tensorflow/core/common_runtime/eager:eager_operation",
#         "//tensorflow/core/common_runtime/eager:tensor_handle",
#         "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
#     ],
# )
#
# tf_cc_shared_test(
#     name = "runtime_fallback_kernels_test",
#     srcs = ["runtime_fallback_kernels_test.cc"],
#     deps = [
#         ":coreruntime_driver",
#         "@com_google_googletest//:gtest",
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
#         "@tf_runtime//:core_runtime",
#         "@tf_runtime//backends/cpu:core_runtime_alwayslink",
#     ] + select({
#         "//tensorflow:android": [
#             "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
#         ],
#         "//conditions:default": [
#             "//tensorflow/c/eager:abstract_tensor_handle",
#             "//tensorflow/core:all_kernels",
#             "//tensorflow/core:test",
#             "//tensorflow/core/framework:tensor_testutil",
#             "//tensorflow/core/framework:types_proto_cc",
#         ],
#     }),
# )
#
# # C++ tests and benchmarks for kernel fallback.
# tf_cc_test(
#     name = "kernel_fallback_coreruntime_test",
#     srcs = ["kernel_fallback_coreruntime_test.cc"],
#     includes = ["third_party/tf_runtime/include"],
#     deps = [
#         ":coreruntime_driver",
#         "@com_google_googletest//:gtest",
#         "//tensorflow/core/platform:test_benchmark",
#         "//tensorflow/core/platform/default/build_config:test_main",
#         "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
#         "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
#         "@tf_runtime//:core_runtime_alwayslink",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:tensor",
#         "@tf_runtime//backends/cpu:core_runtime_alwayslink",
#         "@tf_runtime//backends/cpu:test_ops_alwayslink",
#     ] + select({
#         "//tensorflow:android": [
#             "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
#         ],
#         "//conditions:default": [
#             "//tensorflow/core:all_kernels",
#         ],
#     }),
# )
#
# tf_cc_test(
#     name = "kernel_fallback_compat_request_state_test",
#     srcs = ["kernel_fallback_compat_request_state_test.cc"],
#     includes = ["third_party/tf_runtime/include"],
#     deps = [
#         "@com_google_googletest//:gtest",
#         "//tensorflow/core/framework:tensor_testutil",
#         "//tensorflow/core/platform/default/build_config:test_main",
#         "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
#         "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
#         "@tf_runtime//:core_runtime_alwayslink",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "test_tf_opkernels_alwayslink",
    testonly = True,
    srcs = [
        "test_opkernels.cc",
    ],
    visibility = ["//tensorflow/core/runtime_fallback:__subpackages__"],
    deps = [
        "@com_google_absl//absl/log",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "test_tfrt_kernels_alwayslink",
    testonly = True,
    srcs = [
        "static_registration.cc",
        "test_kernels.cc",
    ],
    visibility = ["//tensorflow/core/runtime_fallback:__subpackages__"],
    deps = [
        "//tensorflow/core:lib",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:tensor",
        ],
    }),
    alwayslink = 1,
)
