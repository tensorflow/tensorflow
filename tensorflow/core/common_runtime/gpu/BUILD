load("@bazel_skylib//lib:selects.bzl", "selects")
load(
    "//tensorflow:tensorflow.bzl",
    "clean_dep",
    "if_cuda_or_rocm",
    "if_google",
    "if_linux_x86_64",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "filegroup",
    "tf_cuda_cc_test",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    features = if_google(
        [
            "-parse_headers",
        ],
    ),
    licenses = ["notice"],
)

# -----------------------------------------------------------------------------
# Libraries with GPU facilities that are useful for writing kernels.

cc_library(
    name = "gpu_lib",
    hdrs = [
        "gpu_event_mgr.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/common_runtime/device:device_event_mgr",
    ],
)

cc_library(
    name = "gpu_headers_lib",
    textual_hdrs = [
        "gpu_event_mgr.h",
    ],
    deps = [
        "//tensorflow/core/common_runtime/device:device_event_mgr_hdrs",
    ],
)

cc_library(
    name = "rocm",
    deps = [
        "@local_xla//xla/stream_executor/rocm:rocm_rpath",
    ],
)

cc_library(
    name = "gpu_id",
    hdrs = [
        "gpu_id.h",
        "gpu_id_manager.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/device:device_id",
    ] + if_static([
        ":gpu_id_impl",
    ]),
)

cc_library(
    name = "gpu_id_impl",
    srcs = ["gpu_id_manager.cc"],
    hdrs = [
        "gpu_id.h",
        "gpu_id_manager.h",
    ],
    features = ["-layering_check"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/device:device_id_impl",
    ],
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = [
        "gpu_event_mgr.h",
        "gpu_managed_allocator.h",
    ],
    visibility = [
        "//tensorflow/core/kernels:__pkg__",
        "//tensorflow/core/kernels/image:__pkg__",
        "//tensorflow/core/kernels/sparse:__pkg__",
    ],
)

filegroup(
    name = "gpu_runtime_headers",
    srcs = [
        "gpu_bfc_allocator.h",
        "gpu_cudamalloc_allocator.h",
        "gpu_debug_allocator.h",
        "gpu_device.h",
        "gpu_id.h",
        "gpu_id_manager.h",
        "gpu_managed_allocator.h",
        "gpu_process_state.h",
        "gpu_util.h",
        "//tensorflow/core/common_runtime:gpu_runtime_headers",
        "//tensorflow/core/common_runtime/device:device_runtime_headers",
        "@local_tsl//tsl/framework:bfc_allocator.h",
    ],
    visibility = ["//visibility:private"],
)

tf_cuda_library(
    name = "gpu_runtime_impl",
    srcs = [
        "gpu_cudamalloc_allocator.cc",
        "gpu_debug_allocator.cc",
        "gpu_device.cc",
        "gpu_device_factory.cc",
        "gpu_managed_allocator.cc",
        "gpu_process_state.cc",
        "gpu_util.cc",
        "gpu_util_platform_specific.cc",
    ],
    hdrs = [":gpu_runtime_headers"],
    copts = tf_copts(),
    cuda_deps = [
        "@local_config_cuda//cuda:cudnn_header",
        "@local_xla//xla/stream_executor/cuda:cuda_platform",
        "@local_xla//xla/stream_executor/gpu:gpu_stream",
    ],
    defines = if_linux_x86_64(["TF_PLATFORM_LINUX_X86_64"]),
    features = ["-layering_check"],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    deps = [
        ":gpu_bfc_allocator",
        ":gpu_id_impl",
        ":gpu_lib",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_impl",
        "//tensorflow/core/common_runtime:device_id_utils",
        "//tensorflow/core/common_runtime:node_file_writer",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/platform:tensor_float_32_utils",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/framework:device_id_utils",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
        "@local_xla//xla/stream_executor/gpu:gpu_init_impl",
    ] + if_google(
        # TODO(b/282068262): PJRT pulls in TFRT components that are incompatible with ARM platform.
        # Clean up so that PJRT can run on ARM (and remove "#if defined(PLATFORM_GOOGLE) ..." use
        # from gpu_util.cc).
        # Also it won't build with WeightWatcher which tracks OSS build binaries.
        # TODO(b/290533709): Clean up this build rule.
        selects.with_or({
            clean_dep("//tensorflow:linux_x86_64_with_weightwatcher"): [],
            (
                clean_dep("//tensorflow:linux_x86_64"),
                clean_dep("//tensorflow:haswell"),
            ): [
                "//tensorflow/compiler/tf2xla:layout_util",
                "//tensorflow/compiler/jit:flags",
                "//tensorflow/compiler/jit:pjrt_device_context",
                "@local_xla//xla/pjrt/gpu:gpu_helpers",
                "@local_xla//xla/pjrt/gpu:se_gpu_pjrt_client",
                "@local_xla//xla/stream_executor/integrations:tf_allocator_adapter",
                "//tensorflow/core/tfrt/common:pjrt_util",
            ],
            "//conditions:default": [],
        }) + if_cuda_or_rocm([
            "@local_xla//xla/service:gpu_plugin_impl",  # for registering cuda compiler.
        ]),
    ),
    alwayslink = 1,
)

tf_cuda_library(
    name = "gpu_runtime",
    hdrs = [":gpu_runtime_headers"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor/gpu:gpu_init",
    ] + if_static([":gpu_runtime_impl"]),
)

# This is redundant with the "gpu_runtime_*" targets above. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).
tf_cuda_library(
    name = "gpu_bfc_allocator",
    srcs = [
        "gpu_bfc_allocator.cc",
    ],
    hdrs = ["gpu_bfc_allocator.h"],
    features = [
        "-layering_check",
        "parse_headers",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:bfc_allocator",
        "//tensorflow/core/common_runtime/device:device_mem_allocator",
    ],
)

# -----------------------------------------------------------------------------
# Tests

tf_cc_test(
    name = "gpu_device_on_non_gpu_machine_test",
    size = "small",
    srcs = ["gpu_device_on_non_gpu_machine_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":gpu_headers_lib",
        ":gpu_id",
        ":gpu_runtime",
        "//tensorflow/core:test",
    ],
)

tf_cuda_cc_test(
    name = "gpu_bfc_allocator_test",
    size = "small",
    srcs = [
        "gpu_bfc_allocator_test.cc",
    ],
    features = ["-layering_check"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
        "@local_xla//xla/stream_executor/integrations:device_mem_allocator",
    ],
)

tf_cuda_cc_test(
    name = "gpu_device_test",
    size = "small",
    srcs = [
        "gpu_device_test.cc",
    ],
    features = ["-layering_check"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_id",
        ":gpu_runtime",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
        "@local_tsl//tsl/framework:device_id",
        "@local_xla//xla/stream_executor/gpu:gpu_cudamallocasync_allocator_header",
        "@local_xla//xla/tests:test_macros_header",
    ],
)

tf_cuda_cc_test(
    name = "pool_allocator_test",
    size = "small",
    srcs = [
        "pool_allocator_test.cc",
    ],
    features = ["-layering_check"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/stream_executor/gpu:gpu_init",
    ],
)

tf_cuda_cc_test(
    name = "gpu_device_unified_memory_test",
    size = "small",
    srcs = [
        "gpu_device_test.cc",
    ],
    features = ["-layering_check"],
    # Runs test on a Guitar cluster that uses P100s to test unified memory
    # allocations.
    tags = tf_cuda_tests_tags() + [
        "guitar",
        # "multi_gpu", # TODO(b/287692888): re-enable once the 2gpu test passes.
    ],
    deps = [
        ":gpu_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
        "@local_xla//xla/stream_executor/gpu:gpu_cudamallocasync_allocator_header",
        "@local_xla//xla/tests:test_macros_header",
    ],
)

tf_cuda_cc_test(
    name = "gpu_allocator_retry_test",
    size = "medium",
    srcs = ["gpu_allocator_retry_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:direct_session_internal",
    ],
)

tf_cuda_cc_test(
    name = "gpu_debug_allocator_test",
    size = "medium",
    srcs = ["gpu_debug_allocator_test.cc"],
    args = ["--gtest_death_test_style=threadsafe"],
    features = ["-layering_check"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor/integrations:device_mem_allocator",
    ],
)

cc_library(
    name = "gpu_serving_device_selector",
    srcs = ["gpu_serving_device_selector.cc"],
    hdrs = ["gpu_serving_device_selector.h"],
    features = ["-layering_check"],
    deps = [
        ":gpu_scheduling_metrics_storage",
        "//tensorflow/core/framework:resource_base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/framework:serving_device_selector",
    ],
)

tf_cc_test(
    name = "gpu_serving_device_selector_test",
    size = "small",
    srcs = ["gpu_serving_device_selector_test.cc"],
    deps = [
        ":gpu_scheduling_metrics_storage",
        ":gpu_serving_device_selector",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/framework:serving_device_selector",
        "@local_tsl//tsl/framework:serving_device_selector_policies",
    ],
)

cc_library(
    name = "gpu_scheduling_metrics_storage",
    srcs = ["gpu_scheduling_metrics_storage.cc"],
    hdrs = ["gpu_scheduling_metrics_storage.h"],
    deps = ["@local_tsl//tsl/framework:real_time_in_memory_metric"],
)
