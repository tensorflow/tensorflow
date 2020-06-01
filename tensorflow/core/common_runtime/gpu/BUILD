load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cc_tests_gpu")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# -----------------------------------------------------------------------------
# Libraries with GPU facilities that are useful for writing kernels.
cc_library(
    name = "gpu_lib",
    srcs = [
        "gpu_event_mgr.cc",
    ],
    hdrs = [
        "gpu_event_mgr.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor",
    ],
)

cc_library(
    name = "gpu_headers_lib",
    hdrs = [
        "gpu_event_mgr.h",
    ],
)

cc_library(
    name = "cuda",
    deps = [
        "//tensorflow/core/platform/default/build_config:cuda",
    ],
)

cc_library(
    name = "rocm",
    deps = [
        "//tensorflow/core/platform/default/build_config:rocm",
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
    deps = [
        "//tensorflow/core:lib",
    ],
)

filegroup(
    name = "gpu_runtime_headers",
    srcs = [
        "gpu_bfc_allocator.h",
        "gpu_cudamalloc_allocator.h",
        "gpu_debug_allocator.h",
        "gpu_device.h",
        "gpu_event_mgr.h",
        "gpu_host_allocator.h",
        "gpu_id.h",
        "gpu_id_manager.h",
        "gpu_id_utils.h",
        "gpu_init.h",
        "gpu_managed_allocator.h",
        "gpu_mem_allocator.h",
        "gpu_process_state.h",
        "gpu_util.h",
        "//tensorflow/core/common_runtime:gpu_runtime_headers",
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
    ],
    deps = [
        ":gpu_bfc_allocator",
        ":gpu_id_impl",
        ":gpu_init_impl",
        ":gpu_lib",
        "//tensorflow/core:core_cpu_impl",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "gpu_runtime",
    hdrs = [":gpu_runtime_headers"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor",
        "//third_party/eigen3",
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
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_mem_allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:bfc_allocator",
    ],
)

tf_cuda_library(
    name = "gpu_mem_allocator",
    srcs = [
        "gpu_id.h",
    ],
    hdrs = [
        "gpu_host_allocator.h",
        "gpu_mem_allocator.h",
    ],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core/framework:allocator",
    ],
)

tf_cuda_library(
    name = "gpu_init",
    hdrs = [
        "gpu_init.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor",
    ] + if_static(
        [":gpu_init_impl"],
    ),
)

tf_cuda_library(
    name = "gpu_init_impl",
    srcs = [
        "gpu_init.cc",
    ],
    hdrs = [
        "gpu_init.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor",
    ],
    alwayslink = 1,
)

# -----------------------------------------------------------------------------
# Tests

tf_cc_test(
    name = "gpu_device_on_non_gpu_machine_test",
    size = "small",
    srcs = ["gpu_device_on_non_gpu_machine_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":gpu_headers_lib",
        ":gpu_id",
        ":gpu_runtime",
        "//tensorflow/core:test",
    ],
)

tf_cc_tests_gpu(
    name = "gpu_related_tests",
    size = "small",
    srcs = [
        "gpu_bfc_allocator_test.cc",
        "gpu_device_test.cc",
        "gpu_id_manager_test.cc",
        "pool_allocator_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
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
    ],
)

tf_cc_test_gpu(
    name = "gpu_event_mgr_test",
    srcs = ["gpu_event_mgr_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
    ],
)

tf_cuda_cc_test(
    name = "gpu_device_unified_memory_test",
    size = "small",
    srcs = [
        "gpu_device_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    # Runs test on a Guitar cluster that uses P100s to test unified memory
    # allocations.
    tags = tf_cuda_tests_tags() + [
        "guitar",
        "multi_gpu",
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
    ],
)

tf_cc_test_gpu(
    name = "gpu_allocator_retry_test",
    size = "medium",
    srcs = ["gpu_allocator_retry_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
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
    ],
)

tf_cc_test_gpu(
    name = "gpu_debug_allocator_test",
    size = "medium",
    srcs = ["gpu_debug_allocator_test.cc"],
    args = ["--gtest_death_test_style=threadsafe"],
    linkstatic = tf_kernel_tests_linkstatic(),
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
    ],
)
