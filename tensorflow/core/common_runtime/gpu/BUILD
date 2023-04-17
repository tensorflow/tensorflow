load(
    "//tensorflow:tensorflow.bzl",
    "if_google",
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
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    features = if_google(
        [
            "-layering_check",
            "-parse_headers",
        ],
        ["-layering_check"],
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
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_rpath",
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
        "gpu_virtual_mem_allocator.h",
        "//tensorflow/core/common_runtime:gpu_runtime_headers",
        "//tensorflow/core/common_runtime/device:device_runtime_headers",
        "//tensorflow/tsl/framework:bfc_allocator.h",
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
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
        ":gpu_virtual_mem_allocator",
    ],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    deps = [
        ":gpu_bfc_allocator",
        ":gpu_id_impl",
        ":gpu_lib",
        "//tensorflow/compiler/xla/stream_executor:device_id_utils",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_init_impl",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_impl",
        "//tensorflow/core/common_runtime:node_file_writer",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/platform:tensor_float_32_utils",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "gpu_runtime",
    hdrs = [":gpu_runtime_headers"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor:device_id_utils",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_init",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
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
        ":gpu_virtual_mem_allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:bfc_allocator",
        "//tensorflow/core/common_runtime/device:device_mem_allocator",
    ],
)

tf_cuda_library(
    name = "gpu_virtual_mem_allocator",
    srcs = [
        "gpu_virtual_mem_allocator.cc",
    ],
    hdrs = [
        "gpu_virtual_mem_allocator.h",
    ],
    copts = tf_copts(),
    cuda_deps = [
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_driver_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
    ],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_id",
        "//tensorflow/compiler/xla/stream_executor:platform",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# -----------------------------------------------------------------------------
# Tests

tf_cc_test(
    name = "gpu_device_on_non_gpu_machine_test",
    size = "small",
    srcs = ["gpu_device_on_non_gpu_machine_test.cc"],
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
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/compiler/xla/stream_executor:device_mem_allocator",
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

tf_cuda_cc_test(
    name = "gpu_device_test",
    size = "small",
    srcs = [
        "gpu_device_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_id",
        ":gpu_runtime",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_cudamallocasync_allocator_header",
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
        "//tensorflow/core/platform:test",
    ],
)

tf_cuda_cc_test(
    name = "pool_allocator_test",
    size = "small",
    srcs = [
        "pool_allocator_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_init",
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

tf_cuda_cc_test(
    name = "gpu_device_unified_memory_test",
    size = "small",
    srcs = [
        "gpu_device_test.cc",
    ],
    # Runs test on a Guitar cluster that uses P100s to test unified memory
    # allocations.
    tags = tf_cuda_tests_tags() + [
        "guitar",
        "multi_gpu",
        "no_rocm" # fail on CI
    ],
    deps = [
        ":gpu_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_cudamallocasync_allocator_header",
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
        "//tensorflow/core/platform:test",
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
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/compiler/xla/stream_executor:device_mem_allocator",
        "//tensorflow/compiler/xla/stream_executor:platform",
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

tf_cc_test(
    name = "gpu_virtual_mem_allocator_test",
    size = "small",
    srcs = ["gpu_virtual_mem_allocator_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_virtual_mem_allocator",
        "//tensorflow/compiler/xla/stream_executor:device_mem_allocator",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_init",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/platform:stream_executor",
    ],
)
