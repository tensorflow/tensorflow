load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "if_static",
)
load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
load(
    "//xla/stream_executor:build_defs.bzl",
    "cuda_only_cc_library",
    "stream_executor_friends",
    "tf_additional_cuda_platform_deps",
    "tf_additional_cudnn_plugin_copts",
    "tf_additional_gpu_compilation_copts",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "if_nccl", "internal_visibility", "tsl_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

bool_flag(
    name = "enable_libnvptxcompiler_support",
    build_setting_default = if_google(
        True,
        oss_value = False,
    ),
)

config_setting(
    name = "libnvptxcompiler_support_enabled",
    flag_values = {
        ":enable_libnvptxcompiler_support": "True",
    },
)

cc_library(
    name = "cuda_platform_id",
    srcs = ["cuda_platform_id.cc"],
    hdrs = ["cuda_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cuda_only_cc_library(
    name = "cuda_platform",
    srcs = ["cuda_platform.cc"],
    hdrs = ["cuda_platform.h"],
    visibility = ["//visibility:public"],
    deps =
        [
            ":cuda_activation",
            ":cuda_collectives",
            ":cuda_driver",
            ":cuda_executor",
            ":cuda_platform_id",
            ":cuda_runtime",
            "//xla/stream_executor",
            "//xla/stream_executor:platform_manager",
            "//xla/stream_executor:stream_executor_interface",
            "//xla/stream_executor/gpu:gpu_driver_header",
            "//xla/stream_executor/gpu:gpu_executor_header",
            "//xla/stream_executor/platform",
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/log",
            "@com_google_absl//absl/log:check",
            "@com_google_absl//absl/memory",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/synchronization",
            "@local_tsl//tsl/platform:errors",
            "@local_tsl//tsl/platform:status",
            "@local_tsl//tsl/platform:statusor",
        ] + tf_additional_cuda_platform_deps(),
    alwayslink = True,  # Registers itself with the PlatformManager.
)

cuda_only_cc_library(
    name = "cuda_diagnostics",
    srcs = ["cuda_diagnostics.cc"],
    hdrs = ["cuda_diagnostics.h"],
    deps = [
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

# Buildozer can not remove dependencies inside select guards, so we have to use
# an intermediate target.
cc_library(name = "ptxas_wrapper")

cc_library(name = "nvlink_wrapper")

cuda_only_cc_library(
    name = "cuda_driver",
    srcs = ["cuda_driver.cc"],
    hdrs = ["cuda_driver.h"],
    deps = [
        ":cuda_diagnostics",  # buildcleaner: keep
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/tsl/cuda",
        "//xla/tsl/cuda:cudart",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:stacktrace",
    ],
)

cuda_only_cc_library(
    name = "cuda_runtime",
    srcs = ["cuda_runtime.cc"],
    deps = [
        "//xla/stream_executor/gpu:gpu_runtime_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cuda_only_cc_library(
    name = "cuda_collectives",
    srcs = ["cuda_collectives.cc"],
    defines = if_nccl(["STREAM_EXECUTOR_GPU_ENABLE_XCCL"]),
    deps = [
        ":cuda_driver",
        "//xla/stream_executor/gpu:gpu_collectives_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:statusor",
    ] + if_nccl(["@local_config_nccl//:nccl"]),
)

xla_test(
    name = "cuda_driver_test",
    srcs = ["cuda_driver_test.cc"],
    backends = ["gpu"],
    tags = [
        # TODO(b/317293391) Remove once Bazel test_suite handles tags correctly
        "gpu",
        "no_rocm",
    ],
    deps = [
        ":cuda_driver",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "@com_google_absl//absl/log",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

# The activation library is tightly coupled to the executor library.
# TODO(leary) split up cuda_executor.cc so that this can stand alone.
cc_library(
    name = "cuda_activation_header",
    hdrs = ["cuda_activation.h"],
    visibility = ["//visibility:public"],
    deps = ["//xla/stream_executor/gpu:gpu_activation_header"],
)

cuda_only_cc_library(
    name = "cuda_activation",
    srcs = [],
    hdrs = ["cuda_activation.h"],
    deps = [
        ":cuda_driver",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "//xla/stream_executor/gpu:gpu_activation",
        "//xla/stream_executor/platform",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_only_cc_library(
    name = "cublas_lt_header",
    hdrs = [
        "cuda_blas_lt.h",
        "cuda_blas_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:types",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
    ],
)

cuda_only_cc_library(
    name = "cublas_plugin",
    srcs = [
        "cuda_blas.cc",
        "cuda_blas_lt.cc",
    ],
    hdrs = [
        "cuda_blas.h",
        "cuda_blas_lt.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_activation",
        ":cuda_blas_utils",
        ":cuda_executor",
        ":cuda_helpers",
        ":cuda_platform_id",
        ":cuda_stream",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor/gpu:gpu_activation_header",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_timer",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/platform",
        "//xla/tsl/cuda:cublas",
        "//xla/tsl/cuda:cublas_lt",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@eigen_archive//:eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@local_tsl//tsl/protobuf:dnn_proto_cc",
    ] + if_static([
        "@local_tsl//tsl/platform:tensor_float_32_utils",
    ]),
    alwayslink = True,
)

cuda_only_cc_library(
    name = "cuda_blas_utils",
    srcs = ["cuda_blas_utils.cc"],
    hdrs = ["cuda_blas_utils.h"],
    deps = [
        "//xla/stream_executor",
        "//xla/tsl/cuda:cublas",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
    ],
)

cuda_only_cc_library(
    name = "cufft_plugin",
    srcs = ["cuda_fft.cc"],
    hdrs = ["cuda_fft.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_activation_header",
        ":cuda_platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_interface",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/platform",
        "//xla/tsl/cuda:cufft",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "cuda_dnn_headers",
    textual_hdrs = ["cuda_dnn.h"],
    deps = if_cuda_is_configured([
        ":cuda_activation_header",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:plugin_registry",
    ]) + [
        "//xla/stream_executor",  # build_cleaner: keep
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/protobuf:dnn_proto_cc",
    ],
)

cuda_only_cc_library(
    name = "cudnn_plugin",
    srcs = ["cuda_dnn.cc"],
    hdrs = ["cuda_dnn.h"],
    copts = tf_additional_cudnn_plugin_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_activation",
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_executor",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cudnn_frontend_helpers",
        "//xla/stream_executor",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_interface",
        "//xla/stream_executor/gpu:gpu_activation_header",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:gpu_timer",
        "//xla/stream_executor/platform",
        "//xla/tsl/cuda:cudnn",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@cudnn_frontend_archive//:cudnn_frontend",
        "@eigen_archive//:eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@local_tsl//tsl/platform:tensor_float_32_utils",
        "@local_tsl//tsl/protobuf:dnn_proto_cc",
    ],
    alwayslink = True,
)

cuda_only_cc_library(
    name = "cuda_kernel",
    srcs = ["cuda_kernel.cc"],
    hdrs = ["cuda_kernel.h"],
    deps = [
        ":cuda_driver",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "cuda_conditional_kernels",
    srcs = ["cuda_conditional_kernels.cc"],
)

# TODO(leary) we likely need to canonicalize/eliminate this.
cc_library(
    name = "cuda_helpers",
    textual_hdrs = if_cuda_is_configured(["cuda_helpers.h"]),
    deps = if_cuda_is_configured([
        "//xla/stream_executor/gpu:gpu_helpers_header",
    ]),
)

cuda_only_cc_library(
    name = "cuda_event",
    srcs = ["cuda_event.cc"],
    hdrs = ["cuda_event.h"],
    deps = [
        ":cuda_driver",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_event",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_only_cc_library(
    name = "cuda_stream",
    srcs = [],
    hdrs = ["cuda_stream.h"],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_stream",
    ],
)

cc_library(
    name = "ptx_compiler_support",
    srcs = ["ptx_compiler_support.cc"],
    hdrs = ["ptx_compiler_support.h"],
    local_defines = select({
        ":libnvptxcompiler_support_enabled": [
            "LIBNVPTXCOMPILER_SUPPORT=true",
        ],
        "//conditions:default": [
            "LIBNVPTXCOMPILER_SUPPORT=false",
        ],
    }),
)

cc_library(
    name = "ptx_compiler_stub",
    srcs = [
        "ptx_compiler_stub.cc",
    ],
    deps = [
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ptx_compiler_impl",
    srcs = [
        "ptx_compiler_impl.cc",
    ],
    tags = ["manual"],
    deps = [
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:nvptxcompiler",
    ],
)

cc_library(
    name = "ptx_compiler",
    hdrs = ["ptx_compiler.h"],
    deps = select({
        ":libnvptxcompiler_support_enabled": [":ptx_compiler_impl"],
        "//conditions:default": [":ptx_compiler_stub"],
    }) + [
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "ptx_compiler_test",
    srcs = ["ptx_compiler_test.cc"],
    deps = [
        ":ptx_compiler",
        ":ptx_compiler_support",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cuda_only_cc_library(
    name = "cuda_asm_compiler",
    srcs = ["cuda_asm_compiler.cc"],
    hdrs = ["cuda_asm_compiler.h"],
    copts = tf_additional_gpu_compilation_copts(),
    # copybara:uncomment_begin
    # data = [
    # "@local_config_cuda//cuda:runtime_fatbinary",
    # "@local_config_cuda//cuda:runtime_nvlink",
    # "@local_config_cuda//cuda:runtime_ptxas",
    # ],
    # copybara:uncomment_end
    visibility = internal_visibility([
        "//third_party/py/jax:__subpackages__",
        "//tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
    ]),
    deps = [
        ":cuda_driver",
        ":ptx_compiler",
        ":ptx_compiler_support",
        "//xla:status_macros",
        "//xla:util",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:cuda_libdevice_path",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:subprocess",
    ],
)

cuda_only_cc_library(
    name = "cuda_executor",
    srcs = ["cuda_executor.cc"],
    deps = [
        ":cuda_collectives",  # buildcleaner: keep
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_event",  # buildcleaner: keep
        ":cuda_kernel",  # buildcleaner: keep
        ":cuda_platform_id",
        ":cuda_runtime",  # buildcleaner: keep
        "//xla/stream_executor",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_interface",
        "//xla/stream_executor/gpu:gpu_collectives_header",
        "//xla/stream_executor/gpu:gpu_command_buffer",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_event_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/gpu:gpu_runtime_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_timer",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/integrations:device_mem_allocator",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cublas_plugin",
        ":cuda_driver",
        ":cuda_platform",
        ":cudnn_plugin",
        ":cufft_plugin",
        "//xla/tsl/cuda:cusolver",
        "//xla/tsl/cuda:cusparse",
        "//xla/tsl/cuda:tensorrt_rpath",
    ],
    alwayslink = 1,
)

# OSX framework for device driver access
cc_library(
    name = "IOKit",
    linkopts = ["-framework IOKit"],
)

cc_library(
    name = "stream_executor_cuda",
    deps = [
        "//xla/stream_executor:stream_executor_bundle",
    ] + if_google(
        select({
            # copybara:uncomment_begin(different config setting in OSS)
            # "//tools/cc_target_os:gce": [],
            # copybara:uncomment_end_and_comment_begin
            "//conditions:default": [
                "@local_config_cuda//cuda:cudart_static",  # buildcleaner: keep
                ":cuda_platform",
            ],
        }),
        [
            "//xla/tsl/cuda:cudart",
        ] + select({
            "//xla/tsl:macos": ["IOKit"],
            "//conditions:default": [],
        }),
    ),
)

cc_library(
    name = "cudnn_frontend_helpers",
    srcs = ["cudnn_frontend_helpers.cc"],
    hdrs = ["cudnn_frontend_helpers.h"],
)

cc_library(
    name = "cuda_nvptxcompiler",
)
