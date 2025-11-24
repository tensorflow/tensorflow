load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:py_strict.bzl", "py_strict_binary")
load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "if_oss",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

cc_library(
    name = "llvm_gpu_backend",
    srcs = ["gpu_backend_lib.cc"],
    hdrs = ["gpu_backend_lib.h"],
    deps = [
        ":load_ir_module",
        ":utils",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/codegen:intrinsic_lib",
        "//xla/codegen/intrinsic",
        "//xla/codegen/intrinsic:intrinsic_compiler_lib",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

cc_library(
    name = "ptx_version_util",
    srcs = ["ptx_version_util.cc"],
    hdrs = ["ptx_version_util.h"],
    deps = [
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "nvptx_backend",
    srcs = ["nvptx_backend.cc"],
    hdrs = ["nvptx_backend.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":llvm_gpu_backend",
        ":load_ir_module",
        ":nvptx_libdevice_path",
        ":ptx_version_util",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/service/gpu:metrics",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:subprocess_compilation",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:NVPTXCodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

py_strict_binary(
    name = "generate_amdgpu_device_lib_data_tool",
    srcs = ["generate_amdgpu_device_lib_data_tool.py"],
)

genrule(
    name = "generate_amdgpu_device_lib_data",
    srcs = [
        "@rocm_device_libs//:ockl",
        "@rocm_device_libs//:ocml",
    ],
    outs = ["amdgpu_device_lib_data.h"],
    cmd = "$(location {}) --llvm_link_bin $(location {}) $(SRCS) -o $@ --cpp_identifier=kAMDGPUDeviceLibData".format(
        ":generate_amdgpu_device_lib_data_tool",
        "@llvm-project//llvm:llvm-link",
    ),
    tags = if_google([
        # Embedding libdevice is not supported in the Google-internal build.
        "manual",
        "notap",
        "nobuilder",
    ]),
    tools = [
        ":generate_amdgpu_device_lib_data_tool",
        "@llvm-project//llvm:llvm-link",
    ],
)

cc_library(
    name = "amdgpu_device_lib_data",
    hdrs = [
        ":generate_amdgpu_device_lib_data",
    ],
    tags = if_google([
        # Embedding libdevice is not supported in the Google-internal build.
        "manual",
        "notap",
        "nobuilder",
    ]),
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "amdgpu_backend",
    srcs = ["amdgpu_backend.cc"],
    hdrs = ["amdgpu_backend.h"],
    local_defines = if_oss([
        "HAS_SUPPORT_FOR_LLD_AS_A_LIBRARY=1",
        "HAS_SUPPORT_FOR_EMBEDDED_LIB_DEVICE=1",
    ]),
    deps = [
        ":llvm_gpu_backend",
        ":load_ir_module",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:rocm_rocdl_path",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:AMDGPUAsmParser",  # buildcleaner: keep
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/profiler/lib:traceme",
    ] + if_oss([
        # keep sorted
        ":amdgpu_device_lib_data",
        "@llvm-project//lld:Common",
        "@llvm-project//lld:ELF",  # buildcleaner: keep
    ]),
)

cc_library(
    name = "load_ir_module",
    hdrs = ["load_ir_module.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ] + if_google(
        ["//xla/service/gpu/llvm_gpu_backend/google:load_ir_module"],
        ["//xla/service/gpu/llvm_gpu_backend/default:load_ir_module"],
    ),
)

cc_library(
    name = "nvptx_libdevice_path",
    hdrs = ["nvptx_libdevice_path.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ] + if_google(
        ["//xla/service/gpu/llvm_gpu_backend/google:nvptx_libdevice_path"],
        ["//xla/service/gpu/llvm_gpu_backend/default:nvptx_libdevice_path"],
    ),
)

cc_library(
    name = "nvptx_utils",
    srcs = ["nvptx_utils.cc"],
    hdrs = ["nvptx_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:cuda_root_path",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "nvptx_backend_test",
    size = "small",
    srcs = ["nvptx_backend_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":nvptx_backend",
        ":ptx_version_util",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "amdgpu_bitcode_link_test",
    size = "small",
    srcs = ["amdgpu_bitcode_link_test.cc"],
    data = [
        "tests_data/amdgpu.ll",
    ],
    tags = if_google([
        # Embedded libdevice is required for this test, but not supported in the Google-internal build.
        "notap",
        "manual",
        "nobuilder",
    ]) + [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":amdgpu_backend",
        ":load_ir_module",
        "//xla/tsl/platform:rocm_rocdl_path",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "load_ir_module_test",
    size = "small",
    srcs = ["load_ir_module_test.cc"],
    data = ["tests_data/saxpy.ll"],
    deps = [
        ":load_ir_module",
        "//xla/tests:xla_internal_test_main",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "nvptx_utils_test",
    srcs = ["nvptx_utils_test.cc"],
    deps = [
        ":nvptx_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "spirv_backend",
    srcs = [
        "spirv_backend.cc",
    ],
    hdrs = [
        "spirv_backend.h",
    ],
    tags = [
        "gpu",
        "oneapi-only",
    ] + if_google([
        # TODO(b/456585142): Currently we don't support building the SYCL backend.
        "notap",
        "nobuilder",
        "manual",
    ]),
    deps = [
        ":llvm_gpu_backend",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "@llvm-project//llvm:SPIRVCodeGen",
        "@local_tsl//tsl/platform:errors",
    ],
)
