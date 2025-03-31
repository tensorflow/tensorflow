load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "llvm_gpu_backend",
    srcs = [
        "gpu_backend_lib.cc",
    ],
    hdrs = [
        "gpu_backend_lib.h",
    ],
    deps = [
        ":load_ir_module",
        ":utils",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/stream_executor:device_description",
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
    name = "nvptx_backend",
    srcs = [
        "nvptx_backend.cc",
    ],
    hdrs = [
        "nvptx_backend.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":llvm_gpu_backend",
        ":load_ir_module",
        ":nvptx_libdevice_path",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/service/gpu:metrics",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/cuda:ptx_compiler_helpers",
        "//xla/stream_executor/cuda:subprocess_compilation",
        "@com_google_absl//absl/base",
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
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "amdgpu_backend",
    srcs = [
        "amdgpu_backend.cc",
    ],
    hdrs = [
        "amdgpu_backend.h",
    ],
    deps = [
        ":llvm_gpu_backend",
        ":load_ir_module",
        ":utils",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/service/gpu:metrics",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/cuda:subprocess_compilation",
        "//xla/tsl/platform:rocm_rocdl_path",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:AMDGPUAsmParser",
        "@llvm-project//llvm:AMDGPUCodeGen",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:NVVMDialect",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:cuda_root_path",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
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
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "load_ir_module_test",
    size = "small",
    srcs = ["load_ir_module_test.cc"],
    data = [
        "tests_data/saxpy.ll",
    ],
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
