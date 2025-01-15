load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

licenses(["restricted"])  # NVIDIA proprietary license

load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

exports_files([
    "version.txt",
])

filegroup(
    name = "static",
    srcs = ["lib/libcudart_static.a"],
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
%{multiline_comment}
cc_import(
    name = "cuda_stub",
    interface_library = "lib/stubs/libcuda.so",
    system_provided = 1,
)

cc_import(
    name = "cudart_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcudart.so.%{libcudart_version}",
)
%{multiline_comment}
cc_library(
    name = "cuda_driver_stub",
    %{comment}deps = [":cuda_stub"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart",
    %{comment}deps = select({
        %{comment}":forward_compatibility_mode": ["@cuda_driver//:nvidia_driver"],
        %{comment}":forward_compatibility_mode_override": ["@cuda_driver//:nvidia_driver"],
        %{comment}":nvidia_cuda_driver_stub": [":cuda_driver_stub"],
        %{comment}"//conditions:default": [],
    %{comment}}) + [
        %{comment}":cudart_shared_library",
    %{comment}],
    %{comment}linkopts = cuda_rpath_flags("nvidia/cuda_runtime/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/builtin_types.h",
        %{comment}"include/channel_descriptor.h",
        %{comment}"include/common_functions.h",
        %{comment}"include/cooperative_groups/**",
        %{comment}"include/cooperative_groups.h",
        %{comment}"include/cuComplex.h",
        %{comment}"include/cuda.h",
        %{comment}"include/cudaEGL.h",
        %{comment}"include/cudaEGLTypedefs.h",
        %{comment}"include/cudaGL.h",
        %{comment}"include/cudaGLTypedefs.h",
        %{comment}"include/cudaProfilerTypedefs.h",
        %{comment}"include/cudaTypedefs.h",
        %{comment}"include/cudaVDPAU.h",
        %{comment}"include/cudaVDPAUTypedefs.h",
        %{comment}"include/cuda_awbarrier.h",
        %{comment}"include/cuda_awbarrier_helpers.h",
        %{comment}"include/cuda_awbarrier_primitives.h",
        %{comment}"include/cuda_bf16.h",
        %{comment}"include/cuda_bf16.hpp",
        %{comment}"include/cuda_device_runtime_api.h",
        %{comment}"include/cuda_egl_interop.h",
        %{comment}"include/cuda_fp16.h",
        %{comment}"include/cuda_fp16.hpp",
        %{comment}"include/cuda_fp8.h",
        %{comment}"include/cuda_fp8.hpp",
        %{comment}"include/cuda_gl_interop.h",
        %{comment}"include/cuda_occupancy.h",
        %{comment}"include/cuda_pipeline.h",
        %{comment}"include/cuda_pipeline_helpers.h",
        %{comment}"include/cuda_pipeline_primitives.h",
        %{comment}"include/cuda_runtime.h",
        %{comment}"include/cuda_runtime_api.h",
        %{comment}"include/cuda_surface_types.h",
        %{comment}"include/cuda_texture_types.h",
        %{comment}"include/cuda_vdpau_interop.h",
        %{comment}"include/cudart_platform.h",
        %{comment}"include/device_atomic_functions.h",
        %{comment}"include/device_atomic_functions.hpp",
        %{comment}"include/device_double_functions.h",
        %{comment}"include/device_functions.h",
        %{comment}"include/device_launch_parameters.h",
        %{comment}"include/device_types.h",
        %{comment}"include/driver_functions.h",
        %{comment}"include/driver_types.h",
        %{comment}"include/host_config.h",
        %{comment}"include/host_defines.h",
        %{comment}"include/library_types.h",
        %{comment}"include/math_constants.h",
        %{comment}"include/math_functions.h",
        %{comment}"include/mma.h",
        %{comment}"include/nvfunctional",
        %{comment}"include/sm_20_atomic_functions.h",
        %{comment}"include/sm_20_atomic_functions.hpp",
        %{comment}"include/sm_20_intrinsics.h",
        %{comment}"include/sm_20_intrinsics.hpp",
        %{comment}"include/sm_30_intrinsics.h",
        %{comment}"include/sm_30_intrinsics.hpp",
        %{comment}"include/sm_32_atomic_functions.h",
        %{comment}"include/sm_32_atomic_functions.hpp",
        %{comment}"include/sm_32_intrinsics.h",
        %{comment}"include/sm_32_intrinsics.hpp",
        %{comment}"include/sm_35_atomic_functions.h",
        %{comment}"include/sm_35_intrinsics.h",
        %{comment}"include/sm_60_atomic_functions.h",
        %{comment}"include/sm_60_atomic_functions.hpp",
        %{comment}"include/sm_61_intrinsics.h",
        %{comment}"include/sm_61_intrinsics.hpp",
        %{comment}"include/surface_functions.h",
        %{comment}"include/surface_indirect_functions.h",
        %{comment}"include/surface_types.h",
        %{comment}"include/texture_fetch_functions.h",
        %{comment}"include/texture_indirect_functions.h",
        %{comment}"include/texture_types.h",
        %{comment}"include/vector_functions.h",
        %{comment}"include/vector_functions.hpp",
        %{comment}"include/vector_types.h",
    %{comment}]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)

# Flag indicating if we should add dependency on libcuda stub.
bool_flag(
    name = "enable_driver_stub",
    build_setting_default = True,
)

config_setting(
    name = "driver_stub",
    flag_values = {":enable_driver_stub": "True"},
)

config_setting(
    name = "no_driver_stub",
    flag_values = {":enable_driver_stub": "False"},
)

selects.config_setting_group(
    name = "forward_compatibility_mode",
    match_all = [
        "@cuda_driver//:forward_compatibility",
        ":no_driver_stub"
    ],
)

selects.config_setting_group(
    name = "forward_compatibility_mode_override",
    match_all = [
        "@cuda_driver//:forward_compatibility",
        ":driver_stub"
    ],
)

selects.config_setting_group(
    name = "nvidia_cuda_driver_stub",
    match_all = [
        "@cuda_driver//:no_forward_compatibility",
        ":driver_stub"
    ],
)