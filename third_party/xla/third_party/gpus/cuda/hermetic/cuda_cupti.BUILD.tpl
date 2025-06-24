licenses(["restricted"])  # NVIDIA proprietary license
load("@local_config_cuda//cuda:build_defs.bzl", "if_version_equal_or_greater_than")
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

%{multiline_comment}
cc_import(
    name = "cupti_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcupti.so.%{libcupti_version}",
)
%{multiline_comment}
cc_library(
    name = "cupti",
    %{comment}deps = [":cupti_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/cuda_cupti/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/Openacc/**",
        %{comment}"include/Openmp/**",
        %{comment}"include/cuda_stdint.h",
        %{comment}"include/cupti.h",
        %{comment}"include/cupti_activity.h",
        %{comment}"include/cupti_activity_deprecated.h",
        %{comment}"include/cupti_callbacks.h",
        %{comment}"include/cupti_checkpoint.h",
        %{comment}"include/cupti_driver_cbid.h",
        %{comment}"include/cupti_events.h",
        %{comment}"include/cupti_metrics.h",
        %{comment}"include/cupti_nvtx_cbid.h",
        %{comment}"include/cupti_pcsampling.h",
        %{comment}"include/cupti_pcsampling_util.h",
        %{comment}"include/cupti_profiler_target.h",
        %{comment}"include/cupti_result.h",
        %{comment}"include/cupti_runtime_cbid.h",
        %{comment}"include/cupti_sass_metrics.h",
        %{comment}"include/cupti_target.h",
        %{comment}"include/cupti_version.h",
        %{comment}"include/generated_cudaGL_meta.h",
        %{comment}"include/generated_cudaVDPAU_meta.h",
        %{comment}"include/generated_cuda_gl_interop_meta.h",
        %{comment}"include/generated_cuda_meta.h",
        %{comment}"include/generated_cuda_runtime_api_meta.h",
        %{comment}"include/generated_cuda_vdpau_interop_meta.h",
        %{comment}"include/generated_cudart_removed_meta.h",
        %{comment}"include/generated_nvtx_meta.h",
        %{comment}"include/nvperf_common.h",
        %{comment}"include/nvperf_cuda_host.h",
        %{comment}"include/nvperf_host.h",
        %{comment}"include/nvperf_target.h",
    %{comment}]) + if_version_equal_or_greater_than(
        %{comment}"%{libcupti_minor_version}",
        %{comment}"2024.0",
        %{comment}["include/cupti_common.h"],
    %{comment}),
    include_prefix = "third_party/gpus/cuda/extras/CUPTI/include",
    includes = ["include/"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
