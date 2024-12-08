licenses(["restricted"])  # NVIDIA proprietary license
%{multiline_comment}
cc_import(
    name = "nvrtc_main",
    shared_library = "lib/libnvrtc.so.%{libnvrtc_version}",
    linkopts = ["-Wl,-rpath,$ORIGIN/../nvidia/cuda_nvrtc/lib",
                "-Wl,-rpath,$ORIGIN/../../nvidia/cuda_nvrtc/lib",
               ],
)

cc_import(
    name = "nvrtc_builtins",
    shared_library = "lib/libnvrtc-builtins.so.%{libnvrtc-builtins_version}",
    linkopts = ["-Wl,-rpath,$ORIGIN/../nvidia/cuda_nvrtc/lib",
                "-Wl,-rpath,$ORIGIN/../../nvidia/cuda_nvrtc/lib",
               ],
)
%{multiline_comment}
cc_library(
    name = "nvrtc",
    %{comment}deps = [
        %{comment}":nvrtc_main",
        %{comment}":nvrtc_builtins",
    %{comment}],
    visibility = ["//visibility:public"],
)
