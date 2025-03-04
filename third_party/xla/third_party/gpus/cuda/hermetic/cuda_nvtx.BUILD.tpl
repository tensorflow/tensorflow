licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/nvToolsExt*.h",
        %{comment}"include/nvtx3/**",
    %{comment}]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
