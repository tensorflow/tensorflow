licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_library(
    name = "headers",
    hdrs = glob([
        %{comment}"include/cub/**",
        %{comment}"include/cuda/**",
        %{comment}"include/nv/**",
        %{comment}"include/thrust/**",
    ]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
