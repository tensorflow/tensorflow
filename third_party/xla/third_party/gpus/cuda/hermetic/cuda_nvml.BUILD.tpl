licenses(["restricted"])  # NVIDIA proprietary license

cc_library(
    name = "headers",
    %{comment}hdrs = ["include/nvml.h"],
    include_prefix = "third_party/gpus/cuda/nvml/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
