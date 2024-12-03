licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "cufft_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcufft.so.%{libcufft_version}",
)
%{multiline_comment}
cc_library(
    name = "cufft",
    %{comment}deps = [":cufft_shared_library"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/cudalibxt.h", 
        %{comment}"include/cufft*.h"
    %{comment}]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
