licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "curand_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcurand.so.%{libcurand_version}",
)
%{multiline_comment}
cc_library(
    name = "curand",
    %{comment}deps = [":curand_shared_library"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob(["include/curand*.h"]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
