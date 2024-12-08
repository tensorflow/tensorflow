licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "nvjitlink_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libnvJitLink.so.%{libnvjitlink_version}",
    linkopts = ["-Wl,-rpath,$ORIGIN/../nvidia/nvjitlink/lib",
                "-Wl,-rpath,$ORIGIN/../../nvidia/nvjitlink/lib",
               ],
)
%{multiline_comment}
cc_library(
    name = "nvjitlink",
    %{comment}deps = [":nvjitlink_shared_library"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = ["include/nvJitLink.h"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)

