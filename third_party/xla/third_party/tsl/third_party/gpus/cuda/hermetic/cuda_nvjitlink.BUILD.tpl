licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "nvjitlink_shared_library",
    shared_library = "lib/libnvJitLink.so.%{libnvjitlink_version}",
)
%{multiline_comment}
cc_library(
    name = "nvjitlink",
    %{comment}deps = [":nvjitlink_shared_library"],
    visibility = ["//visibility:public"],
)

