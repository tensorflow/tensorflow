licenses(["restricted"])  # NVIDIA proprietary license

%{multiline_comment}
cc_import(
    name = "driver_shared_library",
    shared_library = "lib/libcuda.so.%{libcuda_version}",
)
cc_import(
    name = "driver_symlink",
    shared_library = "lib/libcuda.so.1",
)
%{multiline_comment}
cc_library(
    name = "nvidia_driver",
    %{comment}deps = [":driver_shared_library", ":driver_symlink"],
    visibility = ["//visibility:public"],
)
