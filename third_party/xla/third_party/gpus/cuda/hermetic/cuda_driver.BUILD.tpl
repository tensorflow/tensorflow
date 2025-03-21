load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

licenses(["restricted"])  # NVIDIA proprietary license

%{multiline_comment}
cc_import(
    name = "driver_shared_library",
    shared_library = "lib/libcuda.so.%{libcuda_version}",
)

cc_import(
    name = "libcuda_so_1",
    shared_library = "lib/libcuda.so.1",
)

# TODO(ybaturina): remove workaround when min CUDNN version in JAX is updated to
# 9.3.0.
# Workaround for adding path of driver library symlink to RPATH of cc_binaries.
cc_import(
    name = "libcuda_so",
    shared_library = "lib/libcuda.so",
)

# Workaround for adding libcuda.so to NEEDED section of cc_binaries.
genrule(
    name = "fake_libcuda_cc",
    outs = ["libcuda.cc"],
    cmd = "echo '' > $@",
)

cc_binary(
    name = "fake_libcuda_binary",
    srcs = [":fake_libcuda_cc"],
    linkopts = ["-Wl,-soname,libcuda.so"],
    linkshared = True,
)

cc_import(
    name = "fake_libcuda",
    shared_library = ":fake_libcuda_binary",
)
%{multiline_comment}
cc_library(
    name = "nvidia_driver",
    %{comment}deps = [
        %{comment}":libcuda_so",
        %{comment}":fake_libcuda",
        %{comment}":libcuda_so_1",
        %{comment}":driver_shared_library",
    %{comment}],
    visibility = ["//visibility:public"],
)

# Flag indicating if we should enable forward compatibility.
bool_flag(
    name = "enable_forward_compatibility",
    build_setting_default = False,
)

config_setting(
    name = "forward_compatibility",
    flag_values = {":enable_forward_compatibility": "True"},
)
