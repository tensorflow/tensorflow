licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

filegroup(
    name = "libnvshmem_device",
    srcs = [
        "lib/libnvshmem_device.bc",
    ],
    visibility = ["//visibility:public"],
)

%{multiline_comment}
cc_import(
    name = "nvshmem_host_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libnvshmem_host.so.%{libnvshmem_host_version}",
)

cc_import(
    name = "nvshmem_bootstrap_uid_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_bootstrap_uid.so.%{nvshmem_bootstrap_uid_version}",
)

cc_import(
    name = "nvshmem_transport_ibrc_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_transport_ibrc.so.%{nvshmem_transport_ibrc_version}",
)

# Workaround for adding nvshmem_bootstrap_uid library symlink to cc_binaries.
cc_import(
    name = "nvshmem_bootstrap_uid_so",
    shared_library = "lib/nvshmem_bootstrap_uid.so",
)

# Workaround for adding nvshmem_bootstrap_uid.so to NEEDED section
# of cc_binaries.
genrule(
    name = "fake_nvshmem_bootstrap_uid_cc",
    outs = ["nvshmem_bootstrap_uid.cc"],
    cmd = "echo '' > $@",
)

cc_binary(
    name = "nvshmem_bootstrap_uid.so",
    srcs = [":fake_nvshmem_bootstrap_uid_cc"],
    linkopts = ["-Wl,-soname,nvshmem_bootstrap_uid.so"],
    linkshared = True,
)

cc_import(
    name = "fake_nvshmem_bootstrap_uid",
    shared_library = ":nvshmem_bootstrap_uid.so",
)
%{multiline_comment}
cc_library(
    name = "nvshmem",
    %{comment}deps = [
      %{comment}":nvshmem_host_shared_library",
      %{comment}":nvshmem_bootstrap_uid_so",
      %{comment}":fake_nvshmem_bootstrap_uid",
      %{comment}":nvshmem_bootstrap_uid_shared_library",
      %{comment}":nvshmem_transport_ibrc_shared_library",
    %{comment}],
    %{comment}linkopts = cuda_rpath_flags("nvidia/nvshmem/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/**",
    %{comment}]),
    include_prefix = "third_party/nvshmem",
    includes = ["include"],
    strip_include_prefix = "include",
)