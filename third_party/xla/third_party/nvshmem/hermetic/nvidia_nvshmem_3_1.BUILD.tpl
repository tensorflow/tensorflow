licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

%{multiline_comment}
cc_import(
    name = "nvshmem_host_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libnvshmem_host.so.%{libnvshmem_host_version}",
)

cc_import(
    name = "nvshmem_bootstrap_mpi_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_bootstrap_mpi.so.%{nvshmem_bootstrap_mpi_version}",
)

cc_import(
    name = "nvshmem_bootstrap_pmi_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_bootstrap_pmi.so.%{nvshmem_bootstrap_pmi_version}",
)

cc_import(
    name = "nvshmem_bootstrap_pmi2_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_bootstrap_pmi2.so.%{nvshmem_bootstrap_pmi2_version}",
)

cc_import(
    name = "nvshmem_bootstrap_pmix_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_bootstrap_pmix.so.%{nvshmem_bootstrap_pmix_version}",
)

cc_import(
    name = "nvshmem_bootstrap_shmem_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_bootstrap_shmem.so.%{nvshmem_bootstrap_shmem_version}",
)

cc_import(
    name = "nvshmem_bootstrap_uid_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_bootstrap_uid.so.%{nvshmem_bootstrap_uid_version}",
)

cc_import(
    name = "nvshmem_transport_ibdevx_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_transport_ibdevx.so.%{nvshmem_transport_ibdevx_version}",
)

cc_import(
    name = "nvshmem_transport_ibgda_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_transport_ibgda.so.%{nvshmem_transport_ibgda_version}",
)

cc_import(
    name = "nvshmem_transport_ibrc_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_transport_ibrc.so.%{nvshmem_transport_ibrc_version}",
)

cc_import(
    name = "nvshmem_transport_libfabric_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_transport_libfabric.so.%{nvshmem_transport_libfabric_version}",
)

cc_import(
    name = "nvshmem_transport_ucx_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/nvshmem_transport_ucx.so.%{nvshmem_transport_ucx_version}",
)
%{multiline_comment}
cc_library(
    name = "nvshmem",
    %{comment}deps = [
      %{comment}":nvshmem_host_shared_library", 
      %{comment}":nvshmem_bootstrap_mpi_shared_library",
      %{comment}":nvshmem_bootstrap_pmi_shared_library",
      %{comment}":nvshmem_bootstrap_pmi2_shared_library",
      %{comment}":nvshmem_bootstrap_pmix_shared_library",
      %{comment}":nvshmem_bootstrap_shmem_shared_library",
      %{comment}":nvshmem_bootstrap_uid_shared_library",
      %{comment}":nvshmem_transport_ibdevx_shared_library",
      %{comment}":nvshmem_transport_ibgda_shared_library",
      %{comment}":nvshmem_transport_ibrc_shared_library",
      %{comment}":nvshmem_transport_libfabric_shared_library",
      %{comment}":nvshmem_transport_ucx_shared_library",
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