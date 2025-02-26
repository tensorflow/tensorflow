load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_library",
)
load(
    "@local_config_nccl//:build_defs.bzl",
    "cuda_rdc_library",
)

licenses(["notice"])

exports_files(["LICENSE.txt"])

bool_flag(
    name = "nvshmem_enable_all_device_inlining",
    build_setting_default = False,
)

config_setting(
    name = "enable_all_device_inlining",
    flag_values = {
        ":nvshmem_enable_all_device_inlining": "True",
    },
)

cc_binary(
    name = "perftest_coll_reduction_on_stream",
    srcs = [
        "perftest/host/coll/reduction_on_stream.cpp",
    ],
    includes = [
        "perftest/host/coll",
    ],
    deps = [
        ":nvshmem",
        ":perftest_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
)

cuda_library(
    name = "perftest_common",
    srcs = [
        "perftest/common/utils.cu",
    ],
    hdrs = glob([
        "perftest/host/coll/coll_test.h",
        "perftest/common/*.h",
    ]),
    copts = [
        "-Wno-vla-cxx-extension",
    ],
    includes = [
        "perftest/common",
    ],
    deps = [
        ":nvshmem_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
)

cc_binary(
    name = "hello_example",
    srcs = [
        "examples/hello.cpp",
    ],
    copts = [
        "-Wno-unused-variable",
    ],
    includes = [
        "src/include",
    ],
    local_defines = [
        "NVSHMEM_BOOTSTRAP=PLUGIN",
    ],
    deps = [
        ":nvshmem",
        ":nvshmem_hdrs",
    ],
)

cc_binary(
    name = "nvshmem_info",
    srcs = [
        "src/bin/nvshmem-info.cpp",
    ],
    includes = [
        "src/include",
        "src/modules/bootstrap/common",
    ],
    deps = [
        ":nvshmem",
        ":nvshmem_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "nvshmem_host",
    deps = [
        ":bootstrap_uid_plugin",
        ":nvshmem_hdrs",
        ":nvshmem_host_sources",
        ":nvshmem_host_sources_nomaxregcount",
    ],
)

cc_library(
    name = "nvshmem",
    hdrs = [
        ":nvshmem_build_options_h",
        ":nvshmem_version_h",
    ],
    include_prefix = "third_party/nvshmem",
    includes = ["src/include"],
    strip_include_prefix = "src/include",
    visibility = ["//visibility:public"],
    deps = [
        ":bootstrap_uid_plugin",
        ":nvshmem_device_sources",
        ":nvshmem_hdrs",
        ":nvshmem_host_sources",
        ":nvshmem_host_sources_nomaxregcount",
    ],
)

cuda_library(
    name = "nvshmem_device_sources",
    srcs = [
        "src/device/launch/collective_launch.cpp",
    ] + select(
        {
            ":enable_all_device_inlining": [],
            "//conditions:default": [
                ":transfer_device_source",
            ],
        },
    ),
    copts = [
        "-Xcuda-ptxas --maxrregcount=32",
    ],
    local_defines = [
        "__CUDACC_RDC__",
    ],
    deps = [
        ":nvshmem_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
    alwayslink = 1,
)

cuda_rdc_library(
    name = "nvshmem_host_sources_nomaxregcount",
    srcs = [
        # ToDo(patrios): Temporary solution.
        # In cmake configurations init_device.cu is in the nvshmem_device_sources.
        # However with bazel+nvcc this leads to unresolved nvshmemi_device_state_d reference.
        # Command to validate this issue:
        # bazel build --config=cuda @nvshmem//:nvshmem_host_sources_nomaxregcount_dlink_hdrsd
        "src/device/init/init_device.cu.cc",
        "src/host/comm/rma.cu.cc",
        "src/host/stream/coll/alltoall/alltoall.cu.cc",
        "src/host/stream/coll/barrier/barrier.cu.cc",
        "src/host/stream/coll/broadcast/broadcast.cu.cc",
        "src/host/stream/coll/fcollect/fcollect.cu.cc",
        "src/host/stream/coll/rdxn/reduce_and.cu.cc",
        "src/host/stream/coll/rdxn/reduce_max.cu.cc",
        "src/host/stream/coll/rdxn/reduce_min.cu.cc",
        "src/host/stream/coll/rdxn/reduce_or.cu.cc",
        "src/host/stream/coll/rdxn/reduce_prod.cu.cc",
        "src/host/stream/coll/rdxn/reduce_sum.cu.cc",
        "src/host/stream/coll/rdxn/reduce_team.cu.cc",
        "src/host/stream/coll/rdxn/reduce_xor.cu.cc",
        "src/host/stream/coll/reducescatter/reducescatter_and.cu.cc",
        "src/host/stream/coll/reducescatter/reducescatter_max.cu.cc",
        "src/host/stream/coll/reducescatter/reducescatter_min.cu.cc",
        "src/host/stream/coll/reducescatter/reducescatter_or.cu.cc",
        "src/host/stream/coll/reducescatter/reducescatter_prod.cu.cc",
        "src/host/stream/coll/reducescatter/reducescatter_sum.cu.cc",
        "src/host/stream/coll/reducescatter/reducescatter_xor.cu.cc",
        "src/host/stream/comm/cuda_interface_sync.cu.cc",
        "src/host/stream/comm/quiet_on_stream.cu.cc",
    ] + select(
        {
            ":enable_all_device_inlining": [],
            "//conditions:default": [
                ":transfer_device_source",
            ],
        },
    ),
    hdrs = [
        "src/host/stream/coll/rdxn/reduce_common.cuh",
        "src/host/stream/coll/reducescatter/reducescatter_common.cuh",
        "src/include/device/nvshmem_defines.h",
        "src/include/internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh",
        "src/include/internal/non_abi/nvshmemi_h_to_d_rma_defs.cuh",
        "src/include/internal/non_abi/nvshmemi_h_to_d_sync_defs.cuh",
    ],
    includes = [
        "src/host/stream/coll/rdxn",
        "src/host/stream/coll/reducescatter",
    ],
    local_defines = [
        "__CUDACC_RDC__",
    ],
    deps = [
        ":nvshmem_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
)

cuda_library(
    name = "nvshmem_host_sources",
    srcs = [
        "src/host/bootstrap/bootstrap.cpp",
        "src/host/bootstrap/bootstrap_loader.cpp",
        "src/host/coll/alltoall/alltoall_on_stream.cpp",
        "src/host/coll/barrier/barrier.cpp",
        "src/host/coll/barrier/barrier_on_stream.cpp",
        "src/host/coll/broadcast/broadcast.cpp",
        "src/host/coll/broadcast/broadcast_on_stream.cpp",
        "src/host/coll/cpu_coll.cpp",
        "src/host/coll/fcollect/fcollect.cpp",
        "src/host/coll/fcollect/fcollect_on_stream.cpp",
        "src/host/coll/rdxn/rdxn.cpp",
        "src/host/coll/rdxn/rdxn_on_stream.cpp",
        "src/host/coll/reducescatter/reducescatter.cpp",
        "src/host/coll/reducescatter/reducescatter_on_stream.cpp",
        "src/host/comm/amo.cpp",
        "src/host/comm/fence.cpp",
        "src/host/comm/putget.cpp",
        "src/host/comm/quiet.cpp",
        "src/host/comm/sync.cpp",
        "src/host/init/cudawrap.cpp",
        "src/host/init/init.cu.cc",
        "src/host/init/init_nvtx.cpp",
        "src/host/init/nvmlwrap.cpp",
        "src/host/init/query_host.cpp",
        "src/host/mem/custom_malloc.cpp",
        "src/host/mem/mem.cpp",
        "src/host/mem/mem_heap.cpp",
        "src/host/mem/mem_transport.cpp",
        "src/host/proxy/proxy.cpp",
        "src/host/team/team.cu.cc",
        "src/host/team/team_internal.cpp",
        "src/host/team/team_internal_cuda.cu.cc",
        "src/host/team/team_internal_nvls.cpp",
        "src/host/topo/topo.cpp",
        "src/host/transport/p2p/p2p.cpp",
        "src/host/transport/transport.cpp",
        "src/host/util/cs.cpp",
        "src/host/util/debug.cpp",
        "src/host/util/env_vars.cpp",
        "src/host/util/shared_memory.cpp",
        "src/host/util/sockets.cpp",
        "src/host/util/util.cpp",
    ],
    copts = [
        "-Wno-vla-cxx-extension",
        "-Wno-missing-braces",
        "-Xcuda-ptxas --maxrregcount=32",
    ],
    includes = [
        "src/host/coll",
        "src/host/coll/fcollect",
        "src/host/coll/reducescatter",
        "src/host/proxy",
        "src/host/stream/comm/rdxn",
        "src/host/topo",
        "src/host/transport/p2p",
    ],
    linkopts = ["-lm"],
    deps = [
        ":nvshmem_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
)

cc_library(
    name = "bootstrap_uid_plugin",
    srcs = glob(
        [
            "src/modules/bootstrap/common/bootstrap_util.cpp",
            "src/modules/bootstrap/uid/bootstrap_uid.cpp",
            "src/modules/bootstrap/uid/ncclSocket/ncclsocket_socket.cpp",
        ],
    ),
    hdrs = glob(
        [
            "src/include/internal/bootstrap_host/nvshmemi_bootstrap.h",
            "src/include/internal/host/nvshmemi_bootstrap_library.h",
            "src/modules/bootstrap/common/bootstrap_util.h",
            "src/modules/bootstrap/uid/bootstrap_uid_remap.h",
            "src/modules/bootstrap/uid/bootstrap_uid_types.hpp",
            "src/modules/bootstrap/uid/ncclSocket/*.h",
            "src/modules/bootstrap/uid/ncclSocket/*.hpp",
        ],
    ),
    copts = [
        "-Wno-missing-braces",
    ],
    includes = [
        "src/modules/bootstrap/common",
        "src/modules/bootstrap/uid/ncclSocket",
    ],
    deps = [
        ":nvshmem_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
    alwayslink = 1,
)

cc_library(
    name = "nvshmem_hdrs",
    hdrs = glob(
        [
            "src/host/coll/alltoall/alltoall.h",
            "src/host/coll/barrier/barrier.h",
            "src/host/coll/broadcast/broadcast.h",
            "src/host/coll/cpu_coll.h",
            "src/host/coll/fcollect/fcollect.h",
            "src/host/coll/rdxn/rdxn.h",
            "src/host/coll/reducescatter/reducescatter.h",
            "src/host/proxy/proxy_host.h",
            "src/host/team/team_internal.h",
            "src/host/topo/topo.h",
            "src/host/transport/p2p/p2p.h",
            "src/include/bootstrap_device_host/nvshmem_uniqueid.h",
            "src/include/bootstrap_host_transport/env_defs_internal.h",
            "src/include/device_host_transport/nvshmem_common_transport.h",
            "src/include/device_host_transport/nvshmem_constants.h",
            "src/include/device_host/nvshmem_common.cuh",
            "src/include/device_host/nvshmem_proxy_channel.h",
            "src/include/device_host/nvshmem_types.h",
            "src/include/device/nvshmem_coll_defines.cuh",
            "src/include/device/nvshmem_device_macros.h",
            "src/include/device/nvshmemx_coll_defines.cuh",
            "src/include/device/nvshmemx_collective_launch_apis.h",
            "src/include/device/nvshmemx_defines.h",
            "src/include/host/env/env_defs.h",
            "src/include/host/nvshmem_api.h",
            "src/include/host/nvshmem_coll_api.h",
            "src/include/host/nvshmem_macros.h",
            "src/include/host/nvshmemx_api.h",
            "src/include/host/nvshmemx_coll_api.h",
            "src/include/internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h",
            "src/include/internal/bootstrap_host/nvshmemi_bootstrap.h",
            "src/include/internal/host_transport/cudawrap.h",
            "src/include/internal/host_transport/nvshmemi_transport_defines.h",
            "src/include/internal/host_transport/transport.h",
            "src/include/internal/host/cuda_interface_sync.h",
            "src/include/internal/host/custom_malloc.h",
            "src/include/internal/host/debug.h",
            "src/include/internal/host/error_codes_internal.h",
            "src/include/internal/host/nvmlwrap.h",
            "src/include/internal/host/nvshmem_internal.h",
            "src/include/internal/host/nvshmem_nvtx.hpp",
            "src/include/internal/host/nvshmemi_bootstrap_library.h",
            "src/include/internal/host/nvshmemi_coll.h",
            "src/include/internal/host/nvshmemi_mem_transport.hpp",
            "src/include/internal/host/nvshmemi_nvls_rsc.hpp",
            "src/include/internal/host/nvshmemi_symmetric_heap.hpp",
            "src/include/internal/host/nvshmemi_team.h",
            "src/include/internal/host/nvshmemi_types.h",
            "src/include/internal/host/nvtx3.hpp",
            "src/include/internal/host/shared_memory.h",
            "src/include/internal/host/sockets.h",
            "src/include/internal/host/util.h",
            "src/include/internal/device/nvshmemi_device.h",
            "src/include/non_abi/device/coll/alltoall.cuh",
            "src/include/non_abi/device/coll/barrier.cuh",
            "src/include/non_abi/device/coll/broadcast.cuh",
            "src/include/non_abi/device/coll/defines.cuh",
            "src/include/non_abi/device/coll/fcollect.cuh",
            "src/include/non_abi/device/coll/reduce.cuh",
            "src/include/non_abi/device/coll/reducescatter.cuh",
            "src/include/non_abi/device/coll/utils.cuh",
            "src/include/non_abi/device/common/nvshmemi_common_device.cuh",
            "src/include/non_abi/device/pt-to-pt/ibgda_device.cuh",
            "src/include/non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh",
            "src/include/non_abi/device/pt-to-pt/proxy_device.cuh",
            "src/include/non_abi/device/pt-to-pt/utils_device.h",
            "src/include/non_abi/device/team/nvshmemi_team_defines.cuh",
            "src/include/non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh",
            "src/include/non_abi/device/wait/nvshmemi_wait_until_apis.cuh",
            "src/include/non_abi/nvshmemx_error.h",
            "src/include/nvshmem.h",
            "src/include/nvshmemx.h",
            "src/modules/bootstrap/common/env_defs.h",
            "src/include/device/nvshmem_defines.h",
        ],
    ) + [
        ":nvshmem_build_options_h",
        ":nvshmem_version_h",
    ] + select(
        {
            ":enable_all_device_inlining": [
                ":transfer_device_header",
            ],
            "//conditions:default": [
            ],
        },
    ),
    copts = [
        "-Wno-missing-braces",
    ],
    includes = [
        "src/host/coll",
        "src/include",
        "src/include/host/env",
        "src/modules/bootstrap/uid/ncclSocket",
    ],
    local_defines = [
        "__CUDACC_RDC__=1",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
)

NVSHMEM_MAJOR = 3

version_substitions = {
    "@PROJECT_VERSION_MAJOR@": str(NVSHMEM_MAJOR),
    "@PROJECT_VERSION_MINOR@": "0",
    "@PROJECT_VERSION_PATCH@": "6",
    "@PROJECT_VERSION_TWEAK@": "4",
    "@TRANSPORT_VERSION_MAJOR@": "3",
    "@TRANSPORT_VERSION_MINOR@": "0",
    "@TRANSPORT_VERSION_PATCH@": "0",
    "@BOOTSTRAP_VERSION_MAJOR@": "3",
    "@BOOTSTRAP_VERSION_MINOR@": "0",
    "@BOOTSTRAP_VERSION_PATCH@": "0",
    "@INTERLIB_VERSION_MAJOR@": "3",
    "@INTERLIB_VERSION_MINOR@": "0",
    "@INTERLIB_VERSION_PATCH@": "0",
    "@INFO_BUILD_VARS@": "\"\"",
}

expand_template(
    name = "nvshmem_version_h",
    out = "src/include/non_abi/nvshmem_version.h",
    substitutions = version_substitions,
    template = "src/include/non_abi/nvshmem_version.h.in",
)

expand_template(
    name = "transfer_device_header",
    out = "src/include/non_abi/device/pt-to-pt/transfer_device.cuh",
    substitutions = {},
    template = "src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in",
)

expand_template(
    name = "transfer_device_source",
    out = "src/device/comm/transfer_device.cu.cc",
    substitutions = {},
    template = "src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in",
)

options_substitions_common = {
    "#cmakedefine NVSHMEM_COMPLEX_SUPPORT": "/* #undef NVSHMEM_COMPLEX_SUPPORT */",
    "#cmakedefine NVSHMEM_DEBUG": "/* #undef NVSHMEM_DEBUG */",
    "#cmakedefine NVSHMEM_DEVEL": "/* #undef NVSHMEM_DEVEL */",
    "#cmakedefine NVSHMEM_TRACE": "/* #undef NVSHMEM_TRACE */",
    "#cmakedefine NVSHMEM_DEFAULT_PMI2": "/* #undef NVSHMEM_DEFAULT_PMI2 */",
    "#cmakedefine NVSHMEM_DEFAULT_PMIX": "/* #undef NVSHMEM_DEFAULT_PMIX */",
    "#cmakedefine NVSHMEM_DEFAULT_UCX": "/* #undef NVSHMEM_DEFAULT_UCX */",
    "#cmakedefine NVSHMEM_DISABLE_COLL_POLL": "#define NVSHMEM_DISABLE_COLL_POLL",
    "#cmakedefine NVSHMEM_GPU_COLL_USE_LDST": "/* #undef NVSHMEM_GPU_COLL_USE_LDST */",
    "#cmakedefine NVSHMEM_IBDEVX_SUPPORT": "/* #undef NVSHMEM_IBDEVX_SUPPORT */",
    "#cmakedefine NVSHMEM_IBRC_SUPPORT": "",  # "#define NVSHMEM_IBRC_SUPPORT",
    "#cmakedefine NVSHMEM_LIBFABRIC_SUPPORT": "/* #undef NVSHMEM_LIBFABRIC_SUPPORT */",
    "#cmakedefine NVSHMEM_MPI_SUPPORT": "/* #undef NVSHMEM_MPI_SUPPORT */",
    "#cmakedefine NVSHMEM_NVTX": "#define NVSHMEM_NVTX",
    "#cmakedefine NVSHMEM_PMIX_SUPPORT": "/* #undef NVSHMEM_PMIX_SUPPORT */",
    "#cmakedefine NVSHMEM_SHMEM_SUPPORT": "/* #undef NVSHMEM_SHMEM_SUPPORT */",
    "#cmakedefine NVSHMEM_TEST_STATIC_LIB": "/* #undef NVSHMEM_TEST_STATIC_LIB */",
    "#cmakedefine NVSHMEM_TIMEOUT_DEVICE_POLLING": "/* #undef NVSHMEM_TIMEOUT_DEVICE_POLLING */",
    "#cmakedefine NVSHMEM_UCX_SUPPORT": "/* #undef NVSHMEM_UCX_SUPPORT */",
    "#cmakedefine NVSHMEM_USE_DLMALLOC": "/* #undef NVSHMEM_USE_DLMALLOC */",
    "#cmakedefine NVSHMEM_USE_NCCL": "/* #undef NVSHMEM_USE_NCCL */",
    "#cmakedefine NVSHMEM_USE_GDRCOPY": "/* #undef NVSHMEM_USE_GDRCOPY */",
    "#cmakedefine NVSHMEM_VERBOSE": "/* #undef NVSHMEM_VERBOSE */",
    "#cmakedefine NVSHMEM_BUILD_TESTS": "#define NVSHMEM_BUILD_TESTS",
    "#cmakedefine NVSHMEM_BUILD_EXAMPLES": "#define NVSHMEM_BUILD_EXAMPLES",
    "#cmakedefine NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY": "/* #undef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY */",
    "#cmakedefine NVSHMEM_IBGDA_SUPPORT": "/* #undef NVSHMEM_IBGDA_SUPPORT */",
}

options_substitions_no_inlining = {
    "#cmakedefine NVSHMEM_ENABLE_ALL_DEVICE_INLINING": "/* #undef NVSHMEM_ENABLE_ALL_DEVICE_INLINING */",
}

options_substitions_no_inlining.update(options_substitions_common)

options_substitions_inlining = {
    "#cmakedefine NVSHMEM_ENABLE_ALL_DEVICE_INLINING": "#define NVSHMEM_ENABLE_ALL_DEVICE_INLINING",
}

options_substitions_inlining.update(options_substitions_common)

expand_template(
    name = "nvshmem_build_options_h",
    out = "src/include/non_abi/nvshmem_build_options.h",
    substitutions =
        select({
            ":enable_all_device_inlining": options_substitions_inlining,
            "//conditions:default": options_substitions_no_inlining,
        }),
    template = "src/include/non_abi/nvshmem_build_options.h.in",
)

# This additional header allows us to determine the configured NVSHMEM version
# without including the rest of NVSHMEM.
write_file(
    name = "nvshmem_config_header",
    out = "nvshmem_config.h",
    content = [
        "#define TF_NVSHMEM_VERSION \"{}\"".format(NVSHMEM_MAJOR),
    ],
)
