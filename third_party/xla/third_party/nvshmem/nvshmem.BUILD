# NVSHMEM

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@bazel_skylib//rules:write_file.bzl", "write_file")

options_substitions = {
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
    "#cmakedefine NVSHMEM_IBRC_SUPPORT": "#define NVSHMEM_IBRC_SUPPORT",
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
    "#cmakedefine NVSHMEM_ENABLE_ALL_DEVICE_INLINING": "/* #undef NVSHMEM_ENABLE_ALL_DEVICE_INLINING */",
}

expand_template(
    name = "nvshmem_build_options_h",
    out = "src/include/non_abi/nvshmem_build_options.h",
    substitutions = options_substitions,
    template = "src/include/non_abi/nvshmem_build_options.h.in",
)

NVSHMEM_MAJOR = 3

version_substitions = {
    "@PROJECT_VERSION_MAJOR@": str(NVSHMEM_MAJOR),
    "@PROJECT_VERSION_MINOR@": "1",
    "@PROJECT_VERSION_PATCH@": "7",
    "@PROJECT_VERSION_TWEAK@": "0",
    "@TRANSPORT_VERSION_MAJOR@": "3",
    "@TRANSPORT_VERSION_MINOR@": "0",
    "@TRANSPORT_VERSION_PATCH@": "0",
    "@BOOTSTRAP_VERSION_MAJOR@": "3",
    "@BOOTSTRAP_VERSION_MINOR@": "0",
    "@BOOTSTRAP_VERSION_PATCH@": "0",
    "@INTERLIB_VERSION_MAJOR@": "3",
    "@INTERLIB_VERSION_MINOR@": "0",
    "@INTERLIB_VERSION_PATCH@": "0",
    "@INFO_BUILD_VARS@": "",
}

expand_template(
    name = "nvshmem_version_h",
    out = "src/include/non_abi/nvshmem_version.h",
    substitutions = version_substitions,
    template = "src/include/non_abi/nvshmem_version.h.in",
)

cc_library(
    name = "nvshmem_lib",
    hdrs = glob([
        "src/include/**",
    ]) + [
        ":nvshmem_build_options_h",
        ":nvshmem_version_h",
    ],
    include_prefix = "third_party/nvshmem",
    includes = ["src/include"],
    strip_include_prefix = "src/include",
    visibility = ["//visibility:public"],
    deps = [
        "@xla//xla/tsl/cuda:nvshmem_stub",
    ],
)

# This additional header allows us to determine the configured NVSHMEM version
# without including the rest of NVSHMEM.
write_file(
    name = "nvshmem_config_header",
    out = "nvshmem_config.h",
    content = [
        "constexpr static char XLA_NVSHMEM_VERSION[] = \"{}\";".format(NVSHMEM_MAJOR),
    ],
)

cc_library(
    name = "nvshmem_config",
    hdrs = ["nvshmem_config.h"],
    include_prefix = "third_party/nvshmem",
    visibility = ["//visibility:public"],
)
