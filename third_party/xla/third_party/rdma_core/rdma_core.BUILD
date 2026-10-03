"""Header-only vendoring of rdma-core's libibverbs public API.

We expose ONLY headers and link against the host's libibverbs.so via -libverbs
(set on @roc_mori//:ibverbs). libibverbs is a thin shim over the kernel uverbs
ABI that dlopen's provider plugins at runtime, so it must come from the host and
must not be built hermetically. The headers, however, are a stable ABI and safe
to pin.

The include layout mirrors what rdma-core's CMake `publish_headers` produces
(see libibverbs/CMakeLists.txt and kernel-headers/CMakeLists.txt upstream):
  libibverbs/*.h            -> infiniband/*.h
  kernel-headers/rdma/*.h   -> rdma/*.h
  kernel-headers/rdma/ib_user_ioctl_verbs.h -> infiniband/ib_user_ioctl_verbs.h
Bazel's strip_include_prefix/include_prefix do this remapping natively, so no
genrule or CMake step is required.
"""

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # OpenIB BSD / MIT (see COPYING.md)

exports_files(["COPYING.md"])

# libibverbs/*.h -> #include <infiniband/*.h>
# Mirrors: publish_headers(infiniband arch.h opcode.h sa-kern-abi.h sa.h
#          verbs.h verbs_api.h tm_types.h)
cc_library(
    name = "verbs_headers",
    hdrs = [
        "libibverbs/arch.h",
        "libibverbs/opcode.h",
        "libibverbs/sa.h",
        "libibverbs/sa-kern-abi.h",
        "libibverbs/tm_types.h",
        "libibverbs/verbs.h",
        "libibverbs/verbs_api.h",
    ],
    include_prefix = "infiniband",
    strip_include_prefix = "libibverbs",
    deps = [
        # verbs_api.h -> <rdma/ib_user_ioctl_verbs.h>
        ":rdma_uapi_headers",
        # rdma-core also republishes this UAPI header under infiniband/.
        ":ib_user_ioctl_verbs_infiniband",
        # MORI's RDMA transport also includes <infiniband/mlx5dv.h>.
        ":mlx5_provider_headers",
    ],
)

# mlx5 direct-verbs provider headers. Mirrors:
# publish_headers(infiniband ../../kernel-headers/rdma/mlx5_user_ioctl_verbs.h
#                 mlx5_api.h mlx5dv.h)  [providers/mlx5/CMakeLists.txt]
# Include chain: mlx5dv.h -> mlx5_api.h -> <infiniband/mlx5_user_ioctl_verbs.h>.
cc_library(
    name = "mlx5_provider_headers",
    hdrs = [
        "providers/mlx5/mlx5_api.h",
        "providers/mlx5/mlx5dv.h",
    ],
    include_prefix = "infiniband",
    strip_include_prefix = "providers/mlx5",
    deps = [":mlx5_user_ioctl_verbs_infiniband"],
)

# kernel-headers/rdma/mlx5_user_ioctl_verbs.h republished under infiniband/.
cc_library(
    name = "mlx5_user_ioctl_verbs_infiniband",
    hdrs = ["kernel-headers/rdma/mlx5_user_ioctl_verbs.h"],
    include_prefix = "infiniband",
    strip_include_prefix = "kernel-headers/rdma",
)

# kernel-headers/rdma/** -> #include <rdma/**> (UAPI headers).
cc_library(
    name = "rdma_uapi_headers",
    hdrs = glob([
        "kernel-headers/rdma/*.h",
        "kernel-headers/rdma/hfi/*.h",
    ]),
    strip_include_prefix = "kernel-headers",
)

# publish_headers(infiniband rdma/ib_user_ioctl_verbs.h): the one UAPI header
# rdma-core also exposes under the infiniband/ prefix.
cc_library(
    name = "ib_user_ioctl_verbs_infiniband",
    hdrs = ["kernel-headers/rdma/ib_user_ioctl_verbs.h"],
    include_prefix = "infiniband",
    strip_include_prefix = "kernel-headers/rdma",
)
