load("@bazel_skylib//rules:write_file.bzl", "write_file")

# Top-level BUILD overlay for @roc_mori. Symlinked over the extracted
# tarball's root BUILD.bazel by tf_http_archive (see workspace.bzl).
#
# Only exposes headers-only libraries plus a small config header. The actual
# compiled sub-libraries live in sub-package BUILD overlays (e.g.
# src/shmem/BUILD.bazel) so they sit next to the sources they describe.
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

# ----------------------------------------------------------------------------
# System-library shims required by MORI.
#
# These four libraries (libibverbs, libdrm/libdrm_amdgpu, libnuma, libpci) are
# not bundled in hermetic ROCm (rocm_dist/lib/) — they live under /usr/lib on
# Debian/Ubuntu (rdma-core, libdrm-dev, libnuma-dev, libpci-dev). Each shim is
# a one-line cc_library with the appropriate -l linkopt; headers come from
# the toolchain's default system include path.
#
# They are hosted here (not under xla/third_party/ as separate repos) because
# right now only @roc_mori references them. If a future ROCm release bundles
# them into the hermetic distribution, these shims become trivial to swap out
# (just point dependents at @local_config_rocm//rocm:<name> instead).
# ----------------------------------------------------------------------------

# libibverbs (rdma-core). Used by transport/rdma/ providers and shmem fabric.
# Headers are vendored hermetically via @rdma_core (see //third_party/rdma_core);
# the shared library itself is resolved at link time from the host (-libverbs),
# which works because the hermetic ROCm toolchain links with CppLink=local.
cc_library(
    name = "ibverbs",
    linkopts = ["-libverbs"],
    deps = ["@rdma_core//:verbs_headers"],
)

# libdrm + libdrm_amdgpu. Pulled in transitively by libhsakmt.a (amdgpu_*,
# drmClose). Only listed by mori_application, since :hsakmt itself is a
# static archive whose link-time deps must be resolved by the consumer.
cc_library(
    name = "libdrm",
    linkopts = [
        "-ldrm",
        "-ldrm_amdgpu",
    ],
)

# libnuma. Pulled in transitively by libhsakmt.a (numa_*, mbind).
cc_library(
    name = "libnuma",
    linkopts = ["-lnuma"],
)

# libpci (pciutils). Used by topology/pci.cpp (pci_alloc / pci_scan_bus /...).
# Headers come from the ROCm CI image via @system_libpci (the hermetic sysroot
# has none); libpci.so is linked from the image (-lpci, resolved CppLink=local).
cc_library(
    name = "libpci",
    linkopts = ["-lpci"],
    deps = ["@system_libpci//:pci_headers"],
)

# Public shmem headers (everything under include/mori/shmem). Sources live
# in src/shmem, so the header glob is hosted here, not in src/shmem/BUILD.bazel.
cc_library(
    name = "mori_shmem_headers",
    hdrs = glob([
        "include/mori/shmem/**/*.hpp",
        "include/mori/shmem/**/*.h",
        "include/mori/shmem/**/*.hh",
        "include/mori/shmem/**/*.inc",
    ]),
    # Using `defines` (not `local_defines`) is intentional: it propagates
    # through CcInfo to all dependents.
    defines = ["MORI_MULTITHREAD_SUPPORT"],
    includes = ["include"],
    deps = [
        ":mori_application_headers",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

# Headers transitively #include'd by mori_shmem's .cpp files. Globs every
# header outside include/mori/shmem so additions to upstream subtrees
# (application/, utils/, core/, ops/, collective/, ...) are picked up
# automatically. Multiple extensions are matched because the MORI tree mixes
# .hpp (most C++ headers), .h (vendor descriptors under core/transport),
# and the occasional .hh / .inc.
cc_library(
    name = "mori_application_headers",
    hdrs = glob(
        [
            "include/mori/**/*.hpp",
            "include/mori/**/*.h",
            "include/mori/**/*.hh",
            "include/mori/**/*.inc",
        ],
        exclude = [
            # shmem/ is exposed by :mori_shmem_headers above.
            "include/mori/shmem/**",
        ],
    ),
    #strip_include_prefix = "include",
    includes = ["include"],
    deps = [
        ":ibverbs",  # infiniband/verbs.h
        "@local_config_rocm//rocm:rocm_headers",  # hip/, hsa/, hsakmt/
        "@spdlog",  # mori_log.hpp -> spdlog
    ],
)

# Small compile-time identifier some XLA call sites reference.
write_file(
    name = "mori_config_h",
    out = "mori_config.h",
    content = [
        "#ifndef THIRD_PARTY_ROCM_MORI_CONFIG_H_",
        "#define THIRD_PARTY_ROCM_MORI_CONFIG_H_",
        "constexpr static char XLA_ROCM_MORI_VERSION[] = \"bazel\";",
        "#endif  // THIRD_PARTY_ROCM_MORI_CONFIG_H_",
        "",
    ],
)

cc_library(
    name = "mori_config",
    hdrs = ["mori_config.h"],
    include_prefix = "third_party",
)
