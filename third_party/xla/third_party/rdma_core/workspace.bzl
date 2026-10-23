"""Loads rdma-core (libibverbs) headers for ROCm/MORI hermetic builds.

MORI's RDMA transport includes <infiniband/verbs.h>. Under the hermetic ROCm
toolchain (--config=rocm_ci) the compiler uses a hermetic sysroot that does not
contain the system rdma-core headers, so we vendor just the public libibverbs
headers here. The shared library is still resolved at link time from the host
(-libverbs), which works because the hermetic toolchain links with
CppLink=local.

To update to a new release: change _RDMA_CORE_VERSION, clear _RDMA_CORE_SHA256
(set to ""), run any bazel build that touches @rdma_core, and paste back the
sha256 Bazel prints.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Matches the rdma-core (libibverbs) major version shipped with the ROCm image.
_RDMA_CORE_VERSION = "48.0"
_RDMA_CORE_SHA256 = "e78abaa3ed16771bc6c40b538fe27082429363b86b6ce47636bb57a59238b2be"

def repo():
    """Imports rdma-core (libibverbs) headers."""
    tf_http_archive(
        name = "rdma_core",
        build_file = "//third_party/rdma_core:rdma_core.BUILD",
        sha256 = _RDMA_CORE_SHA256,
        strip_prefix = "rdma-core-{}".format(_RDMA_CORE_VERSION),
        urls = tf_mirror_urls(
            "https://github.com/linux-rdma/rdma-core/archive/refs/tags/v{}.tar.gz".format(
                _RDMA_CORE_VERSION,
            ),
        ),
    )
