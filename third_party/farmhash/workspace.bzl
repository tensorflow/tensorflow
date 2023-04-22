"""Provides the repository macro to import farmhash."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports farmhash."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    FARMHASH_COMMIT = "816a4ae622e964763ca0862d9dbd19324a1eaf45"
    FARMHASH_SHA256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/farmhash.cmake)

    tf_http_archive(
        name = "farmhash_archive",
        build_file = "//third_party/farmhash:farmhash.BUILD",
        sha256 = FARMHASH_SHA256,
        strip_prefix = "farmhash-{commit}".format(commit = FARMHASH_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/{commit}.tar.gz".format(commit = FARMHASH_COMMIT),
            "https://github.com/google/farmhash/archive/{commit}.tar.gz".format(commit = FARMHASH_COMMIT),
        ],
    )

    tf_http_archive(
        name = "farmhash_gpu_archive",
        build_file = "//third_party/farmhash:farmhash_gpu.BUILD",
        patch_file = "//third_party/farmhash:farmhash_support_cuda.patch",
        sha256 = FARMHASH_SHA256,
        strip_prefix = "farmhash-{commit}".format(commit = FARMHASH_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/{commit}.tar.gz".format(commit = FARMHASH_COMMIT),
            "https://github.com/google/farmhash/archive/{commit}.tar.gz".format(commit = FARMHASH_COMMIT),
        ],
    )
