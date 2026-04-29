"""Provides the repository macro to import rapids_logger."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def tensorflow_repo():
    """Imports rapids_logger."""

    RAPIDS_LOGGER_VERSION = "0.1.1"
    RAPIDS_LOGGER_SHA256 = "9ef22efcc3e00affed254bf12b52c6775050bb55e93e010067c2fcd9620163c9"

    tf_http_archive(
        name = "rapids_logger",
        sha256 = RAPIDS_LOGGER_SHA256,
        strip_prefix = "rapids-logger-{version}".format(version = RAPIDS_LOGGER_VERSION),
        urls = tf_mirror_urls("https://github.com/rapidsai/rapids-logger/archive/refs/tags/v{version}.tar.gz".format(version = RAPIDS_LOGGER_VERSION)),
        build_file = "//third_party/rapids_logger:rapids_logger.BUILD",
        patch_file = ["//third_party/rapids_logger:smoke_test.cc.patch"],
    )

def xla_repo():
    """Imports rapids_logger."""

    RAPIDS_LOGGER_VERSION = "0.2.3"
    RAPIDS_LOGGER_SHA256 = "36578b337993cdbc7f4c52e5d871289628ac408f3a6028ab4e73a64fcdaa9412"

    tf_http_archive(
        name = "rapids_logger",
        sha256 = RAPIDS_LOGGER_SHA256,
        strip_prefix = "rapids-logger-{version}".format(version = RAPIDS_LOGGER_VERSION),
        urls = tf_mirror_urls("https://github.com/rapidsai/rapids-logger/archive/refs/tags/v{version}.tar.gz".format(version = RAPIDS_LOGGER_VERSION)),
        build_file = "//third_party/rapids_logger:rapids_logger.BUILD",
        patch_file = ["//third_party/rapids_logger:smoke_test.cc.patch"],
    )
