"""Provides the repository macro to import TFRT."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports TFRT."""

    # Attention: tools parse and update these lines.
<<<<<<< HEAD
    TFRT_COMMIT = "efcc0f58fec6f2a7f68e29d350bd9e4bedd85740"
    TFRT_SHA256 = "68d23313a99795078f187ed42f0f7aae8e02fc0133a300e4447f511cf0770abb"
=======
    TFRT_COMMIT = "2788d4e9538644eade36b876182fff5d8220a34a"
    TFRT_SHA256 = "2d900059f8feead31c09be6f62c8027efcaba1d8e5acadfccc69983c6d178951"
>>>>>>> upstream/master

    tf_http_archive(
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = tf_mirror_urls("https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT)),
        # A patch file can be provided for atomic commits to both TF and TFRT.
        # The job that bumps the TFRT_COMMIT also resets patch_file to 'None'.
        patch_file = None,
    )
