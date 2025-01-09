"""Provides the repo macro to import ml_dtypes.

ml_dtypes provides machine-learning-specific data-types like bfloat16,
float8 varieties, and int4.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    ML_DTYPES_COMMIT = "215c9f02a121e6286662b2efd30546c71054d5e5"
    ML_DTYPES_SHA256 = "4a03237ef6345e1467a33d126176b9c6a7539b0f60a34b344f39b3c9e8b82438"
    tf_http_archive(
        name = "ml_dtypes",
        build_file = "//third_party/py/ml_dtypes:ml_dtypes.BUILD",
        link_files = {
            "//third_party/py/ml_dtypes:ml_dtypes.tests.BUILD": "tests/BUILD.bazel",
            "//third_party/py/ml_dtypes:LICENSE": "LICENSE",
        },
        sha256 = ML_DTYPES_SHA256,
        strip_prefix = "ml_dtypes-{commit}/ml_dtypes".format(commit = ML_DTYPES_COMMIT),
        urls = tf_mirror_urls("https://github.com/jax-ml/ml_dtypes/archive/{commit}/ml_dtypes-{commit}.tar.gz".format(commit = ML_DTYPES_COMMIT)),
    )
