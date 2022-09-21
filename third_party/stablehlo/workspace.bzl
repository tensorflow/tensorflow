"""Provides the repository macro to import StableHLO."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    STABLEHLO_COMMIT = "31b8d00a7b37716e9e0d1a780301b52d4b78abf1"
    STABLEHLO_SHA256 = "d1af1d898a0d2664aec7f122e88525f73699761c82b772a064b96ef6dcb07bd1"
    # LINT.ThenChange(Google-internal path)

    tf_http_archive(
        name = "stablehlo",
        sha256 = STABLEHLO_SHA256,
        strip_prefix = "stablehlo-{commit}".format(commit = STABLEHLO_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/stablehlo/archive/{commit}.zip".format(commit = STABLEHLO_COMMIT)),
        build_file = "//third_party/stablehlo:BUILD",
        patch_file = [
            "//third_party/stablehlo:temporary.patch",  # Cherry-picks and temporary reverts. Do not remove even if temporary.patch is empty.
        ],
    )
