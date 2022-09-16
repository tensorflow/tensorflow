"""Provides the repository macro to import StableHLO."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    STABLEHLO_COMMIT = "9ca259d5092e9cf2c1fa0788a470df6a4fc95f0a"
    STABLEHLO_SHA256 = "4079eaeea79e952e7265131b28514dba28dc39632146fb63ff1d869c04eb0781"
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
