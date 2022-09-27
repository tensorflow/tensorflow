"""Provides the repository macro to import StableHLO."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    STABLEHLO_COMMIT = "55814280317cc6db499bf2137750bcfa46ca452e"
    STABLEHLO_SHA256 = "f7757b8781f5749c85ad2c5acfe6e49fa11de41982ecd900f664187f1e9cbab4"
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
