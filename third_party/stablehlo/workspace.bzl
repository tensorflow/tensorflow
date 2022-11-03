"""Provides the repository macro to import StableHLO."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    STABLEHLO_COMMIT = "f70b37a3aa578541f8f83d4d0230b86ea5bc77f6"
    STABLEHLO_SHA256 = "cda192b3e94388f16f65ea83bb0ea9b3cabbcd9cf1e3695e1740b5d9188f162d"
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
