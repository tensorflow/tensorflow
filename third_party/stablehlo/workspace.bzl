"""Provides the repository macro to import StableHLO."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    STABLEHLO_COMMIT = "2813676bbbc8ce714babeef12791abdfc7f8ffba"
    STABLEHLO_SHA256 = "5adc88d00e43a048c3b91a8f1bcc13220ce43174b1817e0697c2939f371d1cb9"
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
