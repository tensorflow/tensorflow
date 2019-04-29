"""Loads the Arm Compute Library library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "acl",
        strip_prefix = "ComputeLibrary-refs_heads_branches_arm_compute_19_05",
        sha256 = "c520b7887487ff4179efb0f63c10ac95def9dbf2",
        urls = [
            "https://git.mlplatform.org/ml/ComputeLibrary.git/snapshot/ComputeLibrary-refs/heads/branches/arm_compute_19_05.tar.gz",
        ],
        build_file = "//third_party/acl:BUILD.bazel",
    )
