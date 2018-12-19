"""loads the aws library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

# NOTE: version updates here should also update the major, minor, and patch variables declared in 
# the  copts field of the //third_party/aws:aws target

def repo():
    third_party_http_archive(
        name = "aws",
        urls = [
            "https://mirror.bazel.build/github.com/aws/aws-sdk-cpp/archive/1.5.8.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/1.5.8.tar.gz",
        ],
        sha256 = "89905075fe50aa13e0337ff905c2e8c1ce9caf77a3504484a7cda39179120ffc",
        strip_prefix = "aws-sdk-cpp-1.5.8",
        build_file = "//third_party/aws:BUILD.bazel",
    )
