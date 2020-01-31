"""loads the aws library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

# NOTE: version updates here should also update the major, minor, and patch variables declared in
# the  copts field of the //third_party/aws:aws target

def repo():
    third_party_http_archive(
        name = "aws",
        urls = [
            "https://mirror.bazel.build/github.com/aws/aws-sdk-cpp/archive/1.7.226.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/1.7.226.tar.gz",
        ],
        sha256 = "3a6eff15ee73a1a73c4c16ef2582eaef8647821750dab6d5cd0f137103b5c488",
        strip_prefix = "aws-sdk-cpp-1.7.226",
        build_file = "//third_party/aws:BUILD.bazel",
    )
 
    third_party_http_archive( 
        name = "aws-c-common",
        urls = [
            "http://mirror.tensorflow.org/github.com/awslabs/aws-c-common/archive/v0.4.20.tar.gz",
            "https://github.com/awslabs/aws-c-common/archive/v0.4.20.tar.gz"
        ],
        sha256 = "b0a86df4731fb5de00c5caaf95450ca26a1c0405919aee39927a9455bc5a6b05",
        strip_prefix = "aws-c-common-0.4.20",
        build_file = "//third_party/aws:aws-c-common.bazel",
    )
      
    third_party_http_archive( 
        name = "aws-c-event-stream",
        urls = [
            "https://mirror.tensorflow.org/github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
            "https://github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
        ],
        sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
        strip_prefix = "aws-c-event-stream-0.1.4",
        build_file = "//third_party/aws:aws-c-event-stream.bazel",
    )
      
    third_party_http_archive( 
        name = "aws-checksums",
        urls = [
            "https://mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
            "https://github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
        ],
        sha256 = "6e6bed6f75cf54006b6bafb01b3b96df19605572131a2260fddaf0e87949ced0",
        strip_prefix = "aws-checksums-0.1.5",
        build_file = "//third_party/aws:aws-checksums.bazel",
    )
  
