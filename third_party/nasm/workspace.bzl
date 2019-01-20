"""loads the nasm library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "nasm",
        urls = [
            "https://mirror.bazel.build/www.nasm.us/pub/nasm/releasebuilds/2.13.03/nasm-2.13.03.tar.bz2",
            "http://pkgs.fedoraproject.org/repo/pkgs/nasm/nasm-2.13.03.tar.bz2/sha512/d7a6b4cee8dfd603d8d4c976e5287b5cc542fa0b466ff989b743276a6e28114e64289bf02a7819eca63142a5278aa6eed57773007e5f589e15768e6456a8919d/nasm-2.13.03.tar.bz2",
            "http://www.nasm.us/pub/nasm/releasebuilds/2.13.03/nasm-2.13.03.tar.bz2",
        ],
        sha256 = "63ec86477ad3f0f6292325fd89e1d93aea2e2fd490070863f17d48f7cd387011",
        strip_prefix = "nasm-2.13.03",
        build_file = "//third_party/nasm:BUILD.bazel",
        system_build_file = "//third_party/nasm:BUILD.system",
    )
