"""loads the nasm library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "nasm",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/www.nasm.us/pub/nasm/releasebuilds/2.14.02/nasm-2.14.02.tar.bz2",
            "http://pkgs.fedoraproject.org/repo/pkgs/nasm/nasm-2.14.02.tar.bz2/sha512/d7a6b4cee8dfd603d8d4c976e5287b5cc542fa0b466ff989b743276a6e28114e64289bf02a7819eca63142a5278aa6eed57773007e5f589e15768e6456a8919d/nasm-2.14.02.tar.bz2",
            "http://www.nasm.us/pub/nasm/releasebuilds/2.14.02/nasm-2.14.02.tar.bz2",
        ],
        sha256 = "34fd26c70a277a9fdd54cb5ecf389badedaf48047b269d1008fbc819b24e80bc",
        strip_prefix = "nasm-2.14.02",
        build_file = "//third_party/nasm:nasm.BUILD",
        system_build_file = "//third_party/nasm:BUILD.system",
        link_files = {"//third_party/nasm:config.h": "config/config.h"},
    )
