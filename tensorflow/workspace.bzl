# TensorFlow external dependencies that can be loaded in WORKSPACE files.

load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//third_party:nccl/nccl_configure.bzl", "nccl_configure")
load("//third_party/mkl:build_defs.bzl", "mkl_repository")
load("//third_party/git:git_configure.bzl", "git_configure")
load("//third_party/py:python_configure.bzl", "python_configure")

load("//third_party/sycl:sycl_configure.bzl", "sycl_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("//third_party/toolchains/clang6:repo.bzl", "clang6_configure")
load("//third_party/toolchains/cpus/arm:arm_compiler_configure.bzl", "arm_compiler_configure")
load("//third_party:repo.bzl", "tf_http_archive")
load("//third_party/clang_toolchain:cc_configure_clang.bzl", "cc_download_clang_toolchain")
load("@io_bazel_rules_closure//closure/private:java_import_external.bzl", "java_import_external")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")
load(
    "//tensorflow/tools/def_file_filter:def_file_filter_configure.bzl",
    "def_file_filter_configure",
)
load("//third_party/aws:workspace.bzl", aws = "repo")
load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("//third_party/icu:workspace.bzl", icu = "repo")
load("//third_party/jpeg:workspace.bzl", jpeg = "repo")
load("//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/keras_applications_archive:workspace.bzl", keras_applications = "repo")

def initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    aws()
    flatbuffers()
    highwayhash()
    icu()
    keras_applications()
    kissfft()
    jpeg()
    nasm()

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

# If TensorFlow is linked as a submodule.
# path_prefix is no longer used.
# tf_repo_name is thought to be under consideration.
def tf_workspace(path_prefix = "", tf_repo_name = ""):
    # Note that we check the minimum bazel version in WORKSPACE.
    clang6_configure(name = "local_config_clang6")
    cc_download_clang_toolchain(name = "local_config_download_clang")
    cuda_configure(name = "local_config_cuda")
    tensorrt_configure(name = "local_config_tensorrt")
    nccl_configure(name = "local_config_nccl")
    git_configure(name = "local_config_git")
    sycl_configure(name = "local_config_sycl")
    syslibs_configure(name = "local_config_syslibs")
    python_configure(name = "local_config_python")
    rocm_configure(name = "local_config_rocm")

    initialize_third_party()

    # For windows bazel build
    # TODO: Remove def file filter when TensorFlow can export symbols properly on Windows.
    def_file_filter_configure(name = "local_config_def_file_filter")

    # Point //external/local_config_arm_compiler to //external/arm_compiler
    arm_compiler_configure(
        name = "local_config_arm_compiler",
        build_file = clean_dep("//third_party/toolchains/cpus/arm:BUILD"),
        remote_config_repo = "../arm_compiler",
    )

    mkl_repository(
        name = "mkl_linux",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "f00dc3b142a5be399bdeebd7e7ea369545a35d4fb84c86f98b6b048d72685295",
        strip_prefix = "mklml_lnx_2019.0.1.20180928",
        urls = [
            "https://mirror.bazel.build/github.com/intel/mkl-dnn/releases/download/v0.17-rc/mklml_lnx_2019.0.1.20180928.tgz",
            "https://github.com/intel/mkl-dnn/releases/download/v0.17-rc/mklml_lnx_2019.0.1.20180928.tgz",
        ],
    )
    mkl_repository(
        name = "mkl_windows",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "efef90b7b9613fab10f44c8ac4ff28db613a112c64ed94826d7e44df09c44b0b",
        strip_prefix = "mklml_win_2019.0.1.20180928",
        urls = [
            "https://mirror.bazel.build/github.com/intel/mkl-dnn/releases/download/v0.17-rc/mklml_win_2019.0.1.20180928.zip",
            "https://github.com/intel/mkl-dnn/releases/download/v0.17-rc/mklml_win_2019.0.1.20180928.zip",
        ],
    )
    mkl_repository(
        name = "mkl_darwin",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "83f02938a0c095274db7b8b7b694157abafa3837c5cbaef740440d466c86a477",
        strip_prefix = "mklml_mac_2019.0.1.20180928",
        urls = [
            "https://mirror.bazel.build/github.com/intel/mkl-dnn/releases/download/v0.17-rc/mklml_mac_2019.0.1.20180928.tgz",
            "https://github.com/intel/mkl-dnn/releases/download/v0.17-rc/mklml_mac_2019.0.1.20180928.tgz",
        ],
    )

    if path_prefix:
        print("path_prefix was specified to tf_workspace but is no longer used " +
              "and will be removed in the future.")

    tf_http_archive(
        name = "mkl_dnn",
        build_file = clean_dep("//third_party/mkl_dnn:mkldnn.BUILD"),
        sha256 = "b100f57af4a2b59a3a37a1ba38f77b644d2107d758a1a7f4e51310063cd21e73",
        strip_prefix = "mkl-dnn-733fc908874c71a5285043931a1cf80aa923165c",
        urls = [
            "https://mirror.bazel.build/github.com/intel/mkl-dnn/archive/733fc908874c71a5285043931a1cf80aa923165c.tar.gz",
            "https://github.com/intel/mkl-dnn/archive/733fc908874c71a5285043931a1cf80aa923165c.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_absl",
        build_file = clean_dep("//third_party:com_google_absl.BUILD"),
        sha256 = "3cf6132129ba87f0781c383bfaf381b7174b5818e81fffcc5d04bb451154f0f2",
        strip_prefix = "abseil-cpp-f95179062eb65ce40895cc76f1398cce25394369",
        urls = [
            "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/f95179062eb65ce40895cc76f1398cce25394369.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/f95179062eb65ce40895cc76f1398cce25394369.tar.gz",
        ],
    )

    tf_http_archive(
        name = "eigen_archive",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        sha256 = "6e505fa8bf8d234d0338679b390cb89f850d870214c751b01a5db7f647e4d438",
        strip_prefix = "eigen-eigen-135398e50bed",
        urls = [
            "https://mirror.bazel.build/bitbucket.org/eigen/eigen/get/135398e50bed.tar.gz",
            "https://bitbucket.org/eigen/eigen/get/135398e50bed.tar.gz",
        ],
    )

    tf_http_archive(
        name = "arm_compiler",
        build_file = clean_dep("//:arm_compiler.BUILD"),
        sha256 = "970285762565c7890c6c087d262b0a18286e7d0384f13a37786d8521773bc969",
        strip_prefix = "tools-0e906ebc527eab1cdbf7adabff5b474da9562e9f/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf",
        urls = [
            "https://mirror.bazel.build/github.com/raspberrypi/tools/archive/0e906ebc527eab1cdbf7adabff5b474da9562e9f.tar.gz",
            # Please uncomment me, when the next upgrade happens. Then
            # remove the whitelist entry in third_party/repo.bzl.
            # "https://github.com/raspberrypi/tools/archive/0e906ebc527eab1cdbf7adabff5b474da9562e9f.tar.gz",
        ],
    )

    tf_http_archive(
        name = "libxsmm_archive",
        build_file = clean_dep("//third_party:libxsmm.BUILD"),
        sha256 = "cd8532021352b4a0290d209f7f9bfd7c2411e08286a893af3577a43457287bfa",
        strip_prefix = "libxsmm-1.9",
        urls = [
            "https://mirror.bazel.build/github.com/hfp/libxsmm/archive/1.9.tar.gz",
            "https://github.com/hfp/libxsmm/archive/1.9.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "803c7811146edeef8f91064de37c6f19136ff01a2a8cdb3230e940b2fd9f07fe",
        strip_prefix = "re2-2018-07-01",
        system_build_file = clean_dep("//third_party/systemlibs:re2.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/google/re2/archive/2018-07-01.tar.gz",
            "https://github.com/google/re2/archive/2018-07-01.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_github_googlecloudplatform_google_cloud_cpp",
        sha256 = "3ade2072e6588ff56c0434abe6c63aa5f3f2d56be15a299bafc7e9cdf0a12c17",
        strip_prefix = "google-cloud-cpp-0.3.0",
        system_build_file = clean_dep("//third_party/systemlibs:google_cloud_cpp.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:google_cloud_cpp.google.cloud.bigtable.BUILD": "google/cloud/bigtable/BUILD",
        },
        urls = [
            "https://mirror.bazel.build/github.com/GoogleCloudPlatform/google-cloud-cpp/archive/v0.3.0.tar.gz",
            "https://github.com/GoogleCloudPlatform/google-cloud-cpp/archive/v0.3.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_github_googleapis_googleapis",
        build_file = clean_dep("//third_party:googleapis.BUILD"),
        sha256 = "824870d87a176f26bcef663e92051f532fac756d1a06b404055dc078425f4378",
        strip_prefix = "googleapis-f81082ea1e2f85c43649bee26e0d9871d4b41cdb",
        system_build_file = clean_dep("//third_party/systemlibs:googleapis.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/googleapis/googleapis/archive/f81082ea1e2f85c43649bee26e0d9871d4b41cdb.zip",
            "https://github.com/googleapis/googleapis/archive/f81082ea1e2f85c43649bee26e0d9871d4b41cdb.zip",
        ],
    )

    tf_http_archive(
        name = "gemmlowp",
        sha256 = "b87faa7294dfcc5d678f22a59d2c01ca94ea1e2a3b488c38a95a67889ed0a658",
        strip_prefix = "gemmlowp-38ebac7b059e84692f53e5938f97a9943c120d98",
        urls = [
            "https://mirror.bazel.build/github.com/google/gemmlowp/archive/38ebac7b059e84692f53e5938f97a9943c120d98.zip",
            "https://github.com/google/gemmlowp/archive/38ebac7b059e84692f53e5938f97a9943c120d98.zip",
        ],
    )

    tf_http_archive(
        name = "farmhash_archive",
        build_file = clean_dep("//third_party:farmhash.BUILD"),
        sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",
        strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
        urls = [
            "https://mirror.bazel.build/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
            "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
        ],
    )

    tf_http_archive(
        name = "png_archive",
        build_file = clean_dep("//third_party:png.BUILD"),
        patch_file = clean_dep("//third_party:png_fix_rpi.patch"),
        sha256 = "e45ce5f68b1d80e2cb9a2b601605b374bdf51e1798ef1c2c2bd62131dfcf9eef",
        strip_prefix = "libpng-1.6.34",
        system_build_file = clean_dep("//third_party/systemlibs:png.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/glennrp/libpng/archive/v1.6.34.tar.gz",
            "https://github.com/glennrp/libpng/archive/v1.6.34.tar.gz",
        ],
    )

    tf_http_archive(
        name = "org_sqlite",
        build_file = clean_dep("//third_party:sqlite.BUILD"),
        sha256 = "ad68c1216c3a474cf360c7581a4001e952515b3649342100f2d7ca7c8e313da6",
        strip_prefix = "sqlite-amalgamation-3240000",
        system_build_file = clean_dep("//third_party/systemlibs:sqlite.BUILD"),
        urls = [
            "https://mirror.bazel.build/www.sqlite.org/2018/sqlite-amalgamation-3240000.zip",
            "https://www.sqlite.org/2018/sqlite-amalgamation-3240000.zip",
        ],
    )

    tf_http_archive(
        name = "gif_archive",
        build_file = clean_dep("//third_party:gif.BUILD"),
        sha256 = "34a7377ba834397db019e8eb122e551a49c98f49df75ec3fcc92b9a794a4f6d1",
        strip_prefix = "giflib-5.1.4",
        system_build_file = clean_dep("//third_party/systemlibs:gif.BUILD"),
        urls = [
            "https://mirror.bazel.build/ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
            "http://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "six_archive",
        build_file = clean_dep("//third_party:six.BUILD"),
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        system_build_file = clean_dep("//third_party/systemlibs:six.BUILD"),
        urls = [
            "https://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "astor_archive",
        build_file = clean_dep("//third_party:astor.BUILD"),
        sha256 = "ff6d2e2962d834acb125cc4dcc80c54a8c17c253f4cc9d9c43b5102a560bb75d",
        strip_prefix = "astor-0.6.2",
        system_build_file = clean_dep("//third_party/systemlibs:astor.BUILD"),
        urls = [
            "https://mirror.bazel.build/pypi.python.org/packages/d8/be/c4276b3199ec3feee2a88bc64810fbea8f26d961e0a4cd9c68387a9f35de/astor-0.6.2.tar.gz",
            "https://pypi.python.org/packages/d8/be/c4276b3199ec3feee2a88bc64810fbea8f26d961e0a4cd9c68387a9f35de/astor-0.6.2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "gast_archive",
        build_file = clean_dep("//third_party:gast.BUILD"),
        sha256 = "7068908321ecd2774f145193c4b34a11305bd104b4551b09273dfd1d6a374930",
        strip_prefix = "gast-0.2.0",
        system_build_file = clean_dep("//third_party/systemlibs:gast.BUILD"),
        urls = [
            "https://mirror.bazel.build/pypi.python.org/packages/5c/78/ff794fcae2ce8aa6323e789d1f8b3b7765f601e7702726f430e814822b96/gast-0.2.0.tar.gz",
            "https://pypi.python.org/packages/5c/78/ff794fcae2ce8aa6323e789d1f8b3b7765f601e7702726f430e814822b96/gast-0.2.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "termcolor_archive",
        build_file = clean_dep("//third_party:termcolor.BUILD"),
        sha256 = "1d6d69ce66211143803fbc56652b41d73b4a400a2891d7bf7a1cdf4c02de613b",
        strip_prefix = "termcolor-1.1.0",
        system_build_file = clean_dep("//third_party/systemlibs:termcolor.BUILD"),
        urls = [
            "https://mirror.bazel.build/pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz",
            "https://pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "absl_py",
        sha256 = "95160f778a62c7a60ddeadc7bf2d83f85a23a27359814aca12cf949e896fa82c",
        strip_prefix = "abseil-py-pypi-v0.2.2",
        system_build_file = clean_dep("//third_party/systemlibs:absl_py.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:absl_py.absl.flags.BUILD": "absl/flags/BUILD",
            "//third_party/systemlibs:absl_py.absl.testing.BUILD": "absl/testing/BUILD",
        },
        urls = [
            "https://mirror.bazel.build/github.com/abseil/abseil-py/archive/pypi-v0.2.2.tar.gz",
            "https://github.com/abseil/abseil-py/archive/pypi-v0.2.2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "org_python_pypi_backports_weakref",
        build_file = clean_dep("//third_party:backports_weakref.BUILD"),
        sha256 = "8813bf712a66b3d8b85dc289e1104ed220f1878cf981e2fe756dfaabe9a82892",
        strip_prefix = "backports.weakref-1.0rc1/src",
        urls = [
            "https://mirror.bazel.build/pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
            "https://pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
        ],
    )

    filegroup_external(
        name = "org_python_license",
        licenses = ["notice"],  # Python 2.0
        sha256_urls = {
            "7ca8f169368827781684f7f20876d17b4415bbc5cb28baa4ca4652f0dda05e9f": [
                "https://mirror.bazel.build/docs.python.org/2.7/_sources/license.rst.txt",
                "https://docs.python.org/2.7/_sources/license.rst.txt",
            ],
        },
    )

    PROTOBUF_URLS = [
        "https://mirror.bazel.build/github.com/google/protobuf/archive/v3.6.1.1.tar.gz",
        "https://github.com/google/protobuf/archive/v3.6.1.1.tar.gz",
    ]
    PROTOBUF_SHA256 = "1ade182f91f0fa6c6116195def5d22270e01b9d03fe91319e4c6215022d0d24b"
    PROTOBUF_STRIP_PREFIX = "protobuf-3.6.1.1"

    tf_http_archive(
        name = "protobuf_archive",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        urls = PROTOBUF_URLS,
    )

    # We need to import the protobuf library under the names com_google_protobuf
    # and com_google_protobuf_cc to enable proto_library support in bazel.
    # Unfortunately there is no way to alias http_archives at the moment.
    tf_http_archive(
        name = "com_google_protobuf",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        urls = PROTOBUF_URLS,
    )

    tf_http_archive(
        name = "com_google_protobuf_cc",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        urls = PROTOBUF_URLS,
    )

    tf_http_archive(
        name = "nsync",
        sha256 = "692f9b30e219f71a6371b98edd39cef3cbda35ac3abc4cd99ce19db430a5591a",
        strip_prefix = "nsync-1.20.1",
        system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/google/nsync/archive/1.20.1.tar.gz",
            "https://github.com/google/nsync/archive/1.20.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_googletest",
        sha256 = "353ab86e35cea1cd386115279cf4b16695bbf21b897bfbf2721cf4cb5f64ade8",
        strip_prefix = "googletest-997d343dd680e541ef96ce71ee54a91daf2577a0",
        urls = [
            "https://mirror.bazel.build/github.com/google/googletest/archive/997d343dd680e541ef96ce71ee54a91daf2577a0.zip",
            "https://github.com/google/googletest/archive/997d343dd680e541ef96ce71ee54a91daf2577a0.zip",
        ],
    )

    tf_http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "ae27cdbcd6a2f935baa78e4f21f675649271634c092b1be01469440495609d0e",
        strip_prefix = "gflags-2.2.1",
        urls = [
            "https://mirror.bazel.build/github.com/gflags/gflags/archive/v2.2.1.tar.gz",
            "https://github.com/gflags/gflags/archive/v2.2.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "pcre",
        build_file = clean_dep("//third_party:pcre.BUILD"),
        sha256 = "69acbc2fbdefb955d42a4c606dfde800c2885711d2979e356c0636efde9ec3b5",
        strip_prefix = "pcre-8.42",
        system_build_file = clean_dep("//third_party/systemlibs:pcre.BUILD"),
        urls = [
            "https://mirror.bazel.build/ftp.exim.org/pub/pcre/pcre-8.42.tar.gz",
            "http://ftp.exim.org/pub/pcre/pcre-8.42.tar.gz",
        ],
    )

    tf_http_archive(
        name = "swig",
        build_file = clean_dep("//third_party:swig.BUILD"),
        sha256 = "58a475dbbd4a4d7075e5fe86d4e54c9edde39847cdb96a3053d87cb64a23a453",
        strip_prefix = "swig-3.0.8",
        system_build_file = clean_dep("//third_party/systemlibs:swig.BUILD"),
        urls = [
            "https://mirror.bazel.build/ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
            "http://ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
            "http://pilotfiber.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
        ],
    )

    tf_http_archive(
        name = "curl",
        build_file = clean_dep("//third_party:curl.BUILD"),
        sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
        strip_prefix = "curl-7.60.0",
        system_build_file = clean_dep("//third_party/systemlibs:curl.BUILD"),
        urls = [
            "https://mirror.bazel.build/curl.haxx.se/download/curl-7.60.0.tar.gz",
            "https://curl.haxx.se/download/curl-7.60.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "grpc",
        sha256 = "50db9cf2221354485eb7c3bd55a4c27190caef7048a2a1a15fbe60a498f98b44",
        strip_prefix = "grpc-1.13.0",
        system_build_file = clean_dep("//third_party/systemlibs:grpc.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/grpc/grpc/archive/v1.13.0.tar.gz",
            "https://github.com/grpc/grpc/archive/v1.13.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "linenoise",
        build_file = clean_dep("//third_party:linenoise.BUILD"),
        sha256 = "7f51f45887a3d31b4ce4fa5965210a5e64637ceac12720cfce7954d6a2e812f7",
        strip_prefix = "linenoise-c894b9e59f02203dbe4e2be657572cf88c4230c3",
        urls = [
            "https://mirror.bazel.build/github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
            "https://github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
        ],
    )

    # TODO(phawkins): currently, this rule uses an unofficial LLVM mirror.
    # Switch to an official source of snapshots if/when possible.
    tf_http_archive(
        name = "llvm",
        build_file = clean_dep("//third_party/llvm:llvm.autogenerated.BUILD"),
        sha256 = "f4791ba3e166918bca82df34e2f854e8e188d6055888c64cb28743fd43f2d0d7",
        strip_prefix = "llvm-b2a42b2112a511a5077fd747fb21e45349cff08d",
        urls = [
            "https://mirror.bazel.build/github.com/llvm-mirror/llvm/archive/b2a42b2112a511a5077fd747fb21e45349cff08d.tar.gz",
            "https://github.com/llvm-mirror/llvm/archive/b2a42b2112a511a5077fd747fb21e45349cff08d.tar.gz",
        ],
    )

    tf_http_archive(
        name = "lmdb",
        build_file = clean_dep("//third_party:lmdb.BUILD"),
        sha256 = "f3927859882eb608868c8c31586bb7eb84562a40a6bf5cc3e13b6b564641ea28",
        strip_prefix = "lmdb-LMDB_0.9.22/libraries/liblmdb",
        system_build_file = clean_dep("//third_party/systemlibs:lmdb.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
            "https://github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
        ],
    )

    tf_http_archive(
        name = "jsoncpp_git",
        build_file = clean_dep("//third_party:jsoncpp.BUILD"),
        sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
        strip_prefix = "jsoncpp-1.8.4",
        system_build_file = clean_dep("//third_party/systemlibs:jsoncpp.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
            "https://github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "boringssl",
        sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
        strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
        system_build_file = clean_dep("//third_party/systemlibs:boringssl.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
            "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        ],
    )

    tf_http_archive(
        name = "zlib_archive",
        build_file = clean_dep("//third_party:zlib.BUILD"),
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        system_build_file = clean_dep("//third_party/systemlibs:zlib.BUILD"),
        urls = [
            "https://mirror.bazel.build/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    tf_http_archive(
        name = "fft2d",
        build_file = clean_dep("//third_party/fft2d:fft2d.BUILD"),
        sha256 = "52bb637c70b971958ec79c9c8752b1df5ff0218a4db4510e60826e0cb79b5296",
        urls = [
            "https://mirror.bazel.build/www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz",
            "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz",
        ],
    )

    tf_http_archive(
        name = "snappy",
        build_file = clean_dep("//third_party:snappy.BUILD"),
        sha256 = "3dfa02e873ff51a11ee02b9ca391807f0c8ea0529a4924afa645fbf97163f9d4",
        strip_prefix = "snappy-1.1.7",
        system_build_file = clean_dep("//third_party/systemlibs:snappy.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/google/snappy/archive/1.1.7.tar.gz",
            "https://github.com/google/snappy/archive/1.1.7.tar.gz",
        ],
    )

    tf_http_archive(
        name = "nccl_archive",
        build_file = clean_dep("//third_party:nccl/archive.BUILD"),
        sha256 = "19132b5127fa8e02d95a09795866923f04064c8f1e0770b2b42ab551408882a4",
        strip_prefix = "nccl-f93fe9bfd94884cec2ba711897222e0df5569a53",
        urls = [
            "https://mirror.bazel.build/github.com/nvidia/nccl/archive/f93fe9bfd94884cec2ba711897222e0df5569a53.tar.gz",
            "https://github.com/nvidia/nccl/archive/f93fe9bfd94884cec2ba711897222e0df5569a53.tar.gz",
        ],
    )

    tf_http_archive(
        name = "kafka",
        build_file = clean_dep("//third_party:kafka/BUILD"),
        patch_file = clean_dep("//third_party/kafka:config.patch"),
        sha256 = "cc6ebbcd0a826eec1b8ce1f625ffe71b53ef3290f8192b6cae38412a958f4fd3",
        strip_prefix = "librdkafka-0.11.5",
        urls = [
            "https://mirror.bazel.build/github.com/edenhill/librdkafka/archive/v0.11.5.tar.gz",
            "https://github.com/edenhill/librdkafka/archive/v0.11.5.tar.gz",
        ],
    )

    java_import_external(
        name = "junit",
        jar_sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a",
        jar_urls = [
            "https://mirror.bazel.build/repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
            "http://repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
            "http://maven.ibiblio.org/maven2/junit/junit/4.12/junit-4.12.jar",
        ],
        licenses = ["reciprocal"],  # Common Public License Version 1.0
        testonly_ = True,
        deps = ["@org_hamcrest_core"],
    )

    java_import_external(
        name = "org_hamcrest_core",
        jar_sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9",
        jar_urls = [
            "https://mirror.bazel.build/repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
            "http://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
            "http://maven.ibiblio.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
        ],
        licenses = ["notice"],  # New BSD License
        testonly_ = True,
    )

    java_import_external(
        name = "com_google_testing_compile",
        jar_sha256 = "edc180fdcd9f740240da1a7a45673f46f59c5578d8cd3fbc912161f74b5aebb8",
        jar_urls = [
            "http://mirror.bazel.build/repo1.maven.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
            "http://repo1.maven.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
        ],
        licenses = ["notice"],  # New BSD License
        testonly_ = True,
        deps = ["@com_google_guava", "@com_google_truth"],
    )

    java_import_external(
        name = "com_google_truth",
        jar_sha256 = "032eddc69652b0a1f8d458f999b4a9534965c646b8b5de0eba48ee69407051df",
        jar_urls = [
            "http://mirror.bazel.build/repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar",
            "http://repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
        testonly_ = True,
        deps = ["@com_google_guava"],
    )

    java_import_external(
        name = "org_checkerframework_qual",
        jar_sha256 = "a17501717ef7c8dda4dba73ded50c0d7cde440fd721acfeacbf19786ceac1ed6",
        jar_urls = [
            "http://mirror.bazel.build/repo1.maven.org/maven2/org/checkerframework/checker-qual/2.4.0/checker-qual-2.4.0.jar",
            "http://repo1.maven.org/maven2/org/checkerframework/checker-qual/2.4.0/checker-qual-2.4.0.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
    )

    java_import_external(
        name = "com_squareup_javapoet",
        jar_sha256 = "5bb5abdfe4366c15c0da3332c57d484e238bd48260d6f9d6acf2b08fdde1efea",
        jar_urls = [
            "http://mirror.bazel.build/repo1.maven.org/maven2/com/squareup/javapoet/1.9.0/javapoet-1.9.0.jar",
            "http://repo1.maven.org/maven2/com/squareup/javapoet/1.9.0/javapoet-1.9.0.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
    )

    tf_http_archive(
        name = "com_google_pprof",
        build_file = clean_dep("//third_party:pprof.BUILD"),
        sha256 = "e0928ca4aa10ea1e0551e2d7ce4d1d7ea2d84b2abbdef082b0da84268791d0c4",
        strip_prefix = "pprof-c0fb62ec88c411cc91194465e54db2632845b650",
        urls = [
            "https://mirror.bazel.build/github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
            "https://github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
        ],
    )

    tf_http_archive(
        name = "cub_archive",
        build_file = clean_dep("//third_party:cub.BUILD"),
        sha256 = "6bfa06ab52a650ae7ee6963143a0bbc667d6504822cbd9670369b598f18c58c3",
        strip_prefix = "cub-1.8.0",
        urls = [
            "https://mirror.bazel.build/github.com/NVlabs/cub/archive/1.8.0.zip",
            "https://github.com/NVlabs/cub/archive/1.8.0.zip",
        ],
    )

    tf_http_archive(
        name = "cython",
        build_file = clean_dep("//third_party:cython.BUILD"),
        delete = ["BUILD.bazel"],
        sha256 = "bccc9aa050ea02595b2440188813b936eaf345e85fb9692790cecfe095cf91aa",
        strip_prefix = "cython-0.28.4",
        system_build_file = clean_dep("//third_party/systemlibs:cython.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/cython/cython/archive/0.28.4.tar.gz",
            "https://github.com/cython/cython/archive/0.28.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "bazel_toolchains",
        sha256 = "07dfbe80638eb1fe681f7c07e61b34b579c6710c691e49ee90ccdc6e9e75ebbb",
        strip_prefix = "bazel-toolchains-9a111bd82161c1fbe8ed17a593ca1023fd941c70",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/9a111bd82161c1fbe8ed17a593ca1023fd941c70.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/9a111bd82161c1fbe8ed17a593ca1023fd941c70.tar.gz",
        ],
    )

    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = clean_dep("//third_party:arm_neon_2_x86_sse.BUILD"),
        sha256 = "213733991310b904b11b053ac224fee2d4e0179e46b52fe7f8735b8831e04dcc",
        strip_prefix = "ARM_NEON_2_x86_SSE-1200fe90bb174a6224a525ee60148671a786a71f",
        urls = [
            "https://mirror.bazel.build/github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
            "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
        ],
    )

    tf_http_archive(
        name = "double_conversion",
        build_file = clean_dep("//third_party:double_conversion.BUILD"),
        sha256 = "2f7fbffac0d98d201ad0586f686034371a6d152ca67508ab611adc2386ad30de",
        strip_prefix = "double-conversion-3992066a95b823efc8ccc1baf82a1cfc73f6e9b8",
        system_build_file = clean_dep("//third_party/systemlibs:double_conversion.BUILD"),
        urls = [
            "https://mirror.bazel.build/github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
            "https://github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet",
        build_file = clean_dep("//third_party:tflite_mobilenet.BUILD"),
        sha256 = "23f814d1c076bdf03715dfb6cab3713aa4fbdf040fd5448c43196bd2e97a4c1b",
        urls = [
            "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "767057f2837a46d97882734b03428e8dd640b93236052b312b2f0e45613c1cf0",
        urls = [
            "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_ssd_tflite_v1.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_ssd_tflite_v1.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd_quant",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "a809cd290b4d6a2e8a9d5dad076e0bd695b8091974e0eed1052b480b2f21b6dc",
        urls = [
            "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_0.75_quant_2018_06_29.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_0.75_quant_2018_06_29.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd_quant_protobuf",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "09280972c5777f1aa775ef67cb4ac5d5ed21970acd8535aeca62450ef14f0d79",
        strip_prefix = "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18",
        urls = [
            "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz",
            "http://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz",
        ],
    )

    tf_http_archive(
        name = "tflite_conv_actions_frozen",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "d947b38cba389b5e2d0bfc3ea6cc49c784e187b41a071387b3742d1acac7691e",
        urls = [
            "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_smartreply",
        build_file = clean_dep("//third_party:tflite_smartreply.BUILD"),
        sha256 = "8980151b85a87a9c1a3bb1ed4748119e4a85abd3cb5744d83da4d4bd0fbeef7c",
        urls = [
            "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_ovic_testdata",
        build_file = clean_dep("//third_party:tflite_ovic_testdata.BUILD"),
        sha256 = "21288dccc517acee47fa9648d4d3da28bf0fef5381911ed7b4d2ee36366ffa20",
        strip_prefix = "ovic",
        urls = [
            "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/data/ovic_2018_10_23.zip",
            "https://storage.googleapis.com/download.tensorflow.org/data/ovic_2018_10_23.zip",
        ],
    )

    tf_http_archive(
        name = "build_bazel_rules_android",
        sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
        strip_prefix = "rules_android-0.1.1",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
            "https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
        ],
    )

    tf_http_archive(
        name = "tbb",
        build_file = clean_dep("//third_party/ngraph:tbb.BUILD"),
        sha256 = "724686f90bcda78f13b76f297d964008737ccd6399328143c1c0093e73ae6a13",
        strip_prefix = "tbb-tbb_2018",
        urls = [
            "https://mirror.bazel.build/github.com/01org/tbb/archive/tbb_2018.zip",
            "https://github.com/01org/tbb/archive/tbb_2018.zip",
        ],
    )

    tf_http_archive(
        name = "ngraph",
        build_file = clean_dep("//third_party/ngraph:ngraph.BUILD"),
        sha256 = "2b28f9c9f063b96825a96d56d7f7978c9a1c55c9b25175c20dd49a8a77cb0305",
        strip_prefix = "ngraph-0.9.1",
        urls = [
            "https://mirror.bazel.build/github.com/NervanaSystems/ngraph/archive/v0.9.1.tar.gz",
            "https://github.com/NervanaSystems/ngraph/archive/v0.9.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "nlohmann_json_lib",
        build_file = clean_dep("//third_party/ngraph:nlohmann_json.BUILD"),
        sha256 = "9f3549824af3ca7e9707a2503959886362801fb4926b869789d6929098a79e47",
        strip_prefix = "json-3.1.1",
        urls = [
            "https://mirror.bazel.build/github.com/nlohmann/json/archive/v3.1.1.tar.gz",
            "https://github.com/nlohmann/json/archive/v3.1.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "ngraph_tf",
        build_file = clean_dep("//third_party/ngraph:ngraph_tf.BUILD"),
        sha256 = "89accbc702e68a09775f1011a99dd16561038fd1ce59d566d64450176abaae5c",
        strip_prefix = "ngraph-tf-0.7.0",
        urls = [
            "https://mirror.bazel.build/github.com/NervanaSystems/ngraph-tf/archive/v0.7.0.tar.gz",
            "https://github.com/NervanaSystems/ngraph-tf/archive/v0.7.0.tar.gz",
        ],
    )

    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

    # gRPC wants a cares dependency but its contents is not actually
    # important since we have set GRPC_ARES=0 in .bazelrc
    native.bind(
        name = "cares",
        actual = "@grpc//third_party/nanopb:nanopb",
    )

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@grpc//:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@grpc//:grpc_python_plugin",
    )

    native.bind(
        name = "grpc_lib",
        actual = "@grpc//:grpc++",
    )

    native.bind(
        name = "grpc_lib_unsecure",
        actual = "@grpc//:grpc++_unsecure",
    )

    # Needed by gRPC
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    # Needed by gRPC
    native.bind(
        name = "nanopb",
        actual = "@grpc//third_party/nanopb:nanopb",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf",
        actual = "@protobuf_archive//:protobuf",
    )

    # gRPC expects //external:protobuf_clib and //external:protobuf_compiler
    # to point to Protobuf's compiler library.
    native.bind(
        name = "protobuf_clib",
        actual = "@protobuf_archive//:protoc_lib",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf_headers",
        actual = "@protobuf_archive//:protobuf_headers",
    )

    # Needed by Protobuf
    native.bind(
        name = "python_headers",
        actual = clean_dep("//third_party/python_runtime:headers"),
    )

    # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

    # Needed by gRPC
    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )
