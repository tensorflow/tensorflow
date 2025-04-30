"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_features//:deps.bzl", "bazel_features_deps")

# Import third party config rules.
load("@bazel_skylib//lib:versions.bzl", "versions")

# Import external repository rules.
load("@bazel_tools//tools/build_defs/repo:java.bzl", "java_import_external")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Import third party repository rules. See go/tfbr-thirdparty.
load("//third_party/absl:workspace.bzl", absl = "repo")
load("//third_party/benchmark:workspace.bzl", benchmark = "repo")
load("//third_party/clang_toolchain:cc_configure_clang.bzl", "cc_download_clang_toolchain")
load("//third_party/ducc:workspace.bzl", ducc = "repo")
load("//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("//third_party/gemmlowp:workspace.bzl", gemmlowp = "repo")
load("//third_party/git:git_configure.bzl", "git_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/gpus:sycl_configure.bzl", "sycl_configure")
load("//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("//third_party/implib_so:workspace.bzl", implib_so = "repo")
load("//third_party/llvm:setup.bzl", "llvm_setup")
load("//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party/nvshmem:workspace.bzl", nvshmem = "repo")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/py/ml_dtypes:workspace.bzl", ml_dtypes = "repo")
load("//third_party/pybind11_abseil:workspace.bzl", pybind11_abseil = "repo")
load("//third_party/pybind11_bazel:workspace.bzl", pybind11_bazel = "repo")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//third_party/tensorrt:workspace.bzl", tensorrt = "repo")
load("//tools/def_file_filter:def_file_filter_configure.bzl", "def_file_filter_configure")
load("//tools/toolchains:cpus/aarch64/aarch64_compiler_configure.bzl", "aarch64_compiler_configure")
load("//tools/toolchains:cpus/arm/arm_compiler_configure.bzl", "arm_compiler_configure")
load("//tools/toolchains/clang6:repo.bzl", "clang6_configure")
load("//tools/toolchains/embedded/arm-linux:arm_linux_toolchain_configure.bzl", "arm_linux_toolchain_configure")
load("//tools/toolchains/remote:configure.bzl", "remote_execution_configure")
load("//tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")

def _initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    absl()
    benchmark()
    ducc()
    eigen3()
    farmhash()
    gemmlowp()
    highwayhash()
    hwloc()
    implib_so()
    ml_dtypes()
    nasm()
    nvshmem()
    pybind11_abseil()
    pybind11_bazel()
    tensorrt()

    # copybara: tsl vendor

# Toolchains & platforms required by Tensorflow to build.
def _tf_toolchains():
    native.register_execution_platforms("@local_execution_config_platform//:platform")
    native.register_toolchains("@local_execution_config_python//:py_toolchain")

    # Loads all external repos to configure RBE builds.
    initialize_rbe_configs()

    # Note that we check the minimum bazel version in WORKSPACE.
    clang6_configure(name = "local_config_clang6")
    cc_download_clang_toolchain(name = "local_config_download_clang")
    tensorrt_configure(name = "local_config_tensorrt")
    git_configure(name = "local_config_git")
    syslibs_configure(name = "local_config_syslibs")
    python_configure(name = "local_config_python")
    rocm_configure(name = "local_config_rocm")
    sycl_configure(name = "local_config_sycl")
    remote_execution_configure(name = "local_config_remote_execution")

    # For windows bazel build
    # TODO: Remove def file filter when TensorFlow can export symbols properly on Windows.
    def_file_filter_configure(name = "local_config_def_file_filter")

    # Point //external/local_config_arm_compiler to //external/arm_compiler
    arm_compiler_configure(
        name = "local_config_arm_compiler",
        build_file = "//tools/toolchains/cpus/arm:template.BUILD",
        remote_config_repo_arm = "../arm_compiler",
        remote_config_repo_aarch64 = "../aarch64_compiler",
    )

    # Load aarch64 toolchain
    aarch64_compiler_configure()

    # TFLite crossbuild toolchain for embeddeds Linux
    arm_linux_toolchain_configure(
        name = "local_config_embedded_arm",
        build_file = "//tools/toolchains/embedded/arm-linux:template.BUILD",
        aarch64_repo = "../aarch64_linux_toolchain",
        armhf_repo = "../armhf_linux_toolchain",
    )

# Define all external repositories required by TensorFlow
def _tf_repositories():
    """All external dependencies for TF builds."""

    # To update any of the dependencies below:
    # a) update URL and strip_prefix to the new git commit hash
    # b) get the sha256 hash of the commit by running:
    #    curl -L <url> | sha256sum
    # and update the sha256 with the result.

    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "1832b8998252529d73e585b545c3f1a12a69ddd136ba9072ea9f717e17ce452b",
        strip_prefix = "XNNPACK-8a2f5f441833b80806b58b5d704ec8335634182c",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/8a2f5f441833b80806b58b5d704ec8335634182c.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)

    tf_http_archive(
        name = "FXdiv",
        sha256 = "3d7b0e9c4c658a84376a1086126be02f9b7f753caa95e009d9ac38d11da444db",
        strip_prefix = "FXdiv-63058eff77e11aa15bf531df5dd34395ec3017c8",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip"),
    )

    tf_http_archive(
        name = "pthreadpool",
        sha256 = "6416b3ca51c60fbcd4776685ef27e4858760ecf689d113adf074a0749f977ff7",
        strip_prefix = "pthreadpool-290ee6fff0c36614702d6b297c148e3fa08e056a",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/290ee6fff0c36614702d6b297c148e3fa08e056a.zip"),
    )

    tf_http_archive(
        name = "cpuinfo",
        sha256 = "52e0ffd7998d8cb3a927d8a6e1145763744d866d2be09c4eccea27fc157b6bb0",
        strip_prefix = "cpuinfo-cebb0933058d7f181c979afd50601dc311e1bf8c",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/cebb0933058d7f181c979afd50601dc311e1bf8c.zip"),
    )

    tf_http_archive(
        name = "mkl_dnn_v1",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        sha256 = "dc2b9bc851cd8d5a6c4622f7dc215bdb6b32349962875f8bf55cceed45a4c449",
        strip_prefix = "oneDNN-2.7.1",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v2.7.1.tar.gz"),
    )

    tf_http_archive(
        name = "onednn",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        patch_file = ["//third_party/mkl_dnn:setting_init.patch"],
        sha256 = "8356aa9befde4d4ff93f1b016ac4310730b2de0cc0b8c6c7ce306690bc0d7b43",
        strip_prefix = "oneDNN-3.5",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.5.tar.gz"),
    )

    tf_http_archive(
        name = "mkl_dnn_acl_compatible",
        build_file = "//third_party/mkl_dnn:mkldnn_acl.BUILD",
        patch_file = [
            "//third_party/mkl_dnn:onednn_acl_threadpool_default_max.patch",
        ],
        sha256 = "5792cbc07764c6e25c459ff68efb5cfcd7f4a0ba66dca6a4a2c681cd7a644596",
        strip_prefix = "oneDNN-3.7",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.zip"),
    )

    tf_http_archive(
        name = "compute_library",
        patch_file = [
            "//third_party/compute_library:compute_library.patch",
            "//third_party/compute_library:exclude_omp_scheduler.patch",
            "//third_party/compute_library:include_string.patch",
        ],
        sha256 = "8273f68cd0bb17e9231a11a6618d245eb6d623884ae681c00e7a4eabca2dad42",
        strip_prefix = "ComputeLibrary-24.12",
        urls = tf_mirror_urls("https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v24.12.tar.gz"),
    )

    tf_http_archive(
        name = "arm_compiler",
        build_file = "//:arm_compiler.BUILD",
        sha256 = "b9e7d50ffd9996ed18900d041d362c99473b382c0ae049b2fce3290632d2656f",
        strip_prefix = "rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf/",
        urls = tf_mirror_urls("https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz"),
    )

    tf_http_archive(
        # This is the latest `aarch64-none-linux-gnu` compiler provided by ARM
        # See https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads
        # The archive contains GCC version 9.2.1
        name = "aarch64_compiler",
        build_file = "//:arm_compiler.BUILD",
        sha256 = "8dfe681531f0bd04fb9c53cf3c0a3368c616aa85d48938eebe2b516376e06a66",
        strip_prefix = "gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"),
    )

    tf_http_archive(
        name = "aarch64_linux_toolchain",
        build_file = "//tools/toolchains/embedded/arm-linux:aarch64-linux-toolchain.BUILD",
        sha256 = "50cdef6c5baddaa00f60502cc8b59cc11065306ae575ad2f51e412a9b2a90364",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"),
    )

    tf_http_archive(
        name = "armhf_linux_toolchain",
        build_file = "//tools/toolchains/embedded/arm-linux:armhf-linux-toolchain.BUILD",
        sha256 = "3f76650b1d048036473b16b647b8fd005ffccd1a2869c10994967e0e49f26ac2",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-linux-gnueabihf",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-linux-gnueabihf.tar.xz"),
    )

    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        system_build_file = "//third_party/systemlibs:re2.BUILD",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_google_crc32c",
        sha256 = "6b3b1d861bb8307658b2407bc7a4c59e566855ef5368a60b35c893551e4788e9",
        build_file = "@com_github_googlecloudplatform_google_cloud_cpp//bazel:crc32c.BUILD",
        strip_prefix = "crc32c-1.0.6",
        urls = tf_mirror_urls("https://github.com/google/crc32c/archive/1.0.6.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_googlecloudplatform_tensorflow_gcp_tools",
        sha256 = "5e9ebe17eaa2895eb7f77fefbf52deeda7c4b63f5a616916b823eb74f3a0c542",
        strip_prefix = "tensorflow-gcp-tools-2643d8caeba6ca2a6a0b46bb123953cb95b7e7d5",
        urls = tf_mirror_urls("https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools/archive/2643d8caeba6ca2a6a0b46bb123953cb95b7e7d5.tar.gz"),
    )

    tf_http_archive(
        name = "six_archive",
        build_file = "//third_party:six.BUILD",
        sha256 = "1e61c37477a1626458e36f7b1d82aa5c9b094fa4802892072e49de9c60c4c926",
        strip_prefix = "six-1.16.0",
        system_build_file = "//third_party/systemlibs:six.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/source/s/six/six-1.16.0.tar.gz"),
    )

    filegroup_external(
        name = "astunparse_license",
        licenses = ["notice"],  # PSFL
        sha256_urls = {
            "92fc0e4f4fa9460558eedf3412b988d433a2dcbb3a9c45402a145a4fab8a6ac6": tf_mirror_urls("https://raw.githubusercontent.com/simonpercivall/astunparse/v1.6.2/LICENSE"),
        },
    )

    tf_http_archive(
        name = "absl_py",
        sha256 = "a7c51b2a0aa6357a9cbb2d9437e8cd787200531867dc02565218930b6a32166e",
        strip_prefix = "abseil-py-1.0.0",
        system_build_file = "//third_party/systemlibs:absl_py.BUILD",
        system_link_files = {
            "//third_party/systemlibs:absl_py.absl.BUILD": "absl/BUILD",
            "//third_party/systemlibs:absl_py.absl.flags.BUILD": "absl/flags/BUILD",
            "//third_party/systemlibs:absl_py.absl.testing.BUILD": "absl/testing/BUILD",
            "//third_party/systemlibs:absl_py.absl.logging.BUILD": "absl/logging/BUILD",
        },
        urls = tf_mirror_urls("https://github.com/abseil/abseil-py/archive/refs/tags/v1.0.0.tar.gz"),
    )

    filegroup_external(
        name = "org_python_license",
        licenses = ["notice"],  # Python 2.0
        sha256_urls = {
            "e76cacdf0bdd265ff074ccca03671c33126f597f39d0ed97bc3e5673d9170cf6": tf_mirror_urls("https://docs.python.org/2.7/_sources/license.rst.txt"),
        },
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = ["//third_party/protobuf:protobuf.patch"],
        sha256 = "f66073dee0bc159157b0bd7f502d7d1ee0bc76b3c1eac9836927511bdc4b3fc1",
        strip_prefix = "protobuf-3.21.9",
        system_build_file = "//third_party/systemlibs:protobuf.BUILD",
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
            "//third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
        },
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/v3.21.9.zip"),
    )

    tf_http_archive(
        name = "com_google_googletest",
        # Use the commit on 2025/3/21:
        # https://github.com/google/googletest/commit/2ae29b52fdff88c52fef655fa0d245fc514ca35b
        sha256 = "21a3a4021fd5e3127c90547234e2126d24f23571fedefa0d9370bf706a870fba",
        strip_prefix = "googletest-2ae29b52fdff88c52fef655fa0d245fc514ca35b",
        # Patch googletest to:
        #   - avoid dependencies on @fuchsia_sdk,
        #   - refer to re2 as @com_googlesource_code_re2,
        #   - refer to abseil as @com_google_absl.
        #
        # To update the patch, run:
        # $ cd ~
        # $ mkdir -p github
        # $ cd github
        # $ git clone https://github.com/google/googletest.git
        # $ cd googletest
        # $ git checkout 2ae29b52fdff88c52fef655fa0d245fc514ca35b
        # ... make local changes to googletest ...
        # $ git diff > <client-root>/third_party/tensorflow/third_party/googletest/googletest.patch
        #
        # The patch path is relative to third_party/tensorflow.
        patch_file = ["//third_party/googletest:googletest.patch"],
        urls = tf_mirror_urls("https://github.com/google/googletest/archive/2ae29b52fdff88c52fef655fa0d245fc514ca35b.zip"),
    )

    tf_http_archive(
        name = "com_google_fuzztest",
        sha256 = "d922940bde8904937b9e13298f06b1d59388ab4a965122860358b00535438f63",
        strip_prefix = "fuzztest-e576caaece16bd1f8dcd196736743c36474f3c16",
        urls = tf_mirror_urls("https://github.com/google/fuzztest/archive/e576caaece16bd1f8dcd196736743c36474f3c16.zip"),
    )

    tf_http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        urls = tf_mirror_urls("https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"),
    )

    tf_http_archive(
        name = "curl",
        build_file = "//third_party:curl.BUILD",
        sha256 = "264537d90e58d2b09dddc50944baf3c38e7089151c8986715e2aaeaaf2b8118f",
        strip_prefix = "curl-8.11.0",
        system_build_file = "//third_party/systemlibs:curl.BUILD",
        urls = tf_mirror_urls("https://curl.se/download/curl-8.11.0.tar.gz"),
    )

    # WARNING: make sure ncteisen@ and vpai@ are cc-ed on any CL to change the below rule
    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
        strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
        system_build_file = "//third_party/systemlibs:grpc.BUILD",
        patch_file = [
            "//third_party/grpc:generate_cc_env_fix.patch",
            "//third_party/grpc:register_go_toolchain.patch",
        ],
        system_link_files = {
            "//third_party/systemlibs:BUILD.bazel": "bazel/BUILD.bazel",
            "//third_party/systemlibs:grpc.BUILD": "src/compiler/BUILD",
            "//third_party/systemlibs:grpc.bazel.grpc_deps.bzl": "bazel/grpc_deps.bzl",
            "//third_party/systemlibs:grpc.bazel.grpc_extra_deps.bzl": "bazel/grpc_extra_deps.bzl",
            "//third_party/systemlibs:grpc.bazel.cc_grpc_library.bzl": "bazel/cc_grpc_library.bzl",
            "//third_party/systemlibs:grpc.bazel.generate_cc.bzl": "bazel/generate_cc.bzl",
            "//third_party/systemlibs:grpc.bazel.protobuf.bzl": "bazel/protobuf.bzl",
        },
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz"),
    )

    llvm_setup(name = "llvm-project")

    # Intel openMP that is part of LLVM sources.
    tf_http_archive(
        name = "llvm_openmp",
        build_file = "//third_party/llvm_openmp:BUILD.bazel",
        patch_file = ["//third_party/llvm_openmp:openmp_switch_default_patch.patch"],
        sha256 = "d19f728c8e04fb1e94566c8d76aef50ec926cd2f95ef3bf1e0a5de4909b28b44",
        strip_prefix = "openmp-10.0.1.src",
        urls = tf_mirror_urls("https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz"),
    )

    tf_http_archive(
        name = "jsoncpp_git",
        sha256 = "f409856e5920c18d0c2fb85276e24ee607d2a09b5e7d5f0a371368903c275da2",
        strip_prefix = "jsoncpp-1.9.5",
        system_build_file = "//third_party/systemlibs:jsoncpp.BUILD",
        urls = tf_mirror_urls("https://github.com/open-source-parsers/jsoncpp/archive/1.9.5.tar.gz"),
    )

    tf_http_archive(
        name = "zlib",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
        strip_prefix = "zlib-1.3.1",
        system_build_file = "//third_party/systemlibs:zlib.BUILD",
        urls = tf_mirror_urls("https://zlib.net/zlib-1.3.1.tar.gz"),
    )

    tf_http_archive(
        name = "snappy",
        build_file = "//third_party:snappy.BUILD",
        sha256 = "2e458b7017cd58dcf1469ab315389e85e7f445bd035188f2983f81fb19ecfb29",
        strip_prefix = "snappy-984b191f0fefdeb17050b42a90b7625999c13b8d",
        system_build_file = "//third_party/systemlibs:snappy.BUILD",
        urls = tf_mirror_urls("https://github.com/google/snappy/archive/984b191f0fefdeb17050b42a90b7625999c13b8d.tar.gz"),
    )

    tf_http_archive(
        name = "nccl_archive",
        build_file = "//third_party:nccl/archive.BUILD",
        patch_file = ["//third_party/nccl:archive.patch"],
        sha256 = "7b154ad1f8ccafa795ed6696507d402b1b4ccac944c5fceb7f4e29b19a39cc47",
        strip_prefix = "nccl-2.25.1-1",
        urls = tf_mirror_urls("https://github.com/nvidia/nccl/archive/v2.25.1-1.tar.gz"),
    )

    tf_http_archive(
        name = "nvtx_archive",
        build_file = "//third_party:nvtx/BUILD.bazel",
        sha256 = "e4438f921fb88a564b0b92791c1c1fdd0f388901213e6a31fdd0dc3803fb9764",
        strip_prefix = "NVTX-bf31d7859ab3130cbf1ef77c33d18d0ebb8c8d08/c/include",
        urls = tf_mirror_urls("https://github.com/NVIDIA/NVTX/archive/bf31d7859ab3130cbf1ef77c33d18d0ebb8c8d08.tar.gz"),
    )

    java_import_external(
        name = "junit",
        jar_sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
            "https://repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
            "https://maven.ibiblio.org/maven2/junit/junit/4.12/junit-4.12.jar",
        ],
        licenses = ["reciprocal"],  # Common Public License Version 1.0
        testonly_ = True,
        deps = ["@org_hamcrest_core"],
    )

    java_import_external(
        name = "org_hamcrest_core",
        jar_sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
            "https://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
            "https://maven.ibiblio.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
        ],
        licenses = ["notice"],  # New BSD License
        testonly_ = True,
    )

    java_import_external(
        name = "com_google_testing_compile",
        jar_sha256 = "edc180fdcd9f740240da1a7a45673f46f59c5578d8cd3fbc912161f74b5aebb8",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
            "https://repo1.maven.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
        ],
        licenses = ["notice"],  # New BSD License
        testonly_ = True,
        deps = ["@com_google_guava", "@com_google_truth"],
    )

    java_import_external(
        name = "com_google_truth",
        jar_sha256 = "032eddc69652b0a1f8d458f999b4a9534965c646b8b5de0eba48ee69407051df",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar",
            "https://repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
        testonly_ = True,
        deps = ["@com_google_guava"],
    )

    java_import_external(
        name = "org_checkerframework_qual",
        jar_sha256 = "d261fde25d590f6b69db7721d469ac1b0a19a17ccaaaa751c31f0d8b8260b894",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/org/checkerframework/checker-qual/2.10.0/checker-qual-2.10.0.jar",
            "https://repo1.maven.org/maven2/org/checkerframework/checker-qual/2.10.0/checker-qual-2.10.0.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
    )

    java_import_external(
        name = "com_squareup_javapoet",
        jar_sha256 = "5bb5abdfe4366c15c0da3332c57d484e238bd48260d6f9d6acf2b08fdde1efea",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/com/squareup/javapoet/1.9.0/javapoet-1.9.0.jar",
            "https://repo1.maven.org/maven2/com/squareup/javapoet/1.9.0/javapoet-1.9.0.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
    )

    tf_http_archive(
        name = "cython",
        build_file = "//third_party:cython.BUILD",
        sha256 = "0c2eae8a4ceab7955be1e11a4ddc5dcc3aa06ce22ad594262f1555b9d10667f0",
        strip_prefix = "cython-3.0.3",
        system_build_file = "//third_party/systemlibs:cython.BUILD",
        urls = tf_mirror_urls("https://github.com/cython/cython/archive/3.0.3.tar.gz"),
    )

    tf_http_archive(
        name = "build_bazel_rules_android",
        sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
        strip_prefix = "rules_android-0.1.1",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip"),
    )

    # Apple and Swift rules.
    # https://github.com/bazelbuild/rules_apple/releases
    tf_http_archive(
        name = "build_bazel_rules_apple",
        sha256 = "b4df908ec14868369021182ab191dbd1f40830c9b300650d5dc389e0b9266c8d",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_apple/releases/download/3.5.1/rules_apple.3.5.1.tar.gz"),
    )

    # https://github.com/bazelbuild/rules_swift/releases
    tf_http_archive(
        name = "build_bazel_rules_swift",
        sha256 = "bb01097c7c7a1407f8ad49a1a0b1960655cf823c26ad2782d0b7d15b323838e2",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_swift/releases/download/1.18.0/rules_swift.1.18.0.tar.gz"),
    )

    # https://github.com/bazelbuild/apple_support/releases
    tf_http_archive(
        name = "build_bazel_apple_support",
        sha256 = "d71b02d6df0500f43279e22400db6680024c1c439115c57a9a82e9effe199d7b",
        urls = tf_mirror_urls("https://github.com/bazelbuild/apple_support/releases/download/1.18.1/apple_support.1.18.1.tar.gz"),
    )

    # https://github.com/apple/swift-protobuf/releases
    tf_http_archive(
        name = "com_github_apple_swift_swift_protobuf",
        strip_prefix = "swift-protobuf-1.19.0/",
        sha256 = "f057930b9dbd17abeaaceaa45e9f8b3e87188c05211710563d2311b9edf490aa",
        urls = tf_mirror_urls("https://github.com/apple/swift-protobuf/archive/1.19.0.tar.gz"),
    )

    # https://github.com/google/xctestrunner/releases
    tf_http_archive(
        name = "xctestrunner",
        strip_prefix = "xctestrunner-0.2.15",
        sha256 = "b789cf18037c8c28d17365f14925f83b93b1f7dabcabb80333ae4331cf0bcb2f",
        urls = tf_mirror_urls("https://github.com/google/xctestrunner/archive/refs/tags/0.2.15.tar.gz"),
    )

    tf_http_archive(
        name = "nlohmann_json_lib",
        build_file = "//third_party:nlohmann_json.BUILD",
        sha256 = "5daca6ca216495edf89d167f808d1d03c4a4d929cef7da5e10f135ae1540c7e4",
        strip_prefix = "json-3.10.5",
        urls = tf_mirror_urls("https://github.com/nlohmann/json/archive/v3.10.5.tar.gz"),
    )

    tf_http_archive(
        name = "pybind11",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11/archive/v2.13.4.tar.gz"),
        sha256 = "efc901aa0aab439a3fea6efeaf930b5a349fb06394bf845c64ce15a9cf8f0240",
        strip_prefix = "pybind11-2.13.4",
        build_file = "//third_party:pybind11.BUILD",
        system_build_file = "//third_party/systemlibs:pybind11.BUILD",
    )

    # Dependencies required by grpc
    #   - pin rules_go to a newer version so it's compatible with Bazel 6.0
    #   - patch upb so that it's compatible with Bazel 6.0, the latest version of upb doesn't work with the old grpc version.
    tf_http_archive(
        name = "io_bazel_rules_go",
        sha256 = "16e9fca53ed6bd4ff4ad76facc9b7b651a89db1689a2877d6fd7b82aa824e366",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_go/releases/download/v0.34.0/rules_go-v0.34.0.zip"),
    )

    tf_http_archive(
        name = "upb",
        sha256 = "61d0417abd60e65ed589c9deee7c124fe76a4106831f6ad39464e1525cef1454",
        strip_prefix = "upb-9effcbcb27f0a665f9f345030188c0b291e32482",
        patch_file = ["//third_party/grpc:upb_platform_fix.patch"],
        urls = tf_mirror_urls("https://github.com/protocolbuffers/upb/archive/9effcbcb27f0a665f9f345030188c0b291e32482.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_glog_glog",
        sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
        strip_prefix = "glog-0.4.0",
        urls = tf_mirror_urls("https://github.com/google/glog/archive/refs/tags/v0.4.0.tar.gz"),
    )

    tf_http_archive(
        name = "spirv_headers",
        sha256 = "11d835c60297b26532c05c3f3b581ba7a2787b5ae7399e94f72c392169216f11",
        strip_prefix = "SPIRV-Headers-b73e168ca5e123dcf3dea8a34b19a5130f421ae1",
        urls = tf_mirror_urls("https://github.com/KhronosGroup/SPIRV-Headers/archive/b73e168ca5e123dcf3dea8a34b19a5130f421ae1.tar.gz"),
    )

    tf_http_archive(
        name = "spirv_llvm_translator",
        sha256 = "d499769f4fd1e0ce9d4dbd3622ee7e3e641b5623dcdf811521e3e7c0bdb1e6c2",
        strip_prefix = "SPIRV-LLVM-Translator-dad1f0eaab8047a4f73c50ed5f3d1694b78aae97",
        build_file = "//third_party/spirv_llvm_translator:spirv_llvm_translator.BUILD",
        patch_file = ["//third_party/spirv_llvm_translator:spirv_llvm_translator.patch"],
        urls = tf_mirror_urls("https://github.com/KhronosGroup/SPIRV-LLVM-Translator/archive/dad1f0eaab8047a4f73c50ed5f3d1694b78aae97.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_nelhage_rules_boost",
        urls = tf_mirror_urls("https://github.com/nelhage/rules_boost/archive/5160325dbdc8c9e499f9d9917d913f35f1785d52.zip"),
        strip_prefix = "rules_boost-5160325dbdc8c9e499f9d9917d913f35f1785d52",
        sha256 = "feb4b1294684c79df7c1e08f1aec5da0da52021e33db59c88edbe86b4d1a017a",
    )

# buildifier: disable=unnamed-macro
def workspace():
    # Check the bazel version before executing any repository rules, in case
    # those rules rely on the version we require here.
    versions.check("1.0.0")

    # Initialize toolchains and platforms.
    _tf_toolchains()

    # Import third party repositories according to go/tfbr-thirdparty.
    _initialize_third_party()

    # Import all other repositories. This should happen before initializing
    # any external repositories, because those come with their own
    # dependencies. Those recursive dependencies will only be imported if they
    # don't already exist (at least if the external repository macros were
    # written according to common practice to query native.existing_rule()).
    _tf_repositories()

    bazel_features_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tsl_workspace2 = workspace
