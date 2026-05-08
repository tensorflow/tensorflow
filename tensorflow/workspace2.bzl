"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_features//:deps.bzl", "bazel_features_deps")
load("@bazel_skylib//lib:versions.bzl", "versions")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@rules_jvm_external//:defs.bzl", "maven_install")
load("@rules_ml_toolchain//gpu/rocm:hipcc_configure.bzl", "hipcc_configure")
load("@tf_runtime//:dependencies.bzl", "tfrt_dependencies")
load("@xla//third_party/absl:workspace.bzl", absl = "repo")
load("@xla//third_party/benchmark:workspace.bzl", benchmark = "repo")
load("@xla//third_party/clang_toolchain:cc_configure_clang.bzl", "cc_download_clang_toolchain")
load("@xla//third_party/compute_library:workspace.bzl", compute_library = "repo")
load("@xla//third_party/cpuinfo:workspace.bzl", cpuinfo = "repo")
load("@xla//third_party/cudnn_frontend:workspace.bzl", cudnn_frontend = "repo")
load("@xla//third_party/cutlass:workspace.bzl", cutlass = "repo")
load("@xla//third_party/dlpack:workspace.bzl", dlpack = "repo")
load("@xla//third_party/ducc:workspace.bzl", ducc = "repo")
load("@xla//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("@xla//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("@xla//third_party/fmt:workspace.bzl", fmt = "repo")
load("@xla//third_party/FP16:workspace.bzl", FP16 = "repo")
load("@xla//third_party/fxdiv:workspace.bzl", FXdiv = "repo")
load("@xla//third_party/gemmlowp:workspace.bzl", gemmlowp = "repo")
load("@xla//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("@xla//third_party/gpus:sycl_configure.bzl", "sycl_configure")
load("@xla//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("@xla//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("@xla//third_party/implib_so:workspace.bzl", implib_so = "repo")
load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")
load("@xla//third_party/mkl_dnn:workspace.bzl", onednn = "repo")
load("@xla//third_party/nanobind:workspace.bzl", nanobind = "repo")
load("@xla//third_party/nasm:workspace.bzl", nasm = "repo")
load("@xla//third_party/net_zstd:workspace.bzl", net_zstd = "repo")
load("@xla//third_party/nvshmem:workspace.bzl", nvshmem = "repo")
load("@xla//third_party/pthreadpool:workspace.bzl", pthreadpool = "repo")
load("@xla//third_party/pybind11_abseil:workspace.bzl", pybind11_abseil = "repo")
load("@xla//third_party/pybind11_bazel:workspace.bzl", pybind11_bazel = "repo")
load("@xla//third_party/raft:workspace.bzl", raft = "tensorflow_repo")
load("@xla//third_party/rapids_logger:workspace.bzl", rapids_logger = "tensorflow_repo")
load("@xla//third_party/riegeli:workspace.bzl", riegeli = "repo")
load("@xla//third_party/rmm:workspace.bzl", rmm = "tensorflow_repo")
load("@xla//third_party/robin_map:workspace.bzl", robin_map = "repo")
load("@xla//third_party/rocm_device_libs:workspace.bzl", rocm_device_libs = "repo")
load("@xla//third_party/shardy:workspace.bzl", shardy = "repo")
load("@xla//third_party/slinky:workspace.bzl", slinky = "repo")
load("@xla//third_party/spdlog:workspace.bzl", spdlog = "repo")
load("@xla//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("@xla//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("@xla//third_party/tensorrt:workspace.bzl", tensorrt = "repo")
load("@xla//third_party/triton:workspace.bzl", triton = "repo")
load("@xla//third_party/xnnpack:workspace.bzl", xnnpack = "repo")
load("@xla//third_party/xxd:workspace.bzl", xxd = "repo")
load("@xla//tools/def_file_filter:def_file_filter_configure.bzl", "def_file_filter_configure")
load("@xla//tools/toolchains:cpus/aarch64/aarch64_compiler_configure.bzl", "aarch64_compiler_configure")
load("@xla//tools/toolchains:cpus/arm/arm_compiler_configure.bzl", "arm_compiler_configure")
load("@xla//tools/toolchains/clang6:repo.bzl", "clang6_configure")
load("@xla//tools/toolchains/embedded/arm-linux:arm_linux_toolchain_configure.bzl", "arm_linux_toolchain_configure")
load("@xla//tools/toolchains/remote:configure.bzl", "remote_execution_configure")
load("@xla//tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")
load("//third_party:java_repos.bzl", java_repositories = "repo")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/arm_neon_2_x86_sse:workspace.bzl", arm_neon_2_x86_sse = "repo")
load("//third_party/com_google_highway:workspace.bzl", com_google_highway = "repo")
load("//third_party/coremltools:workspace.bzl", coremltools = "repo")
load("//third_party/cython:workspace.bzl", cython = "repo")
load("//third_party/fft2d:workspace.bzl", fft2d = "repo")
load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("//third_party/gif:workspace.bzl", gif = "repo")
load("//third_party/git:git_configure.bzl", "git_configure")
load("//third_party/googleapis:workspace.bzl", googleapis = "repo")
load("//third_party/hexagon:workspace.bzl", hexagon_nn = "repo")
load("//third_party/icu:workspace.bzl", icu = "repo")
load("//third_party/jpeg:workspace.bzl", jpeg = "repo")
load("//third_party/jpegxl:workspace.bzl", jpegxl = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/libprotobuf_mutator:workspace.bzl", libprotobuf_mutator = "repo")
load("//third_party/libwebp:workspace.bzl", libwebp = "repo")
load("//third_party/linenoise:workspace.bzl", linenoise = "repo")
load("//third_party/llvm_openmp:workspace.bzl", llvm_openmp = "repo")
load("//third_party/nccl:workspace.bzl", nccl = "repo")
load("//third_party/nlohmann_json:workspace.bzl", nlohmann_json = "repo")
load("//third_party/nvtx:workspace.bzl", nvtx = "repo")
load("//third_party/opencl_headers:workspace.bzl", opencl_headers = "repo")
load("//third_party/org_brotli:workspace.bzl", org_brotli = "repo")
load("//third_party/png:workspace.bzl", png = "repo")
load("//third_party/pprof:workspace.bzl", pprof = "repo")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/py/absl_py:workspace.bzl", absl_py = "repo")
load("//third_party/py/ml_dtypes:workspace.bzl", ml_dtypes = "repo")
load("//third_party/ruy:workspace.bzl", ruy = "repo")
load("//third_party/six:workspace.bzl", six = "repo")
load("//third_party/skcms:workspace.bzl", skcms = "repo")
load("//third_party/sobol_data:workspace.bzl", sobol_data = "repo")
load("//third_party/sqlite:workspace.bzl", sqlite = "repo")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("//third_party/tf_gcp_tools:workspace.bzl", tensorflow_gcp_tools = "repo")
load("//third_party/tflite_mobilenet:workspace.bzl", tflite_mobilenet = "repo")
load("//third_party/tflite_ovic_testdata:workspace.bzl", tflite_ovic_testdata = "repo")
load("//third_party/vulkan_headers:workspace.bzl", vulkan_headers = "repo")
load("//third_party/xctestrunner:workspace.bzl", xctestrunner = "repo")
load("//third_party/xprof:workspace.bzl", xprof = "repo")

def _initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    FP16()
    absl()
    bazel_features_deps()
    benchmark()
    com_google_highway()
    ducc()
    dlpack()
    eigen3()
    farmhash()
    flatbuffers()
    fmt()
    gemmlowp()
    hexagon_nn()
    highwayhash()
    hwloc()
    icu()
    implib_so()
    jpeg()
    jpegxl()
    kissfft()
    libprotobuf_mutator()
    libwebp()
    ml_dtypes()
    nanobind()
    nasm()
    opencl_headers()
    org_brotli()
    pybind11_abseil()
    pybind11_bazel()
    raft()
    rapids_logger()
    rmm()
    robin_map()
    rocm_device_libs()
    ruy()
    shardy()
    skcms()
    slinky()
    spdlog()
    sobol_data()
    stablehlo()
    vulkan_headers()
    tensorrt()
    nvshmem()
    triton()
    xxd()
    # copybara: tsl vendor

# Toolchains & platforms required by Tensorflow to build.
def _tf_toolchains():
    # Loads all external repos to configure RBE builds.
    initialize_rbe_configs()

    # Note that we check the minimum bazel version in WORKSPACE.
    clang6_configure(name = "local_config_clang6")
    cc_download_clang_toolchain(name = "local_config_download_clang")
    tensorrt_configure(name = "local_config_tensorrt")
    git_configure(name = "local_config_git")
    syslibs_configure(name = "local_config_syslibs")
    python_configure(name = "local_config_python")
    hipcc_configure(name = "config_rocm_hipcc")  # Must be before rocm_configure.
    rocm_configure(name = "local_config_rocm")
    sycl_configure(name = "local_config_sycl")
    remote_execution_configure(name = "local_config_remote_execution")

    # For windows bazel build
    # TODO: Remove def file filter when TensorFlow can export symbols properly on Windows.
    def_file_filter_configure(name = "local_config_def_file_filter")

    # Point //external/local_config_arm_compiler to //external/arm_compiler
    arm_compiler_configure(
        name = "local_config_arm_compiler",
        build_file = "@xla//tools/toolchains/cpus/arm:template.BUILD",
        remote_config_repo_arm = "../arm_compiler",
        remote_config_repo_aarch64 = "../aarch64_compiler",
    )

    # Load aarch64 toolchain
    aarch64_compiler_configure()

    # TFLite crossbuild toolchain for embeddeds Linux
    arm_linux_toolchain_configure(
        name = "local_config_embedded_arm",
        build_file = "@xla//tools/toolchains/embedded/arm-linux:template.BUILD",
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
    # c) TF's automation will then upload the mirrored archive. For more information as well as
    # how to manually upload a mirror if necessary, see go/tf_mirror_md.

    tf_http_archive(
        name = "com_google_sentencepiece",
        build_file = "//third_party/sentencepiece:BUILD.bazel",
        patch_file = ["//third_party/sentencepiece:sp.patch"],
        sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
        strip_prefix = "sentencepiece-0.1.96",
        urls = tf_mirror_urls(
            "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip",
        ),
    )

    tf_http_archive(
        name = "darts_clone",
        build_file = "//third_party:darts_clone.BUILD",
        sha256 = "4a562824ec2fbb0ef7bd0058d9f73300173d20757b33bb69baa7e50349f65820",
        strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
        urls = tf_mirror_urls("https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.tar.gz"),
    )

    tf_http_archive(
        name = "cppitertools",
        build_file = "//third_party:cppitertools.BUILD",
        sha256 = "ba28a077e5099f72cf9e8efab2ced729214ab745a26cc21a1cebb81defd6c2d0",
        strip_prefix = "cppitertools-2.0",
        urls = tf_mirror_urls("https://github.com/ryanhaining/cppitertools/archive/v2.0.tar.gz"),
    )

    xnnpack()

    # XNNPack dependency.
    tf_http_archive(
        name = "KleidiAI",
        sha256 = "be1d6fb524b2a5e3772b38472a24d660e22b210f6b53b73bd8a5437ac2d882a7",
        strip_prefix = "kleidiai-d41219d3db13758074a6440d7b55a87487334c8b",
        urls = tf_mirror_urls("https://github.com/ARM-software/kleidiai/archive/d41219d3db13758074a6440d7b55a87487334c8b.zip"),
    )

    FXdiv()
    pthreadpool()

    cpuinfo()

    cudnn_frontend()

    cutlass()
    onednn()

    compute_library()

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
        build_file = "@xla//tools/toolchains/embedded/arm-linux:aarch64-linux-toolchain.BUILD",
        sha256 = "50cdef6c5baddaa00f60502cc8b59cc11065306ae575ad2f51e412a9b2a90364",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"),
    )

    tf_http_archive(
        name = "armhf_linux_toolchain",
        build_file = "@xla//tools/toolchains/embedded/arm-linux:armhf-linux-toolchain.BUILD",
        sha256 = "3f76650b1d048036473b16b647b8fd005ffccd1a2869c10994967e0e49f26ac2",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-linux-gnueabihf",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-linux-gnueabihf.tar.xz"),
    )

    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "8635bc46ac8d73974b4198229805287c8d620245f2081af155d7d96d4988a3a5",
        strip_prefix = "re2-927f5d53caf8111721e734cf24724686bb745f55",
        system_build_file = "//third_party/systemlibs:re2.BUILD",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/927f5d53caf8111721e734cf24724686bb745f55.tar.gz"),
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
        },
    )

    tf_http_archive(
        name = "com_github_google_crc32c",
        sha256 = "6b3b1d861bb8307658b2407bc7a4c59e566855ef5368a60b35c893551e4788e9",
        build_file = "@com_github_googlecloudplatform_google_cloud_cpp//bazel:crc32c.BUILD",
        strip_prefix = "crc32c-1.0.6",
        urls = tf_mirror_urls("https://github.com/google/crc32c/archive/1.0.6.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_googlecloudplatform_google_cloud_cpp",
        sha256 = "ff82045b9491f0d880fc8e5c83fd9542eafb156dcac9ff8c6209ced66ed2a7f0",
        strip_prefix = "google-cloud-cpp-1.17.1",
        repo_mapping = {
            "@com_github_curl_curl": "@curl",
            "@com_github_nlohmann_json": "@nlohmann_json_lib",
        },
        system_build_file = "//third_party/systemlibs:google_cloud_cpp.BUILD",
        system_link_files = {
            "//third_party/systemlibs:google_cloud_cpp.google.cloud.bigtable.BUILD": "google/cloud/bigtable/BUILD",
        },
        urls = tf_mirror_urls("https://github.com/googleapis/google-cloud-cpp/archive/v1.17.1.tar.gz"),
    )

    tensorflow_gcp_tools()
    googleapis()
    png()
    sqlite()
    gif()
    six()
    absl_py()

    maybe(
        tf_http_archive,
        name = "com_google_protobuf",
        patch_file = [
            "@xla//third_party/protobuf:protobuf.patch",
            "@xla//third_party/protobuf:protobuf_arena.patch",
        ],
        sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
        strip_prefix = "protobuf-6.31.1",
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip"),
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
            "@protobuf_pip_deps": "@pypi",
        },
    )

    tf_http_archive(
        name = "com_google_googletest",
        # Use the commit on 2025/6/09:
        # https://github.com/google/googletest/commit/28e9d1f26771c6517c3b4be10254887673c94018
        sha256 = "f253ca1a07262f8efde8328e4b2c68979e40ddfcfc001f70d1d5f612c7de2974",
        strip_prefix = "googletest-28e9d1f26771c6517c3b4be10254887673c94018",
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
        # $ git checkout 28e9d1f26771c6517c3b4be10254887673c94018
        # ... make local changes to googletest ...
        # $ git diff > <client-root>/third_party/tensorflow/third_party/googletest/googletest.patch
        #
        # The patch path is relative to third_party/tensorflow.
        patch_file = ["@xla//third_party/googletest:googletest.patch"],
        urls = tf_mirror_urls("https://github.com/google/googletest/archive/28e9d1f26771c6517c3b4be10254887673c940189.zip"),
    )

    tf_http_archive(
        name = "com_google_fuzztest",
        sha256 = "c75f224b34c3c62ee901381fb743f6326f7b91caae0ceb8fe62f3fd36f187627",
        strip_prefix = "fuzztest-58b4e7065924f1a284952b84ea827ce35a87e4dc",
        urls = tf_mirror_urls("https://github.com/google/fuzztest/archive/58b4e7065924f1a284952b84ea827ce35a87e4dc.zip"),
    )

    tf_http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        urls = tf_mirror_urls("https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"),
    )

    tf_http_archive(
        name = "curl",
        build_file = "@xla//third_party:curl.BUILD",
        sha256 = "264537d90e58d2b09dddc50944baf3c38e7089151c8986715e2aaeaaf2b8118f",
        strip_prefix = "curl-8.11.0",
        system_build_file = "//third_party/systemlibs:curl.BUILD",
        urls = tf_mirror_urls("https://curl.se/download/curl-8.11.0.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "e2ace790a5f2d0f83259d1390a816a33b013ea34df2e86084d927e58daa4c5d9",
        strip_prefix = "grpc-1.78.0",
        system_build_file = "//third_party/systemlibs:grpc.BUILD",
        patch_file = [
            "@xla//third_party/grpc:grpc.patch",
        ],
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/refs/tags/v1.78.0.tar.gz"),
    )

    linenoise()

    # Load the raw llvm-project.  llvm does not have build rules set up by default,
    # but provides a script for setting up build rules via overlays.
    llvm("llvm-raw")

    llvm_openmp()

    tf_http_archive(
        name = "jsoncpp_git",
        sha256 = "f409856e5920c18d0c2fb85276e24ee607d2a09b5e7d5f0a371368903c275da2",
        strip_prefix = "jsoncpp-1.9.5",
        system_build_file = "//third_party/systemlibs:jsoncpp.BUILD",
        urls = tf_mirror_urls("https://github.com/open-source-parsers/jsoncpp/archive/1.9.5.tar.gz"),
    )

    tf_http_archive(
        name = "boringssl",
        sha256 = "9dc53f851107eaf87b391136d13b815df97ec8f76dadb487b58b2fc45e624d2c",
        strip_prefix = "boringssl-c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc",
        system_build_file = "//third_party/systemlibs:boringssl.BUILD",
        urls = tf_mirror_urls("https://github.com/google/boringssl/archive/c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc.tar.gz"),
    )

    # Note: if you update this, you have to update libpng too. See cl/437813808
    tf_http_archive(
        name = "zlib",
        build_file = "@xla//third_party:zlib.BUILD",
        sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
        strip_prefix = "zlib-1.3.1",
        system_build_file = "//third_party/systemlibs:zlib.BUILD",
        urls = tf_mirror_urls("https://zlib.net/fossils/zlib-1.3.1.tar.gz"),
    )

    fft2d()

    tf_http_archive(
        name = "snappy",
        build_file = "@xla//third_party:snappy.BUILD",
        sha256 = "7ee7540b23ae04df961af24309a55484e7016106e979f83323536a1322cedf1b",
        strip_prefix = "snappy-1.2.0",
        system_build_file = "//third_party/systemlibs:snappy.BUILD",
        urls = tf_mirror_urls("https://github.com/google/snappy/archive/1.2.0.zip"),
    )

    nccl()

    nvtx()
    java_repositories()

    pprof()
    cython()
    arm_neon_2_x86_sse()
    tflite_mobilenet()
    tflite_ovic_testdata()

    tf_http_archive(
        name = "build_bazel_rules_android",
        sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
        strip_prefix = "rules_android-0.1.1",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip"),
    )

    tf_http_archive(
        name = "rules_android_ndk",
        sha256 = "0ab5ddae72dff0dfae92a31a0704d4543e818e360786e44d2093a6b8ff5e8fda",
        strip_prefix = "rules_android_ndk-461e8c99b7f06bc86a15317505d48fc0decd7dcc",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_android_ndk/archive/461e8c99b7f06bc86a15317505d48fc0decd7dcc.zip"),
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
        sha256 = "1ae6fcf983cff3edab717636f91ad0efff2e5ba75607fdddddfd6ad0dbdfaf10",
        urls = tf_mirror_urls("https://github.com/bazelbuild/apple_support/releases/download/1.24.5/apple_support.1.24.5.tar.gz"),
    )

    # https://github.com/apple/swift-protobuf/releases
    tf_http_archive(
        name = "com_github_apple_swift_swift_protobuf",
        strip_prefix = "swift-protobuf-1.19.0/",
        sha256 = "f057930b9dbd17abeaaceaa45e9f8b3e87188c05211710563d2311b9edf490aa",
        urls = tf_mirror_urls("https://github.com/apple/swift-protobuf/archive/1.19.0.tar.gz"),
    )

    xctestrunner()
    nlohmann_json()
    tf_http_archive(
        name = "pybind11",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11/archive/v2.13.6.tar.gz"),
        sha256 = "e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20",
        strip_prefix = "pybind11-2.13.6",
        build_file = "@xla//third_party:pybind11.BUILD",
        system_build_file = "//third_party/systemlibs:pybind11.BUILD",
    )

    tf_http_archive(
        name = "pybind11_protobuf",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_protobuf/archive/f02a2b7653bc50eb5119d125842a3870db95d251.zip"),
        sha256 = "3cf7bf0f23954c5ce6c37f0a215f506efa3035ca06e3b390d67f4cbe684dce23",
        strip_prefix = "pybind11_protobuf-f02a2b7653bc50eb5119d125842a3870db95d251",
    )

    coremltools()

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
        # How to generate/update the patch files:
        # 1. go to a temporary directory.
        # 2. run commands:
        #      git clone https://github.com/protocolbuffers/upb
        #      cd upb
        #      git checkout 9effcbcb27f0a665f9f345030188c0b291e32482
        # 3. Edit the files as needed.
        # 4. run command:
        #      git diff > path-to-the-patch-file
        patch_file = [
            "@xla//third_party/grpc:upb_platform_fix.patch",
            # Disables warning-as-error when building upb, as it generates
            # warnings when compiled with clang.
            "@xla//third_party/grpc:upb_build.patch",
        ],
        urls = tf_mirror_urls("https://github.com/protocolbuffers/upb/archive/9effcbcb27f0a665f9f345030188c0b291e32482.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_glog_glog",
        sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
        strip_prefix = "glog-0.4.0",
        urls = tf_mirror_urls("https://github.com/google/glog/archive/refs/tags/v0.4.0.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_ortools",
        sha256 = "f6a0bd5b9f3058aa1a814b798db5d393c31ec9cbb6103486728997b49ab127bc",
        strip_prefix = "or-tools-9.11",
        patch_file = ["@xla//third_party/ortools:ortools.patch"],
        urls = tf_mirror_urls("https://github.com/google/or-tools/archive/v9.11.tar.gz"),
        repo_mapping = {
            "@com_google_protobuf_cc": "@com_google_protobuf",
            "@eigen": "@eigen_archive",
        },
    )

    tf_http_archive(
        name = "glpk",
        sha256 = "9a5dab356268b4f177c33e00ddf8164496dc2434e83bd1114147024df983a3bb",
        build_file = "//third_party/ortools:glpk.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/ftp.gnu.org/gnu/glpk/glpk-4.52.tar.gz",
            "http://ftp.gnu.org/gnu/glpk/glpk-4.52.tar.gz",
        ],
    )

    tf_http_archive(
        name = "scip",
        sha256 = "ee221bd13a6b24738f2e74321e2efdebd6d7c603574ca6f6cb9d4472ead2c22f",
        strip_prefix = "scip-900",
        build_file = "@com_google_ortools//bazel:scip.BUILD.bazel",
        patch_file = ["@com_google_ortools//bazel:scip-v900.patch"],
        urls = tf_mirror_urls("https://github.com/scipopt/scip/archive/refs/tags/v900.tar.gz"),
    )

    tf_http_archive(
        name = "bliss",
        build_file = "//third_party/ortools:bliss.BUILD",
        sha256 = "f57bf32804140cad58b1240b804e0dbd68f7e6bf67eba8e0c0fa3a62fd7f0f84",
        urls = tf_mirror_urls("https://github.com/google/or-tools/releases/download/v9.0/bliss-0.73.zip"),
        #url = "http://www.tcs.hut.fi/Software/bliss/bliss-0.73.zip",
    )

    riegeli()
    net_zstd()

    # Required by riegeli.
    tf_http_archive(
        name = "net_zstd",
        build_file = "@xla//third_party:net_zstd.BUILD",
        sha256 = "7897bc5d620580d9b7cd3539c44b59d78f3657d33663fe97a145e07b4ebd69a4",
        strip_prefix = "zstd-1.5.7/lib",
        urls = tf_mirror_urls("https://github.com/facebook/zstd/archive/v1.5.7.zip"),  # 2025-05-20
    )

    xprof(
        repo_mapping = {
            "@com_github_nlohmann_json": "@nlohmann_json_lib",
        },
    )

    # used for adding androidx.annotation dependencies in tflite android jni.
    maven_install(
        artifacts = [
            "androidx.annotation:annotation:aar:1.1.0",
        ],
        repositories = [
            "https://jcenter.bintray.com",
            "https://maven.google.com",
            "https://dl.google.com/dl/android/maven2",
            "https://repo1.maven.org/maven2",
        ],
        fetch_sources = True,
        version_conflict_policy = "pinned",
    )

def workspace():
    """TensorFlow workspace initialization."""

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

    tfrt_dependencies()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace2 = workspace
