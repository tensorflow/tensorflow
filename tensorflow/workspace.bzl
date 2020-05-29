# TensorFlow external dependencies that can be loaded in WORKSPACE files.

load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//third_party/nccl:nccl_configure.bzl", "nccl_configure")
load("//third_party/mkl:build_defs.bzl", "mkl_repository")
load("//third_party/git:git_configure.bzl", "git_configure")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/sycl:sycl_configure.bzl", "sycl_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("//third_party/toolchains/remote:configure.bzl", "remote_execution_configure")
load("//third_party/toolchains/clang6:repo.bzl", "clang6_configure")
load("//third_party/toolchains/cpus/arm:arm_compiler_configure.bzl", "arm_compiler_configure")
load("//third_party/toolchains/embedded/arm-linux:arm_linux_toolchain_configure.bzl", "arm_linux_toolchain_configure")
load("//third_party:repo.bzl", "tf_http_archive")
load("//third_party/clang_toolchain:cc_configure_clang.bzl", "cc_download_clang_toolchain")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("@bazel_tools//tools/build_defs/repo:java.bzl", "java_import_external")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")
load(
    "//tensorflow/tools/def_file_filter:def_file_filter_configure.bzl",
    "def_file_filter_configure",
)
load("//third_party/FP16:workspace.bzl", FP16 = "repo")
load("//third_party/aws:workspace.bzl", aws = "repo")
load("//third_party/clog:workspace.bzl", clog = "repo")
load("//third_party/cpuinfo:workspace.bzl", cpuinfo = "repo")
load("//third_party/dlpack:workspace.bzl", dlpack = "repo")
load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("//third_party/hexagon:workspace.bzl", hexagon_nn = "repo")
load("//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("//third_party/icu:workspace.bzl", icu = "repo")
load("//third_party/jpeg:workspace.bzl", jpeg = "repo")
load("//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party/opencl_headers:workspace.bzl", opencl_headers = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/pasta:workspace.bzl", pasta = "repo")
load("//third_party/psimd:workspace.bzl", psimd = "repo")
load("//third_party/ruy:workspace.bzl", ruy = "repo")
load("//third_party/sobol_data:workspace.bzl", sobol_data = "repo")
load("//third_party/vulkan_headers:workspace.bzl", vulkan_headers = "repo")
load("//third_party/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")

def initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    FP16()
    aws()
    clog()
    cpuinfo()
    dlpack()
    flatbuffers()
    hexagon_nn()
    highwayhash()
    hwloc()
    icu()
    kissfft()
    jpeg()
    nasm()
    opencl_headers()
    pasta()
    psimd()
    sobol_data()
    vulkan_headers()
    ruy()

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

# If TensorFlow is linked as a submodule.
# path_prefix is no longer used.
# tf_repo_name is thought to be under consideration.
def tf_workspace(path_prefix = "", tf_repo_name = ""):
    tf_repositories(path_prefix, tf_repo_name)
    tf_bind()

# Toolchains & platforms required by Tensorflow to build.
def tf_toolchains():
    native.register_execution_platforms("@local_execution_config_platform//:platform")
    native.register_toolchains("@local_execution_config_python//:py_toolchain")

# Define all external repositories required by TensorFlow
def tf_repositories(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds."""

    # Initialize toolchains and platforms.
    tf_toolchains()

    # Loads all external repos to configure RBE builds.
    initialize_rbe_configs()

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
    remote_execution_configure(name = "local_config_remote_execution")

    initialize_third_party()

    # For windows bazel build
    # TODO: Remove def file filter when TensorFlow can export symbols properly on Windows.
    def_file_filter_configure(name = "local_config_def_file_filter")

    # Point //external/local_config_arm_compiler to //external/arm_compiler
    arm_compiler_configure(
        name = "local_config_arm_compiler",
        build_file = clean_dep("//third_party/toolchains/cpus/arm:BUILD"),
        remote_config_repo_arm = "../arm_compiler",
        remote_config_repo_aarch64 = "../aarch64_compiler",
    )

    # TFLite crossbuild toolchain for embeddeds Linux
    arm_linux_toolchain_configure(
        name = "local_config_embedded_arm",
        build_file = clean_dep("//third_party/toolchains/embedded/arm-linux:BUILD"),
        aarch64_repo = "../aarch64_linux_toolchain",
        armhf_repo = "../armhf_linux_toolchain",
    )

    mkl_repository(
        name = "mkl_linux",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "a936d6b277a33d2a027a024ea8e65df62bd2e162c7ca52c48486ed9d5dc27160",
        strip_prefix = "mklml_lnx_2019.0.5.20190502",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/mkl-dnn/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz",
            "https://github.com/intel/mkl-dnn/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz",
        ],
    )
    mkl_repository(
        name = "mkl_windows",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "33cc27652df3b71d7cb84b26718b5a2e8965e2c864a502347db02746d0430d57",
        strip_prefix = "mklml_win_2020.0.20190813",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/mkl-dnn/releases/download/v0.21/mklml_win_2020.0.20190813.zip",
            "https://github.com/intel/mkl-dnn/releases/download/v0.21/mklml_win_2020.0.20190813.zip",
        ],
    )
    mkl_repository(
        name = "mkl_darwin",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "2fbb71a0365d42a39ea7906568d69b1db3bfc9914fee75eedb06c5f32bf5fa68",
        strip_prefix = "mklml_mac_2019.0.5.20190502",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/mkl-dnn/releases/download/v0.21/mklml_mac_2019.0.5.20190502.tgz",
            "https://github.com/intel/mkl-dnn/releases/download/v0.21/mklml_mac_2019.0.5.20190502.tgz",
        ],
    )

    if path_prefix:
        print("path_prefix was specified to tf_workspace but is no longer used " +
              "and will be removed in the future.")

    tf_http_archive(
        name = "XNNPACK",
        sha256 = "dfa6181e238f0ca88a641952678cd7f3e38da541d8b731ce3fea1d0eeffb6101",
        strip_prefix = "XNNPACK-b2217ddb5fa74db09d9da1326902269ae18e41ad",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/XNNPACK/archive/b2217ddb5fa74db09d9da1326902269ae18e41ad.zip",
            "https://github.com/google/XNNPACK/archive/b2217ddb5fa74db09d9da1326902269ae18e41ad.zip",
        ],
    )

    tf_http_archive(
        name = "FXdiv",
        sha256 = "ab7dfb08829bee33dca38405d647868fb214ac685e379ec7ef2bebcd234cd44d",
        strip_prefix = "FXdiv-b408327ac2a15ec3e43352421954f5b1967701d1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip",
            "https://github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip",
        ],
    )

    tf_http_archive(
        name = "pthreadpool",
        sha256 = "9f5fb7f87dc778d9c1d638826344b762afa23884d0252526337ae710264faef3",
        strip_prefix = "pthreadpool-18a7156cb9be8e534acefade42e46d4209600c35",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/pthreadpool/archive/18a7156cb9be8e534acefade42e46d4209600c35.zip",
            "https://github.com/Maratyszcza/pthreadpool/archive/18a7156cb9be8e534acefade42e46d4209600c35.zip",
        ],
    )

    # Important: If you are upgrading MKL-DNN, then update the version numbers
    # in third_party/mkl_dnn/mkldnn.BUILD. In addition, the new version of
    # MKL-DNN might require upgrading MKL ML libraries also. If they need to be
    # upgraded then update the version numbers on all three versions above
    # (Linux, Mac, Windows).
    tf_http_archive(
        name = "mkl_dnn",
        build_file = clean_dep("//third_party/mkl_dnn:mkldnn.BUILD"),
        sha256 = "a0211aeb5e7dad50b97fa5dffc1a2fe2fe732572d4164e1ee8750a2ede43fbec",
        strip_prefix = "oneDNN-0.21.3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v0.21.3.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/v0.21.3.tar.gz",
        ],
    )

    tf_http_archive(
        name = "mkl_dnn_v1",
        build_file = clean_dep("//third_party/mkl_dnn:mkldnn_v1.BUILD"),
        sha256 = "54737bcb4dc1961d32ee75da3ecc529fa48198f8b2ca863a079e19a9c4adb70f",
        strip_prefix = "oneDNN-1.4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v1.4.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/v1.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_absl",
        build_file = clean_dep("//third_party:com_google_absl.BUILD"),
        # TODO: Remove the patch when https://github.com/abseil/abseil-cpp/issues/326 is resolved
        # and when TensorFlow is build against CUDA 10.2
        patch_file = clean_dep("//third_party:com_google_absl_fix_mac_and_nvcc_build.patch"),
        sha256 = "f368a8476f4e2e0eccf8a7318b98dafbe30b2600f4e3cf52636e5eb145aba06a",  # SHARED_ABSL_SHA
        strip_prefix = "abseil-cpp-df3ea785d8c30a9503321a3d35ee7d35808f190d",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
        ],
    )

    tf_http_archive(
        name = "eigen_archive",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        patch_file = clean_dep("//third_party/eigen3:gpu_packet_math.patch"),
        sha256 = "854eabe6817e38d7738fde6ec39c3dfc55fd5e68b2523de8cae936f391a38a69",  # SHARED_EIGEN_SHA
        strip_prefix = "eigen-cc86a31e20b48b0f03d714b4d1b1f50d52848d36",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/cc86a31e20b48b0f03d714b4d1b1f50d52848d36/eigen-cc86a31e20b48b0f03d714b4d1b1f50d52848d36.tar.gz",
            "https://gitlab.com/libeigen/eigen/-/archive/cc86a31e20b48b0f03d714b4d1b1f50d52848d36/eigen-cc86a31e20b48b0f03d714b4d1b1f50d52848d36.tar.gz",
        ],
    )

    tf_http_archive(
        name = "arm_compiler",
        build_file = clean_dep("//:arm_compiler.BUILD"),
        sha256 = "b9e7d50ffd9996ed18900d041d362c99473b382c0ae049b2fce3290632d2656f",
        strip_prefix = "rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf/",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz",
            "https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz",
        ],
    )

    tf_http_archive(
        # This is the latest `aarch64-none-linux-gnu` compiler provided by ARM
        # See https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads
        # The archive contains GCC version 9.2.1
        name = "aarch64_compiler",
        build_file = "//:arm_compiler.BUILD",
        sha256 = "8dfe681531f0bd04fb9c53cf3c0a3368c616aa85d48938eebe2b516376e06a66",
        strip_prefix = "gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz",
            "https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz",
        ],
    )

    tf_http_archive(
        name = "aarch64_linux_toolchain",
        build_file = clean_dep("//third_party/toolchains/embedded/arm-linux:aarch64-linux-toolchain.BUILD"),
        sha256 = "8ce3e7688a47d8cd2d8e8323f147104ae1c8139520eca50ccf8a7fa933002731",
        strip_prefix = "gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz",
            "https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz",
        ],
    )

    tf_http_archive(
        name = "armhf_linux_toolchain",
        build_file = clean_dep("//third_party/toolchains/embedded/arm-linux:armhf-linux-toolchain.BUILD"),
        sha256 = "d4f6480ecaa99e977e3833cc8a8e1263f9eecd1ce2d022bb548a24c4f32670f5",
        strip_prefix = "gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz",
            "https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz",
        ],
    )

    tf_http_archive(
        name = "libxsmm_archive",
        build_file = clean_dep("//third_party:libxsmm.BUILD"),
        sha256 = "9c0af4509ea341d1ee2c6c19fc6f19289318c3bd4b17844efeb9e7f9691abf76",
        strip_prefix = "libxsmm-1.14",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/hfp/libxsmm/archive/1.14.tar.gz",
            "https://github.com/hfp/libxsmm/archive/1.14.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "d070e2ffc5476c496a6a872a6f246bfddce8e7797d6ba605a7c8d72866743bf9",
        strip_prefix = "re2-506cfa4bffd060c06ec338ce50ea3468daa6c814",
        system_build_file = clean_dep("//third_party/systemlibs:re2.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
            "https://github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_github_googlecloudplatform_google_cloud_cpp",
        sha256 = "d67fed328d82aa404c3ab8f52814914f419a673573e3bbd98b4e6c405ca3cd06",
        strip_prefix = "google-cloud-cpp-0.17.0",
        system_build_file = clean_dep("//third_party/systemlibs:google_cloud_cpp.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:google_cloud_cpp.google.cloud.bigtable.BUILD": "google/cloud/bigtable/BUILD",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/googleapis/google-cloud-cpp/archive/v0.17.0.tar.gz",
            "https://github.com/googleapis/google-cloud-cpp/archive/v0.17.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_googleapis",
        build_file = clean_dep("//third_party/googleapis:googleapis.BUILD"),
        sha256 = "7ebab01b06c555f4b6514453dc3e1667f810ef91d1d4d2d3aa29bb9fcb40a900",
        strip_prefix = "googleapis-541b1ded4abadcc38e8178680b0677f65594ea6f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/googleapis/googleapis/archive/541b1ded4abadcc38e8178680b0677f65594ea6f.zip",
            "https://github.com/googleapis/googleapis/archive/541b1ded4abadcc38e8178680b0677f65594ea6f.zip",
        ],
    )

    tf_http_archive(
        name = "gemmlowp",
        sha256 = "43146e6f56cb5218a8caaab6b5d1601a083f1f31c06ff474a4378a7d35be9cfb",  # SHARED_GEMMLOWP_SHA
        strip_prefix = "gemmlowp-fda83bdc38b118cc6b56753bd540caa49e570745",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip",
            "https://github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip",
        ],
    )

    tf_http_archive(
        name = "farmhash_archive",
        build_file = clean_dep("//third_party:farmhash.BUILD"),
        sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",  # SHARED_FARMHASH_SHA
        strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
            "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
        ],
    )

    tf_http_archive(
        name = "png",
        build_file = clean_dep("//third_party:png.BUILD"),
        patch_file = clean_dep("//third_party:png_fix_rpi.patch"),
        sha256 = "ca74a0dace179a8422187671aee97dd3892b53e168627145271cad5b5ac81307",
        strip_prefix = "libpng-1.6.37",
        system_build_file = clean_dep("//third_party/systemlibs:png.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
            "https://github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
        ],
    )

    tf_http_archive(
        name = "org_sqlite",
        build_file = clean_dep("//third_party:sqlite.BUILD"),
        sha256 = "f3c79bc9f4162d0b06fa9fe09ee6ccd23bb99ce310b792c5145f87fbcc30efca",
        strip_prefix = "sqlite-amalgamation-3310100",
        system_build_file = clean_dep("//third_party/systemlibs:sqlite.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/www.sqlite.org/2020/sqlite-amalgamation-3310100.zip",
            "https://www.sqlite.org/2020/sqlite-amalgamation-3310100.zip",
        ],
    )

    tf_http_archive(
        name = "gif",
        build_file = clean_dep("//third_party:gif.BUILD"),
        patch_file = clean_dep("//third_party:gif_fix_strtok_r.patch"),
        sha256 = "31da5562f44c5f15d63340a09a4fd62b48c45620cd302f77a6d9acf0077879bd",
        strip_prefix = "giflib-5.2.1",
        system_build_file = clean_dep("//third_party/systemlibs:gif.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz",
            "https://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "six_archive",
        build_file = clean_dep("//third_party:six.BUILD"),
        sha256 = "d16a0141ec1a18405cd4ce8b4613101da75da0e9a7aec5bdd4fa804d0e0eba73",
        strip_prefix = "six-1.12.0",
        system_build_file = clean_dep("//third_party/systemlibs:six.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz",
            "https://pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "astor_archive",
        build_file = clean_dep("//third_party:astor.BUILD"),
        sha256 = "95c30d87a6c2cf89aa628b87398466840f0ad8652f88eb173125a6df8533fb8d",
        strip_prefix = "astor-0.7.1",
        system_build_file = clean_dep("//third_party/systemlibs:astor.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/99/80/f9482277c919d28bebd85813c0a70117214149a96b08981b72b63240b84c/astor-0.7.1.tar.gz",
            "https://pypi.python.org/packages/99/80/f9482277c919d28bebd85813c0a70117214149a96b08981b72b63240b84c/astor-0.7.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "astunparse_archive",
        build_file = clean_dep("//third_party:astunparse.BUILD"),
        sha256 = "5ad93a8456f0d084c3456d059fd9a92cce667963232cbf763eac3bc5b7940872",
        strip_prefix = "astunparse-1.6.3/lib",
        system_build_file = clean_dep("//third_party/systemlibs:astunparse.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/files.pythonhosted.org/packages/f3/af/4182184d3c338792894f34a62672919db7ca008c89abee9b564dd34d8029/astunparse-1.6.3.tar.gz",
            "https://files.pythonhosted.org/packages/f3/af/4182184d3c338792894f34a62672919db7ca008c89abee9b564dd34d8029/astunparse-1.6.3.tar.gz",
        ],
    )

    filegroup_external(
        name = "astunparse_license",
        licenses = ["notice"],  # PSFL
        sha256_urls = {
            "92fc0e4f4fa9460558eedf3412b988d433a2dcbb3a9c45402a145a4fab8a6ac6": [
                "https://storage.googleapis.com/mirror.tensorflow.org/raw.githubusercontent.com/simonpercivall/astunparse/v1.6.2/LICENSE",
                "https://raw.githubusercontent.com/simonpercivall/astunparse/v1.6.2/LICENSE",
            ],
        },
    )

    tf_http_archive(
        name = "functools32_archive",
        build_file = clean_dep("//third_party:functools32.BUILD"),
        sha256 = "f6253dfbe0538ad2e387bd8fdfd9293c925d63553f5813c4e587745416501e6d",
        strip_prefix = "functools32-3.2.3-2",
        system_build_file = clean_dep("//third_party/systemlibs:functools32.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/c5/60/6ac26ad05857c601308d8fb9e87fa36d0ebf889423f47c3502ef034365db/functools32-3.2.3-2.tar.gz",
            "https://pypi.python.org/packages/c5/60/6ac26ad05857c601308d8fb9e87fa36d0ebf889423f47c3502ef034365db/functools32-3.2.3-2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "gast_archive",
        build_file = clean_dep("//third_party:gast.BUILD"),
        sha256 = "b881ef288a49aa81440d2c5eb8aeefd4c2bb8993d5f50edae7413a85bfdb3b57",
        strip_prefix = "gast-0.3.3",
        system_build_file = clean_dep("//third_party/systemlibs:gast.BUILD"),
        urls = [
            "http://mirror.tensorflow.org/files.pythonhosted.org/packages/12/59/eaa15ab9710a20e22225efd042cd2d6a0b559a0656d5baba9641a2a4a921/gast-0.3.3.tar.gz",
            "https://files.pythonhosted.org/packages/12/59/eaa15ab9710a20e22225efd042cd2d6a0b559a0656d5baba9641a2a4a921/gast-0.3.3.tar.gz",
        ],
    )

    tf_http_archive(
        name = "termcolor_archive",
        build_file = clean_dep("//third_party:termcolor.BUILD"),
        sha256 = "1d6d69ce66211143803fbc56652b41d73b4a400a2891d7bf7a1cdf4c02de613b",
        strip_prefix = "termcolor-1.1.0",
        system_build_file = clean_dep("//third_party/systemlibs:termcolor.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz",
            "https://pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "opt_einsum_archive",
        build_file = clean_dep("//third_party:opt_einsum.BUILD"),
        sha256 = "d3d464b4da7ef09e444c30e4003a27def37f85ff10ff2671e5f7d7813adac35b",
        strip_prefix = "opt_einsum-2.3.2",
        system_build_file = clean_dep("//third_party/systemlibs:opt_einsum.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/f6/d6/44792ec668bcda7d91913c75237314e688f70415ab2acd7172c845f0b24f/opt_einsum-2.3.2.tar.gz",
            "https://pypi.python.org/packages/f6/d6/44792ec668bcda7d91913c75237314e688f70415ab2acd7172c845f0b24f/opt_einsum-2.3.2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "absl_py",
        sha256 = "603febc9b95a8f2979a7bdb77d2f5e4d9b30d4e0d59579f88eba67d4e4cc5462",
        strip_prefix = "abseil-py-pypi-v0.9.0",
        system_build_file = clean_dep("//third_party/systemlibs:absl_py.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:absl_py.absl.BUILD": "absl/BUILD",
            "//third_party/systemlibs:absl_py.absl.flags.BUILD": "absl/flags/BUILD",
            "//third_party/systemlibs:absl_py.absl.testing.BUILD": "absl/testing/BUILD",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-py/archive/pypi-v0.9.0.tar.gz",
            "https://github.com/abseil/abseil-py/archive/pypi-v0.9.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "enum34_archive",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz",
            "https://pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz",
        ],
        sha256 = "8ad8c4783bf61ded74527bffb48ed9b54166685e4230386a9ed9b1279e2df5b1",
        build_file = clean_dep("//third_party:enum34.BUILD"),
        system_build_file = clean_dep("//third_party/systemlibs:enum34.BUILD"),
        strip_prefix = "enum34-1.1.6/enum",
    )

    tf_http_archive(
        name = "org_python_pypi_backports_weakref",
        build_file = clean_dep("//third_party:backports_weakref.BUILD"),
        sha256 = "8813bf712a66b3d8b85dc289e1104ed220f1878cf981e2fe756dfaabe9a82892",
        strip_prefix = "backports.weakref-1.0rc1/src",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
            "https://pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
        ],
    )

    filegroup_external(
        name = "org_python_license",
        licenses = ["notice"],  # Python 2.0
        sha256_urls = {
            "e76cacdf0bdd265ff074ccca03671c33126f597f39d0ed97bc3e5673d9170cf6": [
                "https://storage.googleapis.com/mirror.tensorflow.org/docs.python.org/2.7/_sources/license.rst.txt",
                "https://docs.python.org/2.7/_sources/license.rst.txt",
            ],
        },
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = clean_dep("//third_party/protobuf:protobuf.patch"),
        sha256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",
        strip_prefix = "protobuf-3.9.2",
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
            "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
        ],
    )

    tf_http_archive(
        name = "nsync",
        sha256 = "caf32e6b3d478b78cff6c2ba009c3400f8251f646804bcb65465666a9cea93c4",
        strip_prefix = "nsync-1.22.0",
        system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
            "https://github.com/google/nsync/archive/1.22.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_googletest",
        sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
        strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
            "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        ],
    )

    tf_http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "ae27cdbcd6a2f935baa78e4f21f675649271634c092b1be01469440495609d0e",
        strip_prefix = "gflags-2.2.1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/gflags/gflags/archive/v2.2.1.tar.gz",
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
            "https://storage.googleapis.com/mirror.tensorflow.org/ftp.exim.org/pub/pcre/pcre-8.42.tar.gz",
            "https://ftp.exim.org/pub/pcre/pcre-8.42.tar.gz",
        ],
    )

    tf_http_archive(
        name = "swig",
        build_file = clean_dep("//third_party:swig.BUILD"),
        sha256 = "58a475dbbd4a4d7075e5fe86d4e54c9edde39847cdb96a3053d87cb64a23a453",
        strip_prefix = "swig-3.0.8",
        system_build_file = clean_dep("//third_party/systemlibs:swig.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
            "https://ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
            "https://pilotfiber.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
        ],
    )

    tf_http_archive(
        name = "curl",
        build_file = clean_dep("//third_party:curl.BUILD"),
        sha256 = "01ae0c123dee45b01bbaef94c0bc00ed2aec89cb2ee0fd598e0d302a6b5e0a98",
        strip_prefix = "curl-7.69.1",
        system_build_file = clean_dep("//third_party/systemlibs:curl.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/curl.haxx.se/download/curl-7.69.1.tar.gz",
            "https://curl.haxx.se/download/curl-7.69.1.tar.gz",
        ],
    )

    # WARNING: make sure ncteisen@ and vpai@ are cc-ed on any CL to change the below rule
    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
        strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
        system_build_file = clean_dep("//third_party/systemlibs:grpc.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:BUILD": "bazel/BUILD",
            "//third_party/systemlibs:grpc.BUILD": "src/compiler/BUILD",
            "//third_party/systemlibs:grpc.bazel.grpc_deps.bzl": "bazel/grpc_deps.bzl",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
            "https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
        ],
    )

    tf_http_archive(
        name = "linenoise",
        build_file = clean_dep("//third_party:linenoise.BUILD"),
        sha256 = "7f51f45887a3d31b4ce4fa5965210a5e64637ceac12720cfce7954d6a2e812f7",
        strip_prefix = "linenoise-c894b9e59f02203dbe4e2be657572cf88c4230c3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
            "https://github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
        ],
    )

    # Check out LLVM and MLIR from llvm-project.
    LLVM_COMMIT = "b726d071b4aa46004228fc38ee5bfd167f999bfe"
    LLVM_SHA256 = "d7e67036dc89906cb2f80df7b0b7de6344d86eddf6e98bb4d01a578242889a73"
    LLVM_URLS = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    ]
    tf_http_archive(
        name = "llvm-project",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = LLVM_URLS,
        additional_build_files = {
            clean_dep("//third_party/llvm:llvm.autogenerated.BUILD"): "llvm/BUILD",
            "//third_party/mlir:BUILD": "mlir/BUILD",
            "//third_party/mlir:test.BUILD": "mlir/test/BUILD",
        },
    )

    tf_http_archive(
        name = "lmdb",
        build_file = clean_dep("//third_party:lmdb.BUILD"),
        sha256 = "f3927859882eb608868c8c31586bb7eb84562a40a6bf5cc3e13b6b564641ea28",
        strip_prefix = "lmdb-LMDB_0.9.22/libraries/liblmdb",
        system_build_file = clean_dep("//third_party/systemlibs:lmdb.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
            "https://github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
        ],
    )

    tf_http_archive(
        name = "jsoncpp_git",
        build_file = clean_dep("//third_party:jsoncpp.BUILD"),
        sha256 = "77a402fb577b2e0e5d0bdc1cf9c65278915cdb25171e3452c68b6da8a561f8f0",
        strip_prefix = "jsoncpp-1.9.2",
        system_build_file = clean_dep("//third_party/systemlibs:jsoncpp.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/open-source-parsers/jsoncpp/archive/1.9.2.tar.gz",
            "https://github.com/open-source-parsers/jsoncpp/archive/1.9.2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "boringssl",
        sha256 = "a9c3b03657d507975a32732f04563132b4553c20747cec6dc04de475c8bdf29f",
        strip_prefix = "boringssl-80ca9f9f6ece29ab132cce4cf807a9465a18cfac",
        system_build_file = clean_dep("//third_party/systemlibs:boringssl.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/boringssl/archive/80ca9f9f6ece29ab132cce4cf807a9465a18cfac.tar.gz",
            "https://github.com/google/boringssl/archive/80ca9f9f6ece29ab132cce4cf807a9465a18cfac.tar.gz",
        ],
    )

    tf_http_archive(
        name = "zlib",
        build_file = clean_dep("//third_party:zlib.BUILD"),
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        system_build_file = clean_dep("//third_party/systemlibs:zlib.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    tf_http_archive(
        name = "fft2d",
        build_file = clean_dep("//third_party/fft2d:fft2d.BUILD"),
        sha256 = "ada7e99087c4ed477bfdf11413f2ba8db8a840ba9bbf8ac94f4f3972e2a7cec9",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz",
            "https://www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz",
        ],
    )

    tf_http_archive(
        name = "snappy",
        build_file = clean_dep("//third_party:snappy.BUILD"),
        sha256 = "16b677f07832a612b0836178db7f374e414f94657c138e6993cbfc5dcc58651f",
        strip_prefix = "snappy-1.1.8",
        system_build_file = clean_dep("//third_party/systemlibs:snappy.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/snappy/archive/1.1.8.tar.gz",
            "https://github.com/google/snappy/archive/1.1.8.tar.gz",
        ],
    )

    tf_http_archive(
        name = "nccl_archive",
        build_file = clean_dep("//third_party:nccl/archive.BUILD"),
        patch_file = clean_dep("//third_party/nccl:archive.patch"),
        sha256 = "7ff66aca18392b162829612e02c00b123a58ec35869334f72d7e5afaf5ea4a13",
        strip_prefix = "nccl-3701130b3c1bcdb01c14b3cb70fe52498c1e82b7",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nvidia/nccl/archive/3701130b3c1bcdb01c14b3cb70fe52498c1e82b7.tar.gz",
            "https://github.com/nvidia/nccl/archive/3701130b3c1bcdb01c14b3cb70fe52498c1e82b7.tar.gz",
        ],
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
        name = "com_google_pprof",
        build_file = clean_dep("//third_party:pprof.BUILD"),
        sha256 = "e0928ca4aa10ea1e0551e2d7ce4d1d7ea2d84b2abbdef082b0da84268791d0c4",
        strip_prefix = "pprof-c0fb62ec88c411cc91194465e54db2632845b650",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
            "https://github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
        ],
    )

    tf_http_archive(
        name = "cub_archive",
        build_file = clean_dep("//third_party:cub.BUILD"),
        patch_file = clean_dep("//third_party:cub.pr170.patch"),
        sha256 = "6bfa06ab52a650ae7ee6963143a0bbc667d6504822cbd9670369b598f18c58c3",
        strip_prefix = "cub-1.8.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.8.0.zip",
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
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/cython/cython/archive/0.28.4.tar.gz",
            "https://github.com/cython/cython/archive/0.28.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = clean_dep("//third_party:arm_neon_2_x86_sse.BUILD"),
        sha256 = "213733991310b904b11b053ac224fee2d4e0179e46b52fe7f8735b8831e04dcc",
        strip_prefix = "ARM_NEON_2_x86_SSE-1200fe90bb174a6224a525ee60148671a786a71f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
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
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
            "https://github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_float",
        build_file = clean_dep("//third_party:tflite_mobilenet_float.BUILD"),
        sha256 = "2fadeabb9968ec6833bee903900dda6e61b3947200535874ce2fe42a8493abc0",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_quant",
        build_file = clean_dep("//third_party:tflite_mobilenet_quant.BUILD"),
        sha256 = "d32432d28673a936b2d6281ab0600c71cf7226dfe4cdcef3012555f691744166",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "767057f2837a46d97882734b03428e8dd640b93236052b312b2f0e45613c1cf0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_ssd_tflite_v1.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_ssd_tflite_v1.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd_quant",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "a809cd290b4d6a2e8a9d5dad076e0bd695b8091974e0eed1052b480b2f21b6dc",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_0.75_quant_2018_06_29.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_0.75_quant_2018_06_29.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd_quant_protobuf",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "09280972c5777f1aa775ef67cb4ac5d5ed21970acd8535aeca62450ef14f0d79",
        strip_prefix = "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz",
            "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz",
        ],
    )

    tf_http_archive(
        name = "tflite_conv_actions_frozen",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "d947b38cba389b5e2d0bfc3ea6cc49c784e187b41a071387b3742d1acac7691e",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_ovic_testdata",
        build_file = clean_dep("//third_party:tflite_ovic_testdata.BUILD"),
        sha256 = "033c941b7829b05ca55a124a26a6a0581b1ececc154a2153cafcfdb54f80dca2",
        strip_prefix = "ovic",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/data/ovic_2019_04_30.zip",
            "https://storage.googleapis.com/download.tensorflow.org/data/ovic_2019_04_30.zip",
        ],
    )

    tf_http_archive(
        name = "rules_cc",
        sha256 = "cf3b76a90c86c0554c5b10f4b160f05af71d252026b71362c4674e2fb9936cf9",
        strip_prefix = "rules_cc-01d4a48911d5e7591ecb1c06d3b8af47fe872371",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_cc/archive/01d4a48911d5e7591ecb1c06d3b8af47fe872371.zip",
            "https://github.com/bazelbuild/rules_cc/archive/01d4a48911d5e7591ecb1c06d3b8af47fe872371.zip",
        ],
    )

    tf_http_archive(
        name = "rules_python",
        sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
            "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "build_bazel_rules_android",
        sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
        strip_prefix = "rules_android-0.1.1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
            "https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
        ],
    )

    # Apple and Swift rules.
    # https://github.com/bazelbuild/rules_apple/releases
    tf_http_archive(
        name = "build_bazel_rules_apple",
        sha256 = "ee9e6073aeb5a65c100cb9c44b0017c937706a4ae03176e14a7e78620a198079",
        strip_prefix = "rules_apple-5131f3d46794bf227d296c82f30c2499c9de3c5b",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_apple/archive/5131f3d46794bf227d296c82f30c2499c9de3c5b.tar.gz",
            "https://github.com/bazelbuild/rules_apple/archive/5131f3d46794bf227d296c82f30c2499c9de3c5b.tar.gz",
        ],
    )

    # https://github.com/bazelbuild/rules_swift/releases
    tf_http_archive(
        name = "build_bazel_rules_swift",
        sha256 = "d0833bc6dad817a367936a5f902a0c11318160b5e80a20ece35fb85a5675c886",
        strip_prefix = "rules_swift-3eeeb53cebda55b349d64c9fc144e18c5f7c0eb8",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_swift/archive/3eeeb53cebda55b349d64c9fc144e18c5f7c0eb8.tar.gz",
            "https://github.com/bazelbuild/rules_swift/archive/3eeeb53cebda55b349d64c9fc144e18c5f7c0eb8.tar.gz",
        ],
    )

    # https://github.com/bazelbuild/apple_support/releases
    tf_http_archive(
        name = "build_bazel_apple_support",
        sha256 = "ad8ae80e93612b8151019367a3d1604d7a51c14480dae1254e10252007e8260c",
        strip_prefix = "apple_support-501b4afb27745c4813a88ffa28acd901408014e4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/apple_support/archive/501b4afb27745c4813a88ffa28acd901408014e4.tar.gz",
            "https://github.com/bazelbuild/apple_support/archive/501b4afb27745c4813a88ffa28acd901408014e4.tar.gz",
        ],
    )

    # https://github.com/bazelbuild/bazel-skylib/releases
    tf_http_archive(
        name = "bazel_skylib",
        sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        ],
    )

    # https://github.com/apple/swift-protobuf/releases
    tf_http_archive(
        name = "com_github_apple_swift_swift_protobuf",
        strip_prefix = "swift-protobuf-1.6.0/",
        sha256 = "4ccf6e5ea558e8287bf6331f9f6e52b3c321fca5f1d181d03680f415c32a6bba",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apple/swift-protobuf/archive/1.6.0.zip",
            "https://github.com/apple/swift-protobuf/archive/1.6.0.zip",
        ],
    )

    # https://github.com/google/xctestrunner/releases
    http_file(
        name = "xctestrunner",
        executable = 1,
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/xctestrunner/releases/download/0.2.9/ios_test_runner.par",
            "https://github.com/google/xctestrunner/releases/download/0.2.9/ios_test_runner.par",
        ],
    )

    tf_http_archive(
        name = "tbb",
        build_file = clean_dep("//third_party/ngraph:tbb.BUILD"),
        sha256 = "c3245012296f09f1418b78a8c2f17df5188b3bd0db620f7fd5fabe363320805a",
        strip_prefix = "tbb-2019_U1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/01org/tbb/archive/2019_U1.zip",
            "https://github.com/01org/tbb/archive/2019_U1.zip",
        ],
    )

    tf_http_archive(
        name = "ngraph",
        build_file = clean_dep("//third_party/ngraph:ngraph.BUILD"),
        sha256 = "a1780f24a1381fc25e323b4b2d08b6ef5129f42e011305b2a34dcf43a48030d5",
        strip_prefix = "ngraph-0.11.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NervanaSystems/ngraph/archive/v0.11.0.tar.gz",
            "https://github.com/NervanaSystems/ngraph/archive/v0.11.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "nlohmann_json_lib",
        build_file = clean_dep("//third_party/ngraph:nlohmann_json.BUILD"),
        sha256 = "c377963a95989270c943d522bfefe7b889ef5ed0e1e15d535fd6f6f16ed70732",
        strip_prefix = "json-3.4.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nlohmann/json/archive/v3.4.0.tar.gz",
            "https://github.com/nlohmann/json/archive/v3.4.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "ngraph_tf",
        build_file = clean_dep("//third_party/ngraph:ngraph_tf.BUILD"),
        sha256 = "742a642d2c6622277df4c902b6830d616d0539cc8cd843d6cdb899bb99e66e36",
        strip_prefix = "ngraph-tf-0.9.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NervanaSystems/ngraph-tf/archive/v0.9.0.zip",
            "https://github.com/NervanaSystems/ngraph-tf/archive/v0.9.0.zip",
        ],
    )

    tf_http_archive(
        name = "pybind11",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
            "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
        ],
        sha256 = "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
        strip_prefix = "pybind11-2.4.3",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
        system_build_file = clean_dep("//third_party/systemlibs:pybind11.BUILD"),
    )

    tf_http_archive(
        name = "wrapt",
        build_file = clean_dep("//third_party:wrapt.BUILD"),
        sha256 = "8a6fb40e8f8b6a66b4ba81a4044c68e6a7b1782f21cfabc06fb765332b4c3e51",
        strip_prefix = "wrapt-1.11.1/src/wrapt",
        system_build_file = clean_dep("//third_party/systemlibs:wrapt.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/GrahamDumpleton/wrapt/archive/1.11.1.tar.gz",
            "https://github.com/GrahamDumpleton/wrapt/archive/1.11.1.tar.gz",
        ],
    )
    tf_http_archive(
        name = "coremltools",
        sha256 = "0d594a714e8a5fd5bd740ad112ef59155c0482e25fdc8f8efa5758f90abdcf1e",
        strip_prefix = "coremltools-3.3",
        build_file = clean_dep("//third_party:coremltools.BUILD"),
        urls = [
            "http://mirror.tensorflow.org/github.com/apple/coremltools/archive/3.3.zip",
            "https://github.com/apple/coremltools/archive/3.3.zip",
        ],
    )

def tf_bind():
    """Bind targets for some external repositories"""
    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@com_github_grpc_grpc//src/compiler:grpc_python_plugin",
    )

    native.bind(
        name = "grpc_lib",
        actual = "@com_github_grpc_grpc//:grpc++",
    )

    native.bind(
        name = "grpc_lib_unsecure",
        actual = "@com_github_grpc_grpc//:grpc++_unsecure",
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
