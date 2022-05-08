"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

# Import third party config rules.
load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//third_party/nccl:nccl_configure.bzl", "nccl_configure")
load("//third_party/git:git_configure.bzl", "git_configure")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("@tf_toolchains//toolchains/cpus/arm:arm_compiler_configure.bzl", "arm_compiler_configure")
load("@tf_toolchains//toolchains/embedded/arm-linux:arm_linux_toolchain_configure.bzl", "arm_linux_toolchain_configure")
load("//third_party:repo.bzl", "tf_http_archive")
load("//third_party/clang_toolchain:cc_configure_clang.bzl", "cc_download_clang_toolchain")
load("//tensorflow/tools/def_file_filter:def_file_filter_configure.bzl", "def_file_filter_configure")
load("//third_party/llvm:setup.bzl", "llvm_setup")

# Import third party repository rules. See go/tfbr-thirdparty.
load("//third_party/FP16:workspace.bzl", FP16 = "repo")
load("//third_party/absl:workspace.bzl", absl = "repo")
load("//third_party/benchmark:workspace.bzl", benchmark = "repo")
load("//third_party/clog:workspace.bzl", clog = "repo")
load("//third_party/cpuinfo:workspace.bzl", cpuinfo = "repo")
load("//third_party/dlpack:workspace.bzl", dlpack = "repo")
load("//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("//third_party/gemmlowp:workspace.bzl", gemmlowp = "repo")
load("//third_party/hexagon:workspace.bzl", hexagon_nn = "repo")
load("//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("//third_party/icu:workspace.bzl", icu = "repo")
load("//third_party/jpeg:workspace.bzl", jpeg = "repo")
load("//third_party/libprotobuf_mutator:workspace.bzl", libprotobuf_mutator = "repo")
load("//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party/pybind11_abseil:workspace.bzl", pybind11_abseil = "repo")
load("//third_party/opencl_headers:workspace.bzl", opencl_headers = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/pasta:workspace.bzl", pasta = "repo")
load("//third_party/psimd:workspace.bzl", psimd = "repo")
load("//third_party/ruy:workspace.bzl", ruy = "repo")
load("//third_party/sobol_data:workspace.bzl", sobol_data = "repo")
load("//third_party/vulkan_headers:workspace.bzl", vulkan_headers = "repo")

# Import external repository rules.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("@bazel_tools//tools/build_defs/repo:java.bzl", "java_import_external")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")
load("@tf_runtime//:dependencies.bzl", "tfrt_dependencies")
load("@tf_toolchains//toolchains/remote_config:configs.bzl", "initialize_rbe_configs")
load("@tf_toolchains//toolchains/remote:configure.bzl", "remote_execution_configure")
load("@tf_toolchains//toolchains/clang6:repo.bzl", "clang6_configure")

def _initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    FP16()
    absl()
    benchmark()
    clog()
    cpuinfo()
    dlpack()
    eigen3()
    farmhash()
    flatbuffers()
    gemmlowp()
    hexagon_nn()
    highwayhash()
    hwloc()
    icu()
    jpeg()
    kissfft()
    libprotobuf_mutator()
    nasm()
    opencl_headers()
    pasta()
    psimd()
    pybind11_abseil()
    ruy()
    sobol_data()
    vulkan_headers()

# Toolchains & platforms required by Tensorflow to build.
def _tf_toolchains():
    native.register_execution_platforms("@local_execution_config_platform//:platform")
    native.register_toolchains("@local_execution_config_python//:py_toolchain")

    # Loads all external repos to configure RBE builds.
    initialize_rbe_configs()

    # Note that we check the minimum bazel version in WORKSPACE.
    clang6_configure(name = "local_config_clang6")
    cc_download_clang_toolchain(name = "local_config_download_clang")
    cuda_configure(name = "local_config_cuda")
    tensorrt_configure(name = "local_config_tensorrt")
    nccl_configure(name = "local_config_nccl")
    git_configure(name = "local_config_git")
    syslibs_configure(name = "local_config_syslibs")
    python_configure(name = "local_config_python")
    rocm_configure(name = "local_config_rocm")
    remote_execution_configure(name = "local_config_remote_execution")

    # For windows bazel build
    # TODO: Remove def file filter when TensorFlow can export symbols properly on Windows.
    def_file_filter_configure(name = "local_config_def_file_filter")

    # Point //external/local_config_arm_compiler to //external/arm_compiler
    arm_compiler_configure(
        name = "local_config_arm_compiler",
        build_file = "@tf_toolchains//toolchains/cpus/arm:BUILD",
        remote_config_repo_arm = "../arm_compiler",
        remote_config_repo_aarch64 = "../aarch64_compiler",
    )

    # TFLite crossbuild toolchain for embeddeds Linux
    arm_linux_toolchain_configure(
        name = "local_config_embedded_arm",
        build_file = "@tf_toolchains//toolchains/embedded/arm-linux:BUILD",
        aarch64_repo = "../aarch64_linux_toolchain",
        armhf_repo = "../armhf_linux_toolchain",
    )

# Define all external repositories required by TensorFlow
def _tf_repositories():
    """All external dependencies for TF builds."""

    # To update any of the dependencies bellow:
    # a) update URL and strip_prefix to the new git commit hash
    # b) get the sha256 hash of the commit by running:
    #    curl -L <url> | sha256sum
    # and update the sha256 with the result.

    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "0a6ecf3c6fa38e4d4f655fec7e5d33afc9c8a91c8a437033c61f52c5189f3b9b",
        strip_prefix = "XNNPACK-694d2524757f9040e65a02c374e152a462fe57eb",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/XNNPACK/archive/694d2524757f9040e65a02c374e152a462fe57eb.zip",
            "https://github.com/google/XNNPACK/archive/694d2524757f9040e65a02c374e152a462fe57eb.zip",
        ],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)

    tf_http_archive(
        name = "FXdiv",
        sha256 = "3d7b0e9c4c658a84376a1086126be02f9b7f753caa95e009d9ac38d11da444db",
        strip_prefix = "FXdiv-63058eff77e11aa15bf531df5dd34395ec3017c8",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip",
            "https://github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip",
        ],
    )

    tf_http_archive(
        name = "pthreadpool",
        sha256 = "b96413b10dd8edaa4f6c0a60c6cf5ef55eebeef78164d5d69294c8173457f0ec",
        strip_prefix = "pthreadpool-b8374f80e42010941bda6c85b0e3f1a1bd77a1e0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/pthreadpool/archive/b8374f80e42010941bda6c85b0e3f1a1bd77a1e0.zip",
            "https://github.com/Maratyszcza/pthreadpool/archive/b8374f80e42010941bda6c85b0e3f1a1bd77a1e0.zip",
        ],
    )

    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = "//third_party:cudnn_frontend_header_fix.patch",
        sha256 = "fdf4234e9c9c481b3b3a80ad404bc278e6ecb761c5574beb4d3a2cde4a9002ad",
        strip_prefix = "cudnn-frontend-73210a930333eaf66b42b01693bce7b70719c354",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVIDIA/cudnn-frontend/archive/73210a930333eaf66b42b01693bce7b70719c354.zip",
            "https://github.com/NVIDIA/cudnn-frontend/archive/73210a930333eaf66b42b01693bce7b70719c354.zip",
        ],
    )

    tf_http_archive(
        name = "mkl_dnn",
        build_file = "//third_party/mkl_dnn:mkldnn.BUILD",
        sha256 = "a0211aeb5e7dad50b97fa5dffc1a2fe2fe732572d4164e1ee8750a2ede43fbec",
        strip_prefix = "oneDNN-0.21.3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v0.21.3.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/v0.21.3.tar.gz",
        ],
    )

    tf_http_archive(
        name = "mkl_dnn_v1",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        sha256 = "930a2327c4fe4018a604f67ca3c0ebea01436aad7ce1f4e01d088732bd19d16f",
        strip_prefix = "oneDNN-2.4.1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v2.4.1.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/v2.4.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "mkl_dnn_acl_compatible",
        build_file = "//third_party/mkl_dnn:mkldnn_acl.BUILD",
        sha256 = "a7f2a4d80d5406d156dfc1d7b27d10b0af5ed061cf0b6197d3d12cddc6790fcb",
        strip_prefix = "oneDNN-2.4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v2.4.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/v2.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "compute_library",
        sha256 = "1c62d41be62c14c8ff196d6aaa9f9efe0597b82a923350d922e8cde217dd1d86",
        strip_prefix = "ComputeLibrary-21.08",
        build_file = "//third_party/compute_library:BUILD",
        patch_file = "//third_party/compute_library:compute_library.patch",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/ARM-software/ComputeLibrary/archive/v21.08.tar.gz",
            "https://github.com/ARM-software/ComputeLibrary/archive/v21.08.tar.gz",
        ],
    )

    tf_http_archive(
        name = "arm_compiler",
        build_file = "//:arm_compiler.BUILD",
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
        build_file = "@tf_toolchains//toolchains/embedded/arm-linux:aarch64-linux-toolchain.BUILD",
        sha256 = "8ce3e7688a47d8cd2d8e8323f147104ae1c8139520eca50ccf8a7fa933002731",
        strip_prefix = "gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz",
            "https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz",
        ],
    )

    tf_http_archive(
        name = "armhf_linux_toolchain",
        build_file = "@tf_toolchains//toolchains/embedded/arm-linux:armhf-linux-toolchain.BUILD",
        sha256 = "d4f6480ecaa99e977e3833cc8a8e1263f9eecd1ce2d022bb548a24c4f32670f5",
        strip_prefix = "gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz",
            "https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz",
        ],
    )

    tf_http_archive(
        name = "libxsmm_archive",
        build_file = "//third_party:libxsmm.BUILD",
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
        system_build_file = "//third_party/systemlibs:re2.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
            "https://github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_github_google_crc32c",
        sha256 = "6b3b1d861bb8307658b2407bc7a4c59e566855ef5368a60b35c893551e4788e9",
        build_file = "@com_github_googlecloudplatform_google_cloud_cpp//bazel:crc32c.BUILD",
        strip_prefix = "crc32c-1.0.6",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/crc32c/archive/1.0.6.tar.gz",
            "https://github.com/google/crc32c/archive/1.0.6.tar.gz",
        ],
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
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/googleapis/google-cloud-cpp/archive/v1.17.1.tar.gz",
            "https://github.com/googleapis/google-cloud-cpp/archive/v1.17.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_github_googlecloudplatform_tensorflow_gcp_tools",
        sha256 = "5e9ebe17eaa2895eb7f77fefbf52deeda7c4b63f5a616916b823eb74f3a0c542",
        strip_prefix = "tensorflow-gcp-tools-2643d8caeba6ca2a6a0b46bb123953cb95b7e7d5",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/GoogleCloudPlatform/tensorflow-gcp-tools/archive/2643d8caeba6ca2a6a0b46bb123953cb95b7e7d5.tar.gz",
            "https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools/archive/2643d8caeba6ca2a6a0b46bb123953cb95b7e7d5.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_googleapis",
        build_file = "//third_party/googleapis:googleapis.BUILD",
        sha256 = "7ebab01b06c555f4b6514453dc3e1667f810ef91d1d4d2d3aa29bb9fcb40a900",
        strip_prefix = "googleapis-541b1ded4abadcc38e8178680b0677f65594ea6f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/googleapis/googleapis/archive/541b1ded4abadcc38e8178680b0677f65594ea6f.zip",
            "https://github.com/googleapis/googleapis/archive/541b1ded4abadcc38e8178680b0677f65594ea6f.zip",
        ],
    )

    tf_http_archive(
        name = "png",
        build_file = "//third_party:png.BUILD",
        patch_file = "//third_party:png_fix_rpi.patch",
        sha256 = "ca74a0dace179a8422187671aee97dd3892b53e168627145271cad5b5ac81307",
        strip_prefix = "libpng-1.6.37",
        system_build_file = "//third_party/systemlibs:png.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
            "https://github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
        ],
    )

    tf_http_archive(
        name = "org_sqlite",
        build_file = "//third_party:sqlite.BUILD",
        sha256 = "999826fe4c871f18919fdb8ed7ec9dd8217180854dd1fe21eea96aed36186729",
        strip_prefix = "sqlite-amalgamation-3360000",
        system_build_file = "//third_party/systemlibs:sqlite.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/sqlite.org/2021/sqlite-amalgamation-3360000.zip",
            "https://www.sqlite.org/2021/sqlite-amalgamation-3360000.zip",
        ],
    )

    tf_http_archive(
        name = "gif",
        build_file = "//third_party:gif.BUILD",
        patch_file = "//third_party:gif_fix_strtok_r.patch",
        sha256 = "31da5562f44c5f15d63340a09a4fd62b48c45620cd302f77a6d9acf0077879bd",
        strip_prefix = "giflib-5.2.1",
        system_build_file = "//third_party/systemlibs:gif.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz",
            "https://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "six_archive",
        build_file = "//third_party:six.BUILD",
        sha256 = "30639c035cdb23534cd4aa2dd52c3bf48f06e5f4a941509c8bafd8ce11080259",
        strip_prefix = "six-1.15.0",
        system_build_file = "//third_party/systemlibs:six.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.15.0.tar.gz",
            "https://pypi.python.org/packages/source/s/six/six-1.15.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "astor_archive",
        build_file = "//third_party:astor.BUILD",
        sha256 = "95c30d87a6c2cf89aa628b87398466840f0ad8652f88eb173125a6df8533fb8d",
        strip_prefix = "astor-0.7.1",
        system_build_file = "//third_party/systemlibs:astor.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/99/80/f9482277c919d28bebd85813c0a70117214149a96b08981b72b63240b84c/astor-0.7.1.tar.gz",
            "https://pypi.python.org/packages/99/80/f9482277c919d28bebd85813c0a70117214149a96b08981b72b63240b84c/astor-0.7.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "astunparse_archive",
        build_file = "//third_party:astunparse.BUILD",
        sha256 = "5ad93a8456f0d084c3456d059fd9a92cce667963232cbf763eac3bc5b7940872",
        strip_prefix = "astunparse-1.6.3/lib",
        system_build_file = "//third_party/systemlibs:astunparse.BUILD",
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
        build_file = "//third_party:functools32.BUILD",
        sha256 = "f6253dfbe0538ad2e387bd8fdfd9293c925d63553f5813c4e587745416501e6d",
        strip_prefix = "functools32-3.2.3-2",
        system_build_file = "//third_party/systemlibs:functools32.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/c5/60/6ac26ad05857c601308d8fb9e87fa36d0ebf889423f47c3502ef034365db/functools32-3.2.3-2.tar.gz",
            "https://pypi.python.org/packages/c5/60/6ac26ad05857c601308d8fb9e87fa36d0ebf889423f47c3502ef034365db/functools32-3.2.3-2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "gast_archive",
        build_file = "//third_party:gast.BUILD",
        sha256 = "40feb7b8b8434785585ab224d1568b857edb18297e5a3047f1ba012bc83b42c1",
        strip_prefix = "gast-0.4.0",
        system_build_file = "//third_party/systemlibs:gast.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/files.pythonhosted.org/packages/83/4a/07c7e59cef23fb147454663c3271c21da68ba2ab141427c20548ae5a8a4d/gast-0.4.0.tar.gz",
            "https://files.pythonhosted.org/packages/83/4a/07c7e59cef23fb147454663c3271c21da68ba2ab141427c20548ae5a8a4d/gast-0.4.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "termcolor_archive",
        build_file = "//third_party:termcolor.BUILD",
        sha256 = "1d6d69ce66211143803fbc56652b41d73b4a400a2891d7bf7a1cdf4c02de613b",
        strip_prefix = "termcolor-1.1.0",
        system_build_file = "//third_party/systemlibs:termcolor.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz",
            "https://pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "typing_extensions_archive",
        build_file = "//third_party:typing_extensions.BUILD",
        sha256 = "79ee589a3caca649a9bfd2a8de4709837400dfa00b6cc81962a1e6a1815969ae",
        strip_prefix = "typing_extensions-3.7.4.2/src_py3",
        system_build_file = "//third_party/systemlibs:typing_extensions.BUILD",
        urls = [
            "http://mirror.tensorflow.org/files.pythonhosted.org/packages/6a/28/d32852f2af6b5ead85d396249d5bdf450833f3a69896d76eb480d9c5e406/typing_extensions-3.7.4.2.tar.gz",
            "https://files.pythonhosted.org/packages/6a/28/d32852f2af6b5ead85d396249d5bdf450833f3a69896d76eb480d9c5e406/typing_extensions-3.7.4.2.tar.gz",
        ],
    )

    filegroup_external(
        name = "typing_extensions_license",
        licenses = ["notice"],  # PSFL
        sha256_urls = {
            "ff17ce94e102024deb68773eb1cc74ca76da4e658f373531f0ac22d68a6bb1ad": [
                "http://mirror.tensorflow.org/raw.githubusercontent.com/python/typing/master/typing_extensions/LICENSE",
                "https://raw.githubusercontent.com/python/typing/master/typing_extensions/LICENSE",
            ],
        },
    )

    tf_http_archive(
        name = "opt_einsum_archive",
        build_file = "//third_party:opt_einsum.BUILD",
        sha256 = "d3d464b4da7ef09e444c30e4003a27def37f85ff10ff2671e5f7d7813adac35b",
        strip_prefix = "opt_einsum-2.3.2",
        system_build_file = "//third_party/systemlibs:opt_einsum.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/f6/d6/44792ec668bcda7d91913c75237314e688f70415ab2acd7172c845f0b24f/opt_einsum-2.3.2.tar.gz",
            "https://pypi.python.org/packages/f6/d6/44792ec668bcda7d91913c75237314e688f70415ab2acd7172c845f0b24f/opt_einsum-2.3.2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "absl_py",
        sha256 = "588a23406b2e28ea368897dbebc1210165414e87212d4fdd4b2ee968f0a772c6",
        strip_prefix = "abseil-py-pypi-v0.10.0",
        system_build_file = "//third_party/systemlibs:absl_py.BUILD",
        system_link_files = {
            "//third_party/systemlibs:absl_py.absl.BUILD": "absl/BUILD",
            "//third_party/systemlibs:absl_py.absl.flags.BUILD": "absl/flags/BUILD",
            "//third_party/systemlibs:absl_py.absl.testing.BUILD": "absl/testing/BUILD",
            "//third_party/systemlibs:absl_py.absl.logging.BUILD": "absl/logging/BUILD",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-py/archive/pypi-v0.10.0.tar.gz",
            "https://github.com/abseil/abseil-py/archive/pypi-v0.10.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "enum34_archive",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz",
            "https://pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz",
        ],
        sha256 = "8ad8c4783bf61ded74527bffb48ed9b54166685e4230386a9ed9b1279e2df5b1",
        build_file = "//third_party:enum34.BUILD",
        system_build_file = "//third_party/systemlibs:enum34.BUILD",
        strip_prefix = "enum34-1.1.6/enum",
    )

    tf_http_archive(
        name = "dill_archive",
        build_file = "//third_party:dill.BUILD",
        system_build_file = "//third_party/systemlibs:dill.BUILD",
        urls = [
            "http://mirror.tensorflow.org/files.pythonhosted.org/packages/e2/96/518a8ea959a734b70d2e95fef98bcbfdc7adad1c1e5f5dd9148c835205a5/dill-0.3.2.zip",
            "https://files.pythonhosted.org/packages/e2/96/518a8ea959a734b70d2e95fef98bcbfdc7adad1c1e5f5dd9148c835205a5/dill-0.3.2.zip",
        ],
        sha256 = "6e12da0d8e49c220e8d6e97ee8882002e624f1160289ce85ec2cc0a5246b3a2e",
        strip_prefix = "dill-0.3.2",
    )

    tf_http_archive(
        name = "tblib_archive",
        build_file = "//third_party:tblib.BUILD",
        system_build_file = "//third_party/systemlibs:tblib.BUILD",
        urls = [
            "http://mirror.tensorflow.org/files.pythonhosted.org/packages/d3/41/901ef2e81d7b1e834b9870d416cb09479e175a2be1c4aa1a9dcd0a555293/tblib-1.7.0.tar.gz",
            "https://files.pythonhosted.org/packages/d3/41/901ef2e81d7b1e834b9870d416cb09479e175a2be1c4aa1a9dcd0a555293/tblib-1.7.0.tar.gz",
        ],
        sha256 = "059bd77306ea7b419d4f76016aef6d7027cc8a0785579b5aad198803435f882c",
        strip_prefix = "tblib-1.7.0",
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
        patch_file = "//third_party/protobuf:protobuf.patch",
        sha256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",
        strip_prefix = "protobuf-3.9.2",
        system_build_file = "//third_party/systemlibs:protobuf.BUILD",
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
            "//third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
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
        system_build_file = "//third_party/systemlibs:nsync.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
            "https://github.com/google/nsync/archive/1.22.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_googletest",
        sha256 = "bc1cc26d1120f5a7e9eb450751c0b24160734e46a02823a573f3c6b6c0a574a7",
        strip_prefix = "googletest-e2c06aa2497e330bab1c1a03d02f7c5096eb5b0b",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/e2c06aa2497e330bab1c1a03d02f7c5096eb5b0b.zip",
            "https://github.com/google/googletest/archive/e2c06aa2497e330bab1c1a03d02f7c5096eb5b0b.zip",
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
        name = "curl",
        build_file = "//third_party:curl.BUILD",
        sha256 = "c0e64302a33d2fb79e0fc4e674260a22941e92ee2f11b894bf94d32b8f5531af",
        strip_prefix = "curl-7.83.0",
        system_build_file = "//third_party/systemlibs:curl.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/curl.haxx.se/download/curl-7.83.0.tar.gz",
            "https://curl.haxx.se/download/curl-7.83.0.tar.gz",
        ],
    )

    # WARNING: make sure ncteisen@ and vpai@ are cc-ed on any CL to change the below rule
    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
        strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
        system_build_file = "//third_party/systemlibs:grpc.BUILD",
        patch_file = "//third_party/grpc:generate_cc_env_fix.patch",
        system_link_files = {
            "//third_party/systemlibs:BUILD": "bazel/BUILD",
            "//third_party/systemlibs:grpc.BUILD": "src/compiler/BUILD",
            "//third_party/systemlibs:grpc.bazel.grpc_deps.bzl": "bazel/grpc_deps.bzl",
            "//third_party/systemlibs:grpc.bazel.grpc_extra_deps.bzl": "bazel/grpc_extra_deps.bzl",
            "//third_party/systemlibs:grpc.bazel.cc_grpc_library.bzl": "bazel/cc_grpc_library.bzl",
            "//third_party/systemlibs:grpc.bazel.generate_cc.bzl": "bazel/generate_cc.bzl",
            "//third_party/systemlibs:grpc.bazel.protobuf.bzl": "bazel/protobuf.bzl",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
            "https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz",
        ],
    )

    tf_http_archive(
        name = "linenoise",
        build_file = "//third_party:linenoise.BUILD",
        sha256 = "7f51f45887a3d31b4ce4fa5965210a5e64637ceac12720cfce7954d6a2e812f7",
        strip_prefix = "linenoise-c894b9e59f02203dbe4e2be657572cf88c4230c3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
            "https://github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
        ],
    )

    llvm_setup(name = "llvm-project")

    # Intel openMP that is part of LLVM sources.
    tf_http_archive(
        name = "llvm_openmp",
        build_file = "//third_party/llvm_openmp:BUILD",
        sha256 = "d19f728c8e04fb1e94566c8d76aef50ec926cd2f95ef3bf1e0a5de4909b28b44",
        strip_prefix = "openmp-10.0.1.src",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz",
            "https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz",
        ],
    )

    tf_http_archive(
        name = "lmdb",
        build_file = "//third_party:lmdb.BUILD",
        sha256 = "22054926b426c66d8f2bc22071365df6e35f3aacf19ad943bc6167d4cae3bebb",
        strip_prefix = "lmdb-LMDB_0.9.29/libraries/liblmdb",
        system_build_file = "//third_party/systemlibs:lmdb.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/LMDB/lmdb/archive/LMDB_0.9.29.tar.gz",
            "https://github.com/LMDB/lmdb/archive/refs/tags/LMDB_0.9.29.tar.gz",
        ],
    )

    tf_http_archive(
        name = "jsoncpp_git",
        build_file = "//third_party:jsoncpp.BUILD",
        sha256 = "e34a628a8142643b976c7233ef381457efad79468c67cb1ae0b83a33d7493999",
        strip_prefix = "jsoncpp-1.9.4",
        system_build_file = "//third_party/systemlibs:jsoncpp.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/open-source-parsers/jsoncpp/archive/1.9.4.tar.gz",
            "https://github.com/open-source-parsers/jsoncpp/archive/1.9.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "boringssl",
        sha256 = "a9c3b03657d507975a32732f04563132b4553c20747cec6dc04de475c8bdf29f",
        strip_prefix = "boringssl-80ca9f9f6ece29ab132cce4cf807a9465a18cfac",
        system_build_file = "//third_party/systemlibs:boringssl.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/boringssl/archive/80ca9f9f6ece29ab132cce4cf807a9465a18cfac.tar.gz",
            "https://github.com/google/boringssl/archive/80ca9f9f6ece29ab132cce4cf807a9465a18cfac.tar.gz",
        ],
    )

    tf_http_archive(
        name = "zlib",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        system_build_file = "//third_party/systemlibs:zlib.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    # LINT.IfChange
    tf_http_archive(
        name = "fft2d",
        build_file = "//third_party/fft2d:fft2d.BUILD",
        sha256 = "5f4dabc2ae21e1f537425d58a49cdca1c49ea11db0d6271e2a4b27e9697548eb",
        strip_prefix = "OouraFFT-1.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/petewarden/OouraFFT/archive/v1.0.tar.gz",
            "https://github.com/petewarden/OouraFFT/archive/v1.0.tar.gz",
        ],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/fft2d.cmake)

    tf_http_archive(
        name = "snappy",
        build_file = "//third_party:snappy.BUILD",
        sha256 = "16b677f07832a612b0836178db7f374e414f94657c138e6993cbfc5dcc58651f",
        strip_prefix = "snappy-1.1.8",
        system_build_file = "//third_party/systemlibs:snappy.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/snappy/archive/1.1.8.tar.gz",
            "https://github.com/google/snappy/archive/1.1.8.tar.gz",
        ],
    )

    tf_http_archive(
        name = "nccl_archive",
        build_file = "//third_party:nccl/archive.BUILD",
        patch_file = "//third_party/nccl:archive.patch",
        sha256 = "3ae89ddb2956fff081e406a94ff54ae5e52359f5d645ce977c7eba09b3b782e6",
        strip_prefix = "nccl-2.8.3-1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nvidia/nccl/archive/v2.8.3-1.tar.gz",
            "https://github.com/nvidia/nccl/archive/v2.8.3-1.tar.gz",
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
        build_file = "//third_party:pprof.BUILD",
        sha256 = "e0928ca4aa10ea1e0551e2d7ce4d1d7ea2d84b2abbdef082b0da84268791d0c4",
        strip_prefix = "pprof-c0fb62ec88c411cc91194465e54db2632845b650",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
            "https://github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
        ],
    )

    # The CUDA 11 toolkit ships with CUB.  We should be able to delete this rule
    # once TF drops support for CUDA 10.
    tf_http_archive(
        name = "cub_archive",
        build_file = "//third_party:cub.BUILD",
        sha256 = "162514b3cc264ac89d91898b58450190b8192e2af1142cf8ccac2d59aa160dda",
        strip_prefix = "cub-1.9.9",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.9.9.zip",
            "https://github.com/NVlabs/cub/archive/1.9.9.zip",
        ],
    )

    tf_http_archive(
        name = "cython",
        build_file = "//third_party:cython.BUILD",
        sha256 = "e2e38e1f0572ca54d6085df3dec8b607d20e81515fb80215aed19c81e8fe2079",
        strip_prefix = "cython-0.29.21",
        system_build_file = "//third_party/systemlibs:cython.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/cython/cython/archive/0.29.21.tar.gz",
            "https://github.com/cython/cython/archive/0.29.21.tar.gz",
        ],
    )

    # LINT.IfChange
    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = "//third_party:arm_neon_2_x86_sse.BUILD",
        sha256 = "213733991310b904b11b053ac224fee2d4e0179e46b52fe7f8735b8831e04dcc",
        strip_prefix = "ARM_NEON_2_x86_SSE-1200fe90bb174a6224a525ee60148671a786a71f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
            "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
        ],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/neon2sse.cmake)

    tf_http_archive(
        name = "double_conversion",
        build_file = "//third_party:double_conversion.BUILD",
        sha256 = "2f7fbffac0d98d201ad0586f686034371a6d152ca67508ab611adc2386ad30de",
        strip_prefix = "double-conversion-3992066a95b823efc8ccc1baf82a1cfc73f6e9b8",
        system_build_file = "//third_party/systemlibs:double_conversion.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
            "https://github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_float",
        build_file = "//third_party:tflite_mobilenet_float.BUILD",
        sha256 = "2fadeabb9968ec6833bee903900dda6e61b3947200535874ce2fe42a8493abc0",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_quant",
        build_file = "//third_party:tflite_mobilenet_quant.BUILD",
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
        build_file = "//third_party:tflite_ovic_testdata.BUILD",
        sha256 = "033c941b7829b05ca55a124a26a6a0581b1ececc154a2153cafcfdb54f80dca2",
        strip_prefix = "ovic",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/data/ovic_2019_04_30.zip",
            "https://storage.googleapis.com/download.tensorflow.org/data/ovic_2019_04_30.zip",
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
        name = "nlohmann_json_lib",
        build_file = "//third_party:nlohmann_json.BUILD",
        sha256 = "c377963a95989270c943d522bfefe7b889ef5ed0e1e15d535fd6f6f16ed70732",
        strip_prefix = "json-3.4.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nlohmann/json/archive/v3.4.0.tar.gz",
            "https://github.com/nlohmann/json/archive/v3.4.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "pybind11",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.6.0.tar.gz",
            "https://github.com/pybind/pybind11/archive/v2.6.0.tar.gz",
        ],
        sha256 = "90b705137b69ee3b5fc655eaca66d0dc9862ea1759226f7ccd3098425ae69571",
        strip_prefix = "pybind11-2.6.0",
        build_file = "//third_party:pybind11.BUILD",
        patch_file = "//third_party:pybind11.patch",
        system_build_file = "//third_party/systemlibs:pybind11.BUILD",
    )

    tf_http_archive(
        name = "wrapt",
        build_file = "//third_party:wrapt.BUILD",
        sha256 = "8a6fb40e8f8b6a66b4ba81a4044c68e6a7b1782f21cfabc06fb765332b4c3e51",
        strip_prefix = "wrapt-1.11.1/src/wrapt",
        system_build_file = "//third_party/systemlibs:wrapt.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/GrahamDumpleton/wrapt/archive/1.11.1.tar.gz",
            "https://github.com/GrahamDumpleton/wrapt/archive/1.11.1.tar.gz",
        ],
    )
    tf_http_archive(
        name = "coremltools",
        sha256 = "0d594a714e8a5fd5bd740ad112ef59155c0482e25fdc8f8efa5758f90abdcf1e",
        strip_prefix = "coremltools-3.3",
        build_file = "//third_party:coremltools.BUILD",
        urls = [
            "http://mirror.tensorflow.org/github.com/apple/coremltools/archive/3.3.zip",
            "https://github.com/apple/coremltools/archive/3.3.zip",
        ],
    )

def workspace():
    # Check the bazel version before executing any repository rules, in case
    # those rules rely on the version we require here.
    check_bazel_version_at_least("1.0.0")

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
