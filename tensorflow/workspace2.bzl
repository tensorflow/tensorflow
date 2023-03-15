"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

# Import third party config rules.
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
load("@bazel_skylib//lib:versions.bzl", "versions")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//third_party/nccl:nccl_configure.bzl", "nccl_configure")
load("//third_party/git:git_configure.bzl", "git_configure")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("//tensorflow/tools/toolchains:cpus/aarch64/aarch64_compiler_configure.bzl", "aarch64_compiler_configure")
load("//tensorflow/tools/toolchains:cpus/arm/arm_compiler_configure.bzl", "arm_compiler_configure")
load("//tensorflow/tools/toolchains/embedded/arm-linux:arm_linux_toolchain_configure.bzl", "arm_linux_toolchain_configure")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/clang_toolchain:cc_configure_clang.bzl", "cc_download_clang_toolchain")
load("//tensorflow/tools/def_file_filter:def_file_filter_configure.bzl", "def_file_filter_configure")
load("//third_party/llvm:setup.bzl", "llvm_setup")

# Import third party repository rules. See go/tfbr-thirdparty.
load("//third_party/FP16:workspace.bzl", FP16 = "repo")
load("//third_party/absl:workspace.bzl", absl = "repo")
load("//third_party/benchmark:workspace.bzl", benchmark = "repo")
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
load("//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("//third_party/vulkan_headers:workspace.bzl", vulkan_headers = "repo")
load("//third_party/tensorrt:workspace.bzl", tensorrt = "repo")
load("//third_party/triton:workspace.bzl", triton = "repo")

# Import external repository rules.
load("@bazel_tools//tools/build_defs/repo:java.bzl", "java_import_external")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")
load("@tf_runtime//:dependencies.bzl", "tfrt_dependencies")
load("//tensorflow/tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")
load("//tensorflow/tools/toolchains/remote:configure.bzl", "remote_execution_configure")
load("//tensorflow/tools/toolchains/clang6:repo.bzl", "clang6_configure")
load("@rules_jvm_external//:defs.bzl", "maven_install")

def _initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    FP16()
    absl()
    bazel_skylib_workspace()
    benchmark()
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
    stablehlo()
    vulkan_headers()
    tensorrt()
    triton()

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
        build_file = "//tensorflow/tools/toolchains/cpus/arm:template.BUILD",
        remote_config_repo_arm = "../arm_compiler",
        remote_config_repo_aarch64 = "../aarch64_compiler",
    )

    # Load aarch64 toolchain
    aarch64_compiler_configure()

    # TFLite crossbuild toolchain for embeddeds Linux
    arm_linux_toolchain_configure(
        name = "local_config_embedded_arm",
        build_file = "//tensorflow/tools/toolchains/embedded/arm-linux:template.BUILD",
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
        sha256 = "4d047b1ba16e00740aa32f95cc80b40329524bfa175844f9fc61891acc912982",
        strip_prefix = "XNNPACK-06b2705f1b3e1ba0f161dd2979e2901ce93014e3",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/06b2705f1b3e1ba0f161dd2979e2901ce93014e3.zip"),
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
        sha256 = "b96413b10dd8edaa4f6c0a60c6cf5ef55eebeef78164d5d69294c8173457f0ec",
        strip_prefix = "pthreadpool-b8374f80e42010941bda6c85b0e3f1a1bd77a1e0",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/pthreadpool/archive/b8374f80e42010941bda6c85b0e3f1a1bd77a1e0.zip"),
    )

    tf_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-3dc310302210c1891ffcfb12ae67b11a3ad3a150",
        sha256 = "ba668f9f8ea5b4890309b7db1ed2e152aaaf98af6f9a8a63dbe1b75c04e52cb9",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/3dc310302210c1891ffcfb12ae67b11a3ad3a150.zip"),
    )

    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "3c7b842cd67989810955b220fa1116e7e2ed10660a8cfb632118146a64992c30",
        strip_prefix = "cudnn-frontend-0.7.3",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v0.7.3.zip"),
    )

    tf_http_archive(
        name = "mkl_dnn",
        build_file = "//third_party/mkl_dnn:mkldnn.BUILD",
        sha256 = "a0211aeb5e7dad50b97fa5dffc1a2fe2fe732572d4164e1ee8750a2ede43fbec",
        strip_prefix = "oneDNN-0.21.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/v0.21.3.tar.gz"),
    )

    tf_http_archive(
        name = "mkl_dnn_v1",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        sha256 = "a50993aa6265b799b040fe745e0010502f9f7103cc53a9525d59646aef006633",
        strip_prefix = "oneDNN-2.7.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v2.7.3.tar.gz"),
    )

    tf_http_archive(
        name = "mkl_dnn_acl_compatible",
        build_file = "//third_party/mkl_dnn:mkldnn_acl.BUILD",
        patch_file = [
            "//third_party/mkl_dnn:onednn_acl_threadcap.patch",
            "//third_party/mkl_dnn:onednn_acl_fixed_format_kernels.patch",
            "//third_party/mkl_dnn:onednn_acl_depthwise_convolution.patch",
            "//third_party/mkl_dnn:onednn_acl_threadpool_scheduler.patch",
        ],
        sha256 = "a50993aa6265b799b040fe745e0010502f9f7103cc53a9525d59646aef006633",
        strip_prefix = "oneDNN-2.7.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/v2.7.3.tar.gz"),
    )

    tf_http_archive(
        name = "compute_library",
        sha256 = "e20a060d3c4f803889d96c2f0b865004ba3ef4e228299a44339ea1c1ba827c85",
        strip_prefix = "ComputeLibrary-22.11",
        build_file = "//third_party/compute_library:BUILD",
        patch_file = ["//third_party/compute_library:compute_library.patch", "//third_party/compute_library:acl_fixed_format_kernels_striding.patch", "//third_party/compute_library:acl_openmp_fix.patch"],
        urls = tf_mirror_urls("https://github.com/ARM-software/ComputeLibrary/archive/v22.11.tar.gz"),
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
        build_file = "//tensorflow/tools/toolchains/embedded/arm-linux:aarch64-linux-toolchain.BUILD",
        sha256 = "50cdef6c5baddaa00f60502cc8b59cc11065306ae575ad2f51e412a9b2a90364",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"),
    )

    tf_http_archive(
        name = "armhf_linux_toolchain",
        build_file = "//tensorflow/tools/toolchains/embedded/arm-linux:armhf-linux-toolchain.BUILD",
        sha256 = "3f76650b1d048036473b16b647b8fd005ffccd1a2869c10994967e0e49f26ac2",
        strip_prefix = "arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-linux-gnueabihf",
        urls = tf_mirror_urls("https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-arm-none-linux-gnueabihf.tar.xz"),
    )

    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "b90430b2a9240df4459108b3e291be80ae92c68a47bc06ef2dc419c5724de061",
        strip_prefix = "re2-a276a8c738735a0fe45a6ee590fe2df69bcf4502",
        system_build_file = "//third_party/systemlibs:re2.BUILD",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/a276a8c738735a0fe45a6ee590fe2df69bcf4502.tar.gz"),
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

    tf_http_archive(
        name = "com_github_googlecloudplatform_tensorflow_gcp_tools",
        sha256 = "5e9ebe17eaa2895eb7f77fefbf52deeda7c4b63f5a616916b823eb74f3a0c542",
        strip_prefix = "tensorflow-gcp-tools-2643d8caeba6ca2a6a0b46bb123953cb95b7e7d5",
        urls = tf_mirror_urls("https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools/archive/2643d8caeba6ca2a6a0b46bb123953cb95b7e7d5.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_googleapis",
        build_file = "//third_party/googleapis:googleapis.BUILD",
        sha256 = "249d83abc5d50bf372c35c49d77f900bff022b2c21eb73aa8da1458b6ac401fc",
        strip_prefix = "googleapis-6b3fdcea8bc5398be4e7e9930c693f0ea09316a0",
        urls = tf_mirror_urls("https://github.com/googleapis/googleapis/archive/6b3fdcea8bc5398be4e7e9930c693f0ea09316a0.tar.gz"),
    )

    tf_http_archive(
        name = "png",
        build_file = "//third_party:png.BUILD",
        patch_file = ["//third_party:png_fix_rpi.patch"],
        sha256 = "a00e9d2f2f664186e4202db9299397f851aea71b36a35e74910b8820e380d441",
        strip_prefix = "libpng-1.6.39",
        system_build_file = "//third_party/systemlibs:png.BUILD",
        urls = tf_mirror_urls("https://github.com/glennrp/libpng/archive/v1.6.39.tar.gz"),
    )

    tf_http_archive(
        name = "org_sqlite",
        build_file = "//third_party:sqlite.BUILD",
        sha256 = "49112cc7328392aa4e3e5dae0b2f6736d0153430143d21f69327788ff4efe734",
        strip_prefix = "sqlite-amalgamation-3400100",
        system_build_file = "//third_party/systemlibs:sqlite.BUILD",
        urls = tf_mirror_urls("https://www.sqlite.org/2022/sqlite-amalgamation-3400100.zip"),
    )

    tf_http_archive(
        name = "gif",
        build_file = "//third_party:gif.BUILD",
        patch_file = ["//third_party:gif_fix_strtok_r.patch"],
        sha256 = "31da5562f44c5f15d63340a09a4fd62b48c45620cd302f77a6d9acf0077879bd",
        strip_prefix = "giflib-5.2.1",
        system_build_file = "//third_party/systemlibs:gif.BUILD",
        urls = tf_mirror_urls("https://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz"),
    )

    tf_http_archive(
        name = "six_archive",
        build_file = "//third_party:six.BUILD",
        sha256 = "1e61c37477a1626458e36f7b1d82aa5c9b094fa4802892072e49de9c60c4c926",
        strip_prefix = "six-1.16.0",
        system_build_file = "//third_party/systemlibs:six.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/source/s/six/six-1.16.0.tar.gz"),
    )

    tf_http_archive(
        name = "astor_archive",
        build_file = "//third_party:astor.BUILD",
        sha256 = "95c30d87a6c2cf89aa628b87398466840f0ad8652f88eb173125a6df8533fb8d",
        strip_prefix = "astor-0.7.1",
        system_build_file = "//third_party/systemlibs:astor.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/source/a/astor/astor-0.7.1.tar.gz"),
    )

    tf_http_archive(
        name = "astunparse_archive",
        build_file = "//third_party:astunparse.BUILD",
        sha256 = "5ad93a8456f0d084c3456d059fd9a92cce667963232cbf763eac3bc5b7940872",
        strip_prefix = "astunparse-1.6.3/lib",
        system_build_file = "//third_party/systemlibs:astunparse.BUILD",
        urls = tf_mirror_urls("https://files.pythonhosted.org/packages/f3/af/4182184d3c338792894f34a62672919db7ca008c89abee9b564dd34d8029/astunparse-1.6.3.tar.gz"),
    )

    filegroup_external(
        name = "astunparse_license",
        licenses = ["notice"],  # PSFL
        sha256_urls = {
            "92fc0e4f4fa9460558eedf3412b988d433a2dcbb3a9c45402a145a4fab8a6ac6": tf_mirror_urls("https://raw.githubusercontent.com/simonpercivall/astunparse/v1.6.2/LICENSE"),
        },
    )

    tf_http_archive(
        name = "functools32_archive",
        build_file = "//third_party:functools32.BUILD",
        sha256 = "f6253dfbe0538ad2e387bd8fdfd9293c925d63553f5813c4e587745416501e6d",
        strip_prefix = "functools32-3.2.3-2",
        system_build_file = "//third_party/systemlibs:functools32.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/c5/60/6ac26ad05857c601308d8fb9e87fa36d0ebf889423f47c3502ef034365db/functools32-3.2.3-2.tar.gz"),
    )

    tf_http_archive(
        name = "gast_archive",
        build_file = "//third_party:gast.BUILD",
        sha256 = "40feb7b8b8434785585ab224d1568b857edb18297e5a3047f1ba012bc83b42c1",
        strip_prefix = "gast-0.4.0",
        system_build_file = "//third_party/systemlibs:gast.BUILD",
        urls = tf_mirror_urls("https://files.pythonhosted.org/packages/83/4a/07c7e59cef23fb147454663c3271c21da68ba2ab141427c20548ae5a8a4d/gast-0.4.0.tar.gz"),
    )

    tf_http_archive(
        name = "termcolor_archive",
        build_file = "//third_party:termcolor.BUILD",
        sha256 = "1d6d69ce66211143803fbc56652b41d73b4a400a2891d7bf7a1cdf4c02de613b",
        strip_prefix = "termcolor-1.1.0",
        system_build_file = "//third_party/systemlibs:termcolor.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz"),
    )

    tf_http_archive(
        name = "typing_extensions_archive",
        build_file = "//third_party:typing_extensions.BUILD",
        sha256 = "f1c24655a0da0d1b67f07e17a5e6b2a105894e6824b92096378bb3668ef02376",
        strip_prefix = "typing_extensions-4.2.0/src",
        system_build_file = "//third_party/systemlibs:typing_extensions.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/source/t/typing_extensions/typing_extensions-4.2.0.tar.gz"),
    )

    filegroup_external(
        name = "typing_extensions_license",
        licenses = ["notice"],  # PSFL
        sha256_urls = {
            "ff17ce94e102024deb68773eb1cc74ca76da4e658f373531f0ac22d68a6bb1ad": tf_mirror_urls("https://raw.githubusercontent.com/python/typing/master/typing_extensions/LICENSE"),
        },
    )

    tf_http_archive(
        name = "opt_einsum_archive",
        build_file = "//third_party:opt_einsum.BUILD",
        sha256 = "d3d464b4da7ef09e444c30e4003a27def37f85ff10ff2671e5f7d7813adac35b",
        strip_prefix = "opt_einsum-2.3.2",
        system_build_file = "//third_party/systemlibs:opt_einsum.BUILD",
        urls = tf_mirror_urls("https://pypi.python.org/packages/f6/d6/44792ec668bcda7d91913c75237314e688f70415ab2acd7172c845f0b24f/opt_einsum-2.3.2.tar.gz"),
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

    tf_http_archive(
        name = "dill_archive",
        build_file = "//third_party:dill.BUILD",
        system_build_file = "//third_party/systemlibs:dill.BUILD",
        urls = tf_mirror_urls("https://github.com/uqfoundation/dill/releases/download/dill-0.3.6/dill-0.3.6.zip"),
        sha256 = "2159ca9e7568ff47dc7be2e35a6edf18014351da95ad1b59c0930a14dcf37be7",
        strip_prefix = "dill-0.3.6",
    )

    tf_http_archive(
        name = "tblib_archive",
        build_file = "//third_party:tblib.BUILD",
        system_build_file = "//third_party/systemlibs:tblib.BUILD",
        urls = tf_mirror_urls("https://files.pythonhosted.org/packages/d3/41/901ef2e81d7b1e834b9870d416cb09479e175a2be1c4aa1a9dcd0a555293/tblib-1.7.0.tar.gz"),
        sha256 = "059bd77306ea7b419d4f76016aef6d7027cc8a0785579b5aad198803435f882c",
        strip_prefix = "tblib-1.7.0",
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
        name = "nsync",
        patch_file = ["//third_party:nsync.patch"],
        sha256 = "2be9dbfcce417c7abcc2aa6fee351cd4d292518d692577e74a2c6c05b049e442",
        strip_prefix = "nsync-1.25.0",
        system_build_file = "//third_party/systemlibs:nsync.BUILD",
        urls = tf_mirror_urls("https://github.com/google/nsync/archive/1.25.0.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_googletest",
        sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
        strip_prefix = "googletest-release-1.12.1",
        urls = tf_mirror_urls("https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"),
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
        build_file = "//third_party:curl.BUILD",
        sha256 = "dfb8582a05a893e305783047d791ffef5e167d295cf8d12b9eb9cfa0991ca5a9",
        strip_prefix = "curl-7.88.0",
        system_build_file = "//third_party/systemlibs:curl.BUILD",
        urls = tf_mirror_urls("https://curl.haxx.se/download/curl-7.88.0.tar.gz"),
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
            "//third_party/systemlibs:BUILD": "bazel/BUILD",
            "//third_party/systemlibs:grpc.BUILD": "src/compiler/BUILD",
            "//third_party/systemlibs:grpc.bazel.grpc_deps.bzl": "bazel/grpc_deps.bzl",
            "//third_party/systemlibs:grpc.bazel.grpc_extra_deps.bzl": "bazel/grpc_extra_deps.bzl",
            "//third_party/systemlibs:grpc.bazel.cc_grpc_library.bzl": "bazel/cc_grpc_library.bzl",
            "//third_party/systemlibs:grpc.bazel.generate_cc.bzl": "bazel/generate_cc.bzl",
            "//third_party/systemlibs:grpc.bazel.protobuf.bzl": "bazel/protobuf.bzl",
        },
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz"),
    )

    tf_http_archive(
        name = "linenoise",
        build_file = "//third_party:linenoise.BUILD",
        sha256 = "b35a74dbc9cd2fef9e4d56222761d61daf7e551510e6cd1a86f0789b548d074e",
        strip_prefix = "linenoise-4ce393a66b10903a0ef52edf9775ed526a17395f",
        urls = tf_mirror_urls("https://github.com/antirez/linenoise/archive/4ce393a66b10903a0ef52edf9775ed526a17395f.tar.gz"),
    )

    llvm_setup(name = "llvm-project")

    # Intel openMP that is part of LLVM sources.
    tf_http_archive(
        name = "llvm_openmp",
        build_file = "//third_party/llvm_openmp:BUILD",
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
        name = "boringssl",
        sha256 = "9dc53f851107eaf87b391136d13b815df97ec8f76dadb487b58b2fc45e624d2c",
        strip_prefix = "boringssl-c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc",
        system_build_file = "//third_party/systemlibs:boringssl.BUILD",
        urls = tf_mirror_urls("https://github.com/google/boringssl/archive/c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc.tar.gz"),
    )

    # Note: if you update this, you have to update libpng too. See cl/437813808
    tf_http_archive(
        name = "zlib",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
        strip_prefix = "zlib-1.2.13",
        system_build_file = "//third_party/systemlibs:zlib.BUILD",
        urls = tf_mirror_urls("https://zlib.net/fossils/zlib-1.2.13.tar.gz"),
    )

    # LINT.IfChange
    tf_http_archive(
        name = "fft2d",
        build_file = "//third_party/fft2d:fft2d.BUILD",
        sha256 = "5f4dabc2ae21e1f537425d58a49cdca1c49ea11db0d6271e2a4b27e9697548eb",
        strip_prefix = "OouraFFT-1.0",
        urls = tf_mirror_urls("https://github.com/petewarden/OouraFFT/archive/v1.0.tar.gz"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/fft2d.cmake)

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
        sha256 = "0e3d7b6295beed81dc15002e88abf7a3b45b5c686b13b779ceac056f5612087f",
        strip_prefix = "nccl-2.16.5-1",
        urls = tf_mirror_urls("https://github.com/nvidia/nccl/archive/v2.16.5-1.tar.gz"),
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
        sha256 = "b844b75c25cfe7ea34b832b369ab91234009b2dfe2ae1fcea53860c57253fe2e",
        strip_prefix = "pprof-83db2b799d1f74c40857232cb5eb4c60379fe6c2",
        urls = tf_mirror_urls("https://github.com/google/pprof/archive/83db2b799d1f74c40857232cb5eb4c60379fe6c2.tar.gz"),
    )

    # The CUDA 11 toolkit ships with CUB.  We should be able to delete this rule
    # once TF drops support for CUDA 10.
    tf_http_archive(
        name = "cub_archive",
        build_file = "//third_party:cub.BUILD",
        sha256 = "162514b3cc264ac89d91898b58450190b8192e2af1142cf8ccac2d59aa160dda",
        strip_prefix = "cub-1.9.9",
        urls = tf_mirror_urls("https://github.com/NVlabs/cub/archive/1.9.9.zip"),
    )

    tf_http_archive(
        name = "nvtx_archive",
        build_file = "//third_party:nvtx.BUILD",
        sha256 = "bb8d1536aad708ec807bc675e12e5838c2f84481dec4005cd7a9bbd49e326ba1",
        strip_prefix = "NVTX-3.0.1/c/include",
        urls = tf_mirror_urls("https://github.com/NVIDIA/NVTX/archive/v3.0.1.tar.gz"),
    )

    tf_http_archive(
        name = "cython",
        build_file = "//third_party:cython.BUILD",
        sha256 = "08dbdb6aa003f03e65879de8f899f87c8c718cd874a31ae9c29f8726da2f5ab0",
        strip_prefix = "cython-3.0.0a11",
        system_build_file = "//third_party/systemlibs:cython.BUILD",
        urls = tf_mirror_urls("https://github.com/cython/cython/archive/3.0.0a11.tar.gz"),
    )

    # LINT.IfChange
    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = "//third_party:arm_neon_2_x86_sse.BUILD",
        sha256 = "019fbc7ec25860070a1d90e12686fc160cfb33e22aa063c80f52b363f1361e9d",
        strip_prefix = "ARM_NEON_2_x86_SSE-a15b489e1222b2087007546b4912e21293ea86ff",
        urls = tf_mirror_urls("https://github.com/intel/ARM_NEON_2_x86_SSE/archive/a15b489e1222b2087007546b4912e21293ea86ff.tar.gz"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/neon2sse.cmake)

    tf_http_archive(
        name = "double_conversion",
        sha256 = "3dbcdf186ad092a8b71228a5962009b5c96abde9a315257a3452eb988414ea3b",
        strip_prefix = "double-conversion-3.2.0",
        system_build_file = "//third_party/systemlibs:double_conversion.BUILD",
        urls = tf_mirror_urls("https://github.com/google/double-conversion/archive/v3.2.0.tar.gz"),
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
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz"),
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
        sha256 = "36072d4f3614d309d6a703da0dfe48684ec4c65a89611aeb9590b45af7a3e592",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_apple/releases/download/1.0.1/rules_apple.1.0.1.tar.gz"),
    )

    # https://github.com/bazelbuild/rules_swift/releases
    tf_http_archive(
        name = "build_bazel_rules_swift",
        sha256 = "12057b7aa904467284eee640de5e33853e51d8e31aae50b3fb25d2823d51c6b8",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_swift/releases/download/1.0.0/rules_swift.1.0.0.tar.gz"),
    )

    # https://github.com/bazelbuild/apple_support/releases
    tf_http_archive(
        name = "build_bazel_apple_support",
        sha256 = "ce1042cf936540eaa7b49c4549d7cd9b6b1492acbb6e765840a67a34b8e17a97",
        urls = tf_mirror_urls("https://github.com/bazelbuild/apple_support/releases/download/1.1.0/apple_support.1.1.0.tar.gz"),
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
        urls = tf_mirror_urls("https://github.com/pybind/pybind11/archive/v2.10.0.tar.gz"),
        sha256 = "eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec",
        strip_prefix = "pybind11-2.10.0",
        build_file = "//third_party:pybind11.BUILD",
        system_build_file = "//third_party/systemlibs:pybind11.BUILD",
    )

    tf_http_archive(
        name = "pybind11_protobuf",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_protobuf/archive/80f3440cd8fee124e077e2e47a8a17b78b451363.zip"),
        sha256 = "c7ab64b1ccf9a678694a89035a8c865a693e4e872803778f91f0965c2f281d78",
        strip_prefix = "pybind11_protobuf-80f3440cd8fee124e077e2e47a8a17b78b451363",
        patch_file = ["//third_party/pybind11_protobuf:remove_license.patch"],
    )

    tf_http_archive(
        name = "wrapt",
        build_file = "//third_party:wrapt.BUILD",
        sha256 = "866211ed43c2639a2452cd017bd38589e83687b1d843817c96b99d2d9d32e8d7",
        strip_prefix = "wrapt-1.14.1/src/wrapt",
        system_build_file = "//third_party/systemlibs:wrapt.BUILD",
        urls = tf_mirror_urls("https://github.com/GrahamDumpleton/wrapt/archive/1.14.1.tar.gz"),
    )

    tf_http_archive(
        name = "coremltools",
        sha256 = "89bb0bd2c16e19932670838dd5a8b239cd5c0a42338c72239d2446168c467a08",
        strip_prefix = "coremltools-5.2",
        build_file = "//third_party:coremltools.BUILD",
        urls = tf_mirror_urls("https://github.com/apple/coremltools/archive/5.2.tar.gz"),
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
        name = "com_google_ortools",
        sha256 = "b87922b75bbcce9b2ab5da0221751a3c8c0bff54b2a1eafa951dbf70722a640e",
        strip_prefix = "or-tools-7.3",
        patch_file = ["//third_party/ortools:ortools.patch"],
        urls = tf_mirror_urls("https://github.com/google/or-tools/archive/v7.3.tar.gz"),
        repo_mapping = {"@com_google_protobuf_cc": "@com_google_protobuf"},
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
