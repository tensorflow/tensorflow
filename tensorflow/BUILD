# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "ACKNOWLEDGMENTS",
    # The leakr files are used by //third_party/cloud_tpu.
    "leakr_badwords.dic",
    "leakr_badfiles.dic",
    "leakr_file_type_recipe.ftrcp",
])

load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_additional_binary_deps",
)
load(
    "//tensorflow/python/tools/api/generator:api_gen.bzl",
    "gen_api_init_files",  # @unused
)
load("//tensorflow/python/tools/api/generator:api_gen.bzl", "get_compat_files")
load(
    "//tensorflow/python/tools/api/generator:api_init_files.bzl",
    "TENSORFLOW_API_INIT_FILES",  # @unused
)
load(
    "//tensorflow/python/tools/api/generator:api_init_files_v1.bzl",
    "TENSORFLOW_API_INIT_FILES_V1",  # @unused
)
load(
    "//third_party/ngraph:build_defs.bzl",
    "if_ngraph",
)

# @unused
TENSORFLOW_API_INIT_FILES_V2 = (
    TENSORFLOW_API_INIT_FILES + get_compat_files(TENSORFLOW_API_INIT_FILES_V1, 1)
)

# @unused
TENSORFLOW_API_INIT_FILES_V1_WITH_COMPAT = (
    TENSORFLOW_API_INIT_FILES_V1 + get_compat_files(TENSORFLOW_API_INIT_FILES_V1, 1)
)

# Config setting used when building for products
# which requires restricted licenses to be avoided.
config_setting(
    name = "no_lgpl_deps",
    values = {"define": "__TENSORFLOW_NO_LGPL_DEPS__=1"},
    visibility = ["//visibility:public"],
)

# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86_64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armeabi",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "raspberry_pi_armeabi",
    values = {
        "crosstool_top": "@local_config_arm_compiler//:toolchain",
        "cpu": "armeabi",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "mips",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "mips64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_tensorflow_py_deps",
    define_values = {"no_tensorflow_py_deps": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios",
    values = {"crosstool_top": "//tools/osx/crosstool:crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios_x86_64",
    values = {
        "crosstool_top": "//tools/osx/crosstool:crosstool",
        "cpu": "ios_x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc64le",
    values = {"cpu": "ppc"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_s390x",
    values = {"cpu": "s390x"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "optimized",
    values = {
        "compilation_mode": "opt",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm",
    values = {"cpu": "arm"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_default_optimizations",
    define_values = {"with_default_optimizations": "true"},
    visibility = ["//visibility:public"],
)

# Features that are default ON are handled differently below.
#
config_setting(
    name = "no_aws_support",
    define_values = {"no_aws_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_gcp_support",
    define_values = {"no_gcp_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_hdfs_support",
    define_values = {"no_hdfs_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_ignite_support",
    define_values = {"no_ignite_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_kafka_support",
    define_values = {"no_kafka_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_nccl_support",
    define_values = {"no_nccl_support": "true"},
    visibility = ["//visibility:public"],
)

# Crosses between platforms and file system libraries not supported on those
# platforms due to limitations in nested select() statements.
config_setting(
    name = "with_cuda_support_windows_override",
    define_values = {"using_cuda_nvcc": "true"},
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_xla_support",
    define_values = {"with_xla_support": "true"},
    visibility = ["//visibility:public"],
)

# By default, XLA GPU is compiled into tensorflow when building with
# --config=cuda even when `with_xla_support` is false. The config setting
# here allows us to override the behavior if needed.
config_setting(
    name = "no_xla_deps_in_cuda",
    define_values = {"no_xla_deps_in_cuda": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_gdr_support",
    define_values = {"with_gdr_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_verbs_support",
    define_values = {"with_verbs_support": "true"},
    visibility = ["//visibility:public"],
)

# Crosses between framework_shared_object and a bunch of other configurations
# due to limitations in nested select() statements.
config_setting(
    name = "framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_clang",
    define_values = {
        "using_cuda_clang": "true",
    },
)

config_setting(
    name = "using_cuda_clang_with_dynamic_build",
    define_values = {
        "using_cuda_clang": "true",
        "framework_shared_object": "true",
    },
)

# Setting to use when loading kernels dynamically
config_setting(
    name = "dynamic_loaded_kernels",
    define_values = {
        "dynamic_loaded_kernels": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_nvcc",
    define_values = {
        "using_cuda_nvcc": "true",
    },
)

config_setting(
    name = "using_cuda_nvcc_with_dynamic_build",
    define_values = {
        "using_cuda_nvcc": "true",
        "framework_shared_object": "true",
    },
)

config_setting(
    name = "using_rocm_hipcc",
    define_values = {
        "using_rocm_hipcc": "true",
    },
)

config_setting(
    name = "with_mpi_support",
    values = {"define": "with_mpi_support=true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "override_eigen_strong_inline",
    values = {"define": "override_eigen_strong_inline=true"},
    visibility = ["//visibility:public"],
)

# This flag is set from the configure step when the user selects with nGraph option.
# By default it should be false
config_setting(
    name = "with_ngraph_support",
    values = {"define": "with_ngraph_support=true"},
    visibility = ["//visibility:public"],
)

# This flag specifies whether TensorFlow 2.0 API should be built instead
# of 1.* API. Note that TensorFlow 2.0 API is currently under development.
config_setting(
    name = "api_version_2",
    define_values = {"tf_api_version": "2"},
)

# This flag is defined for select statements that match both
# on 'windows' and 'api_version_2'. In this case, bazel requires
# having a flag which is a superset of these two.
config_setting(
    name = "windows_and_api_version_2",
    define_values = {"tf_api_version": "2"},
    values = {"cpu": "x64_windows"},
)

package_group(
    name = "internal",
    packages = [
        "-//third_party/tensorflow/python/estimator",
        "//learning/deepmind/...",
        "//learning/meta_rank/...",
        "//platforms/performance/autograppler/...",
        "//tensorflow/...",
        "//tensorflow_estimator/contrib/...",
        "//tensorflow_fold/llgtm/...",
        "//tensorflow_text/...",
        "//third_party/py/tensor2tensor/...",
    ],
)

load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
)

filegroup(
    name = "intel_binary_blob",
    data = if_mkl_ml(
        [
            "//third_party/mkl:intel_binary_blob",
        ],
    ),
)

cc_library(
    name = "grpc",
    deps = select({
        ":linux_s390x": ["@grpc//:grpc_unsecure"],
        "//conditions:default": ["@grpc"],
    }),
)

cc_library(
    name = "grpc++",
    deps = select({
        ":linux_s390x": ["@grpc//:grpc++_unsecure"],
        "//conditions:default": ["@grpc//:grpc++"],
    }),
)

# A shared object which includes registration mechanisms for ops and
# kernels. Does not include the implementations of any ops or kernels. Instead,
# the library which loads libtensorflow_framework.so
# (e.g. _pywrap_tensorflow_internal.so for Python, libtensorflow.so for the C
# API) is responsible for registering ops with libtensorflow_framework.so. In
# addition to this core set of ops, user libraries which are loaded (via
# TF_LoadLibrary/tf.load_op_library) register their ops and kernels with this
# shared object directly.
#
# For example, from Python tf.load_op_library loads a custom op library (via
# dlopen() on Linux), the library finds libtensorflow_framework.so (no
# filesystem search takes place, since libtensorflow_framework.so has already
# been loaded by pywrap_tensorflow) and registers its ops and kernels via
# REGISTER_OP and REGISTER_KERNEL_BUILDER (which use symbols from
# libtensorflow_framework.so), and pywrap_tensorflow can then use these
# ops. Since other languages use the same libtensorflow_framework.so, op
# libraries are language agnostic.
#
# This shared object is not used unless framework_shared_object=true (set in the
# configure script unconditionally); otherwise if it is false or undefined, the
# build is static and TensorFlow symbols (in Python only) are loaded into the
# global symbol table in order to support op registration. This means that
# projects building with Bazel and importing TensorFlow as a dependency will not
# depend on libtensorflow_framework.so unless they opt in.
tf_cc_shared_object(
    name = "libtensorflow_framework.so",
    framework_so = [],
    linkopts = select({
        "//tensorflow:darwin": [],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location //tensorflow:tf_framework_version_script.lds)",
        ],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu_impl",
        "//tensorflow/core:framework_internal_impl",
        "//tensorflow/core:gpu_runtime_impl",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry_impl",
        "//tensorflow/core:lib_internal_impl",
        "//tensorflow/stream_executor:stream_executor_impl",
        "//tensorflow:tf_framework_version_script.lds",
    ] + tf_additional_binary_deps(),
)

# -------------------------------------------
# New rules should be added above this target.
# -------------------------------------------

# TensorFlow uses several libraries that may also be used by applications
# linking against the C and C++ APIs (such as libjpeg).  When we create
# the shared library, only export the core TF API functions to avoid
# causing library conflicts (e.g., those reported in github issue 1924).
# On Linux, tell the linker (-Wl,<option>) to use a version script that
# excludes all but a subset of function names.
# On MacOS, the linker does not support version_script, but has an
# an "-exported_symbols_list" command.  -z defs disallows undefined
# symbols in object files.

tf_cc_shared_object(
    name = "libtensorflow.so",
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-exported_symbols_list",  # This line must be directly followed by the exported_symbols.lds file
            "$(location //tensorflow/c:exported_symbols.lds)",
            "-Wl,-install_name,@rpath/libtensorflow.so",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location //tensorflow/c:version_script.lds)",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:exported_symbols.lds",
        "//tensorflow/c:version_script.lds",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_shared_object(
    name = "libtensorflow_cc.so",
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-exported_symbols_list",  # This line must be directly followed by the exported_symbols.lds file
            "$(location //tensorflow:tf_exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "$(location //tensorflow:tf_version_script.lds)",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tf_exported_symbols.lds",
        "//tensorflow:tf_version_script.lds",
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/cc/profiler",
        "//tensorflow/core:tensorflow",
    ] + if_ngraph(["@ngraph_tf//:ngraph_tf"]),
)

exports_files(
    [
        "tf_version_script.lds",
        "tf_exported_symbols.lds",
    ],
)

genrule(
    name = "install_headers",
    srcs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/cc:headers",
        "//tensorflow/core:headers",
    ],
    outs = ["include"],
    cmd = """
    mkdir $@
    for f in $(SRCS); do
      d="$${f%/*}"
      d="$${d#bazel-out*genfiles/}"
      d="$${d#*external/eigen_archive/}"

      if [[ $${d} == *local_config_* ]]; then
        continue
      fi

      if [[ $${d} == external* ]]; then
        extname="$${d#*external/}"
        extname="$${extname%%/*}"
        if [[ $${TF_SYSTEM_LIBS:-} == *$${extname}* ]]; then
          continue
        fi
      fi

      mkdir -p "$@/$${d}"
      cp "$${f}" "$@/$${d}/"
    done
    """,
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "root_init_gen",
    srcs = select({
        "api_version_2": [":tf_python_api_gen_v2"],
        "//conditions:default": [":tf_python_api_gen_v1"],
    }),
    outs = ["__init__.py"],
    cmd = select({
        "api_version_2": "cp $(@D)/_api/v2/v2.py $(OUTS)",
        "//conditions:default": "cp $(@D)/_api/v1/v1.py $(OUTS)",
    }),
)

gen_api_init_files(
    name = "tf_python_api_gen_v1",
    srcs = [
        "api_template_v1.__init__.py",
        "compat_template_v1.__init__.py",
    ],
    api_version = 1,
    compat_api_versions = [1],
    compat_init_templates = ["compat_template_v1.__init__.py"],
    output_dir = "_api/v1/",
    output_files = TENSORFLOW_API_INIT_FILES_V1_WITH_COMPAT,
    output_package = "tensorflow._api.v1",
    root_file_name = "v1.py",
    root_init_template = "api_template_v1.__init__.py",
)

gen_api_init_files(
    name = "tf_python_api_gen_v2",
    srcs = [
        "api_template.__init__.py",
        "compat_template_v1.__init__.py",
    ],
    api_version = 2,
    compat_api_versions = [1],
    compat_init_templates = ["compat_template_v1.__init__.py"],
    output_dir = "_api/v2/",
    output_files = TENSORFLOW_API_INIT_FILES_V2,
    output_package = "tensorflow._api.v2",
    root_file_name = "v2.py",
    root_init_template = "api_template.__init__.py",
)

py_library(
    name = "tensorflow_py",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = select({
        "api_version_2": [],
        "//conditions:default": ["//tensorflow/contrib:contrib_py"],
    }) + [
        ":tensorflow_py_no_contrib",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_library(
    name = "tensorflow_py_no_contrib",
    srcs = select({
        "api_version_2": [":tf_python_api_gen_v2"],
        "//conditions:default": [":tf_python_api_gen_v1"],
    }) + [":root_init_gen"] + [
        "//tensorflow/python/keras/api:keras_python_api_gen",
        "//tensorflow/python/keras/api:keras_python_api_gen_compat_v1",
        "//tensorflow/python/keras/api:keras_python_api_gen_compat_v2",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python:no_contrib"],
)
