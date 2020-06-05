# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//tensorflow:tensorflow.bzl", "VERSION", "tf_cc_shared_object", "tf_custom_op_library_additional_deps_impl", "tf_native_cc_binary")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_binary_deps",
)
load(
    "//tensorflow/python/tools/api/generator:api_gen.bzl",
    "gen_api_init_files",  # @unused
    "get_compat_files",
    "get_nested_compat_files",
)
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
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
)
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "LICENSE",
    "ACKNOWLEDGMENTS",
    # The leakr files are used by //third_party/cloud_tpu and
    # //third_party/tensorboard/google:copybara_config_test.
    "leakr_badwords.dic",
    "leakr_badfiles.dic",
    "leakr_file_type_recipe.ftrcp",
])

# @unused
TENSORFLOW_API_INIT_FILES_V2 = (
    TENSORFLOW_API_INIT_FILES +
    get_compat_files(TENSORFLOW_API_INIT_FILES, 2) +
    get_compat_files(TENSORFLOW_API_INIT_FILES_V1, 1) + get_nested_compat_files([
        1,
        2,
    ])
)

# @unused
TENSORFLOW_API_INIT_FILES_V1 = (
    TENSORFLOW_API_INIT_FILES_V1 +
    get_compat_files(TENSORFLOW_API_INIT_FILES, 2) +
    get_compat_files(TENSORFLOW_API_INIT_FILES_V1, 1) + get_nested_compat_files([
        1,
        2,
    ])
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
    name = "emscripten",
    values = {"crosstool_top": "//external:android/emscripten"},
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
    name = "macos",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios",
    values = {"apple_platform_type": "ios"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "fuchsia",
    values = {"cpu": "fuchsia"},
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
    name = "chromiumos",
    values = {"crosstool_top": "//external:android/chromiumos"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_aarch64",
    values = {"cpu": "aarch64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_armhf",
    values = {"cpu": "armhf"},
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
    name = "no_nccl_support",
    define_values = {"no_nccl_support": "true"},
    visibility = ["//visibility:public"],
)

# Experimental features
config_setting(
    name = "no_stackdriver_support",
    define_values = {"no_stackdriver_support": "true"},
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
    name = "with_numa_support",
    define_values = {"with_numa_support": "true"},
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
    name = "macos_with_framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_clang",
    define_values = {
        "using_cuda_clang": "true",
    },
)

# Flag to indicate open source build, .bazelrc always has it set to be true
config_setting(
    name = "oss",
    define_values = {
        "open_source_build": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_clang_with_dynamic_build",
    define_values = {
        "using_cuda_clang": "true",
        "framework_shared_object": "true",
    },
)

config_setting(
    name = "build_oss_using_cuda_clang",
    define_values = {
        "using_cuda_clang": "true",
        "open_source_build": "true",
    },
)

# Setting to use when loading kernels dynamically
config_setting(
    name = "dynamic_loaded_kernels",
    define_values = {
        "dynamic_loaded_kernels": "true",
        "framework_shared_object": "true",
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
    name = "build_oss_using_cuda_nvcc",
    define_values = {
        "using_cuda_nvcc": "true",
        "open_source_build": "true",
    },
)

config_setting(
    name = "using_rocm_hipcc",
    define_values = {
        "using_rocm_hipcc": "true",
    },
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
    visibility = ["//visibility:public"],
)

# This flag is defined for select statements that match both
# on 'windows' and 'api_version_2'. In this case, bazel requires
# having a flag which is a superset of these two.
config_setting(
    name = "windows_and_api_version_2",
    define_values = {"tf_api_version": "2"},
    values = {"cpu": "x64_windows"},
)

# This flag enables experimental MLIR support.
config_setting(
    name = "with_mlir_support",
    values = {"define": "with_mlir_support=true"},
    visibility = ["//visibility:public"],
)

# Specifies via a config setting if this is a mobile build or not, makes
# it easier to combine settings later.
selects.config_setting_group(
    name = "mobile",
    match_any = [
        ":android",
        ":chromiumos",
        ":emscripten",
        ":ios",
    ],
)

config_setting(
    name = "lite_protos_legacy",
    values = {"define": "TENSORFLOW_PROTOS=lite"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "full_protos",
    values = {"define": "TENSORFLOW_PROTOS=full"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "lite_protos",
    match_any = [":lite_protos_legacy"],
)

selects.config_setting_group(
    name = "mobile_lite_protos",
    match_all = [
        ":lite_protos",
        ":mobile",
    ],
)

selects.config_setting_group(
    name = "mobile_full_protos",
    match_all = [
        ":full_protos",
        ":mobile",
    ],
)

# DO NOT ADD ANY NEW EXCEPTIONS TO THIS LIST!
# Instead, please use public APIs or public build rules TF provides.
# If you need functionality that is not exposed, we will work with you to expand our public APIs.
package_group(
    name = "internal",
    packages = [
        # To pass open source testing in the pip Kokoros.
        "//bazel_pip/tensorflow/...",
        "//learning/brain/swift/x10/...",
        "//perftools/accelerators/xprof/api/...",
        "//third_party/py/autograph/...",
        "//third_party/swift/tensorflow/x10/...",
        "//third_party/swift/tensorflow_apis/...",
        "//tensorflow/...",
        "//tensorflow_estimator/python/estimator/...",
        "//tensorflow_models/official/...",
    ],
)

package_group(
    name = "ndarray_tensor_allow_list",
    packages = ["//learning/pathways/..."],
)

# Packages that use composite tensors or dispatch.
# TODO(b/154762408) Remove this package group once it's no longer needed.
package_group(name = "composite_tensor_whitelist")

# Packages that use private types symbols, until they are exported.
# TODO(b/154650521) Remove.
package_group(
    name = "types_whitelist",
    packages = ["//learning/deepmind/tensorflow/replicator/..."],
)

filegroup(
    name = "intel_binary_blob",
    data = if_mkl_ml(
        [
            "//third_party/mkl:intel_binary_blob",
        ],
    ),
)

bzl_library(
    name = "tensorflow_bzl",
    srcs = ["tensorflow.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:build_config_root_bzl",
        "//tensorflow/core/platform:rules_cc_bzl",
        "//tensorflow/core/platform/default:cuda_build_defs_bzl",
        "//third_party/mkl:build_defs_bzl",
        "//third_party/mkl_dnn:build_defs_bzl",
        "//third_party/ngraph:build_defs_bzl",
        "@local_config_cuda//cuda:build_defs_bzl",
        "@local_config_rocm//rocm:build_defs_bzl",
        "@local_config_tensorrt//:build_defs_bzl",
    ],
)

cc_library(
    name = "grpc",
    visibility = ["//visibility:public"],
    deps = select({
        ":linux_s390x": ["@com_github_grpc_grpc//:grpc_unsecure"],
        "//conditions:default": ["@com_github_grpc_grpc//:grpc"],
    }),
)

cc_library(
    name = "grpc++",
    visibility = ["//visibility:public"],
    deps = select({
        ":linux_s390x": ["@com_github_grpc_grpc//:grpc++_unsecure"],
        "//conditions:default": ["@com_github_grpc_grpc//:grpc++"],
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
#
# DEBUGGING DUPLICATE INITIALIZATION
# ----------------------------------
#
# Having a dynamic library introduces a diamond dependency problem:
# if a target X is depended on by both libtensorflow_framework.so and the
# users of libtensorflow_framework.so, the definitions will get duplicated.
# This causes global initializers which need to be run exactly once (e.g.
# protobuf registration) to crash, as the initialization is run once from the
# statically linked in global, and one by the global which comes in from
# libtensorflow_framework.so.
# Even worse, global objects which need to be singletons for semantical
# correctness (e.g. registers) might get dupliacted.
#
# In order to avoid these HARD TO DEBUG CRASHES, it is sufficient to follow
# these rules:
#  - All globals with non-trivial static constructors or for which a
#    single identity is required (e.g. registers) need to live in `*_impl`
#    targets.
#
#  - An `*_impl` target has to be (transitively) included into
#    `libtensorflow_framework.so`.
#
#  - A target T1 can depend on `*_impl` target T2 only if:
#
#    -> It's a tf_cc_shared_object, and there is no other tf_cc_shared_object
#       transitively depending on T2.
#    -> It's an `*_impl` target by itself
#    -> The dependency is guarded by `if_static`. This is discouraged,
#       as it diverges dependency topology between static and dynamic TF.
#
# TODO(cheshire): write tests to check for rule violations
tf_cc_shared_object(
    name = "tensorflow_framework",
    framework_so = [],
    linkopts = select({
        "//tensorflow:macos": [],
        "//tensorflow:windows": [],
        "//tensorflow:freebsd": [
            "-Wl,--version-script,$(location //tensorflow:tf_framework_version_script.lds)",
            "-lexecinfo",
        ],
        "//conditions:default": [
            "-Wl,--version-script,$(location //tensorflow:tf_framework_version_script.lds)",
        ],
    }),
    linkstatic = 1,
    per_os_targets = True,
    soversion = VERSION,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/cc/saved_model:loader_lite_impl",
        "//tensorflow/core:core_cpu_impl",
        "//tensorflow/core:framework_internal_impl",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime_impl",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry_impl",
        "//tensorflow/core:lib_internal_impl",
        "//tensorflow/core/profiler:profiler_impl",
        "//tensorflow/stream_executor:stream_executor_impl",
        "//tensorflow:tf_framework_version_script.lds",
    ] + tf_additional_binary_deps(),
)

# This is intended to be the same as tf_binary_additional_srcs:
# https://github.com/tensorflow/tensorflow/blob/cd67f4f3723f9165aabedd0171aaadc6290636e5/tensorflow/tensorflow.bzl#L396-L425
# And is usable in the "deps" attribute instead of the "srcs" attribute
# as a workaround for https://github.com/tensorflow/tensorflow/issues/34117
cc_import(
    name = "libtensorflow_framework_import_lib",
    shared_library = select({
        "//tensorflow:macos": ":libtensorflow_framework.dylib",
        "//conditions:default": ":libtensorflow_framework.so",
    }),
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
    name = "tensorflow",
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/c:exported_symbols.lds)",
        ],
        "//tensorflow:windows": [
        ],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/c:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    soversion = VERSION,
    visibility = ["//visibility:public"],
    # add win_def_file for tensorflow
    win_def_file = select({
        # We need this DEF file to properly export symbols on Windows
        "//tensorflow:windows": ":tensorflow_filtered_def_file",
        "//conditions:default": None,
    }),
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:exported_symbols.lds",
        "//tensorflow/c:version_script.lds",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/core:distributed_tensorflow_dependencies",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_shared_object(
    name = "tensorflow_cc",
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow:tf_exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow:tf_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    soversion = VERSION,
    visibility = ["//visibility:public"],
    # add win_def_file for tensorflow_cc
    win_def_file = select({
        # We need this DEF file to properly export symbols on Windows
        "//tensorflow:windows": ":tensorflow_filtered_def_file",
        "//conditions:default": None,
    }),
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

# ** Targets for Windows build (start) **

# Build a shared library (DLL) by cc_binary from tf_custom_op_library_additional_deps_impl,
# it contains all object code from its dependencies.
# This target is only used for parsing the symbols to be exported in tensorflow.dll.
# Do NOT depend on it.
tf_native_cc_binary(
    name = "tf_custom_op_library_additional_deps.dll",
    linkshared = 1,
    linkstatic = 1,
    deps = tf_custom_op_library_additional_deps_impl(),
)

# Get a DEF file generated by parsing all object files
# of tf_custom_op_library_additional_deps.so
filegroup(
    name = "tensorflow_def_file",
    srcs = [":tf_custom_op_library_additional_deps.dll"],
    output_group = "def_file",
)

# Filter the DEF file to reduce the number of symbols to 64K or less.
# Note that we also write the name of the pyd file into DEF file so that
# the dynamic libraries of custom ops can find it at runtime.
genrule(
    name = "tensorflow_filtered_def_file",
    srcs = [":tensorflow_def_file"],
    outs = ["tensorflow_filtered_def_file.def"],
    cmd = select({
        "//tensorflow:windows": """
              $(location @local_config_def_file_filter//:def_file_filter) \\
              --input $(location :tensorflow_def_file) \\
              --output $@
          """,
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    tools = ["@local_config_def_file_filter//:def_file_filter"],
    visibility = ["//visibility:public"],
)

# The interface library (tensorflow.dll.if.lib) for linking tensorflow DLL library (tensorflow.dll) on Windows.
# To learn more about import library (called interface library in Bazel):
#     https://docs.microsoft.com/en-us/cpp/build/linking-an-executable-to-a-dll?view=vs-2017#linking-implicitly
filegroup(
    name = "get_tensorflow_dll_import_lib",
    srcs = ["//tensorflow:tensorflow.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:public"],
)

# Rename the import library for tensorflow.dll from tensorflow.dll.if.lib to tensorflow.lib
genrule(
    name = "tensorflow_dll_import_lib",
    srcs = [":get_tensorflow_dll_import_lib"],
    outs = ["tensorflow.lib"],
    cmd = select({
        "//tensorflow:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# The interface library (tensorflow_cc.dll.if.lib) for linking tensorflow DLL library (tensorflow_cc.dll) on Windows.
# To learn more about import library (called interface library in Bazel):
#     https://docs.microsoft.com/en-us/cpp/build/linking-an-executable-to-a-dll?view=vs-2017#linking-implicitly
filegroup(
    name = "get_tensorflow_cc_dll_import_lib",
    srcs = ["//tensorflow:tensorflow_cc.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:public"],
)

# Rename the import library for tensorflow.dll from tensorflow_cc.dll.if.lib to tensorflow.lib
genrule(
    name = "tensorflow_cc_dll_import_lib",
    srcs = [":get_tensorflow_cc_dll_import_lib"],
    outs = ["tensorflow_cc.lib"],
    cmd = select({
        "//tensorflow:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# ** Targets for Windows build (end) **

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
      d="$${d#bazel-out/*/genfiles/}"
      d="$${d#bazel-out/*/bin/}"

      if [[ $${d} == *local_config_* ]]; then
        continue
      fi

      if [[ $${d} == external* ]]; then
        extname="$${d#*external/}"
        extname="$${extname%%/*}"
        if [[ $${TF_SYSTEM_LIBS:-} == *$${extname}* ]]; then
          continue
        fi

        d="$${d#*external/farmhash_archive/src}"
        d="$${d#*external/$${extname}/}"
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
        "api_version_2": "cp $(@D)/_api/v2/v2.py $(OUTS) && sed -i'.original' 's:from . import:from ._api.v2 import:g' $(OUTS)",
        "//conditions:default": "cp $(@D)/_api/v1/v1.py $(OUTS) && sed -i'.original' 's:from . import:from ._api.v1 import:g' $(OUTS)",
    }),
)

genrule(
    name = "virtual_root_init_gen",
    srcs = select({
        "api_version_2": [":virtual_root_template_v2.__init__.py"],
        "//conditions:default": [":virtual_root_template_v1.__init__.py"],
    }),
    outs = ["virtual_root.__init__.py"],
    cmd = "cp $(SRCS) $(OUTS)",
)

gen_api_init_files(
    name = "tf_python_api_gen_v1",
    srcs = [
        "api_template_v1.__init__.py",
        "compat_template.__init__.py",
        "compat_template_v1.__init__.py",
    ],
    api_version = 1,
    compat_api_versions = [
        1,
        2,
    ],
    compat_init_templates = [
        "compat_template_v1.__init__.py",
        "compat_template.__init__.py",
    ],
    output_dir = "_api/v1/",
    output_files = TENSORFLOW_API_INIT_FILES_V1,
    output_package = "tensorflow._api.v1",
    root_file_name = "v1.py",
    root_init_template = "$(location api_template_v1.__init__.py)",
)

gen_api_init_files(
    name = "tf_python_api_gen_v2",
    srcs = [
        "api_template.__init__.py",
        "compat_template.__init__.py",
        "compat_template_v1.__init__.py",
    ],
    api_version = 2,
    compat_api_versions = [
        1,
        2,
    ],
    compat_init_templates = [
        "compat_template_v1.__init__.py",
        "compat_template.__init__.py",
    ],
    output_dir = "_api/v2/",
    output_files = TENSORFLOW_API_INIT_FILES_V2,
    output_package = "tensorflow._api.v2",
    root_file_name = "v2.py",
    root_init_template = "$(location api_template.__init__.py)",
)

py_library(
    name = "tensorflow_py",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = select({
        "api_version_2": [],
        "//conditions:default": [],
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
    }) + [
        ":root_init_gen",
        "//tensorflow/python/keras/api:keras_python_api_gen",
        "//tensorflow/python/keras/api:keras_python_api_gen_compat_v1",
        "//tensorflow/python/keras/api:keras_python_api_gen_compat_v2",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python:no_contrib"],
)
