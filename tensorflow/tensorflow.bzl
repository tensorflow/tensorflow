#
# Returns the options to use for a C++ library or binary build.
# Uses the ":optmode" config_setting to pick the options.
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_dynamic_kernels",
    "if_static",
    "tf_additional_grpc_deps_py",
    "tf_additional_xla_deps_py",
    "tf_exec_properties",
    "tf_gpu_tests_tags",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_binary",
    "cc_library",
    "cc_shared_library",
    "cc_test",
)
load(
    "//tensorflow/tsl:tsl.bzl",
    "tsl_gpu_library",
    _cc_header_only_library = "cc_header_only_library",
    _clean_dep = "clean_dep",
    _if_cuda_or_rocm = "if_cuda_or_rocm",
    _if_nccl = "if_nccl",
    _transitive_hdrs = "transitive_hdrs",
)
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_library",
    "if_cuda",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
    "rocm_copts",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_enable_mkl",
    "if_mkl",
    "if_mkl_ml",
)
load(
    "//third_party/mkl_dnn:build_defs.bzl",
    "if_mkldnn_aarch64_acl",
    "if_mkldnn_aarch64_acl_openmp",
    "if_mkldnn_openmp",
    "if_onednn_v3",
)
load(
    "//third_party/compute_library:build_defs.bzl",
    "if_enable_acl",
)
load(
    "//third_party/llvm_openmp:openmp.bzl",
    "windows_llvm_openmp_linkopts",
)
load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def register_extension_info(**kwargs):
    pass

# version for the shared libraries, can
# not contain rc or alpha, only numbers.
# Also update tensorflow/core/public/version.h
# and tensorflow/tools/pip_package/setup.py
VERSION = "2.14.0"
VERSION_MAJOR = VERSION.split(".")[0]
two_gpu_tags = ["requires-gpu-nvidia:2", "notap", "manual", "no_pip"]

# The workspace root, to be used to set workspace 'include' paths in a way that
# will still work correctly when TensorFlow is included as a dependency of an
# external project.
workspace_root = Label("//:WORKSPACE").workspace_root or "."

clean_dep = _clean_dep
cc_header_only_library = _cc_header_only_library
transitive_hdrs = _transitive_hdrs

def if_oss(oss_value, google_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value  # copybara:comment_replace return google_value

def if_google(google_value, oss_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value  # copybara:comment_replace return google_value

def if_v2(a):
    return select({
        clean_dep("//tensorflow:api_version_2"): a,
        "//conditions:default": [],
    })

def if_not_v2(a):
    return select({
        clean_dep("//tensorflow:api_version_2"): [],
        "//conditions:default": a,
    })

def if_nvcc(a):
    return select({
        "@local_config_cuda//cuda:using_nvcc": a,
        "//conditions:default": [],
    })

def if_xla_available(if_true, if_false = []):
    return select({
        clean_dep("//tensorflow:with_xla_support"): if_true,
        "//conditions:default": if_false,
    })

# Given a source file, generate a test name.
# i.e. "common_runtime/direct_session_test.cc" becomes
#      "common_runtime_direct_session_test"
def src_to_test_name(src):
    return src.replace("/", "_").replace(":", "_").split(".")[0]

def full_path(relative_paths):
    return [native.package_name() + "/" + relative for relative in relative_paths]

def _add_tfcore_prefix(src):
    if src.startswith("//"):
        return src
    return "//tensorflow/core:" + src

def tf_android_core_proto_headers(core_proto_sources_relative):
    """Returns the list of pb.h and proto.h headers that are generated for the provided sources."""
    return ([
        _add_tfcore_prefix(p).replace(":", "/").replace(".proto", ".pb.h")
        for p in core_proto_sources_relative
    ] + [
        _add_tfcore_prefix(p).replace(":", "/").replace(".proto", ".proto.h")
        for p in core_proto_sources_relative
    ])

def tf_portable_full_lite_protos(full, lite):
    return select({
        "//tensorflow:mobile_lite_protos": lite,
        "//tensorflow:mobile_full_protos": full,
        # The default should probably be lite runtime, but since most clients
        # seem to use the non-lite version, let's make that the default for now.
        "//conditions:default": full,
    })

def if_no_default_logger(a):
    return select({
        clean_dep("//tensorflow:no_default_logger"): a,
        "//conditions:default": [],
    })

def if_android_x86(a):
    return select({
        clean_dep("//tensorflow:android_x86"): a,
        clean_dep("//tensorflow:android_x86_64"): a,
        "//conditions:default": [],
    })

def if_android_arm(a):
    return select({
        clean_dep("//tensorflow:android_arm"): a,
        "//conditions:default": [],
    })

def if_android_arm64(a):
    return select({
        clean_dep("//tensorflow:android_arm64"): a,
        "//conditions:default": [],
    })

def if_android_mips(a):
    return select({
        clean_dep("//tensorflow:android_mips"): a,
        "//conditions:default": [],
    })

def if_not_android(a):
    return select({
        clean_dep("//tensorflow:android"): [],
        "//conditions:default": a,
    })

def if_not_android_mips_and_mips64(a):
    return select({
        clean_dep("//tensorflow:android_mips"): [],
        clean_dep("//tensorflow:android_mips64"): [],
        "//conditions:default": a,
    })

def if_android(a):
    return select({
        clean_dep("//tensorflow:android"): a,
        "//conditions:default": [],
    })

def if_android_or_ios(a):
    return select({
        clean_dep("//tensorflow:android"): a,
        clean_dep("//tensorflow:ios"): a,
        "//conditions:default": [],
    })

def if_emscripten(a):
    return select({
        clean_dep("//tensorflow:emscripten"): a,
        "//conditions:default": [],
    })

def if_chromiumos(a, otherwise = []):
    return select({
        clean_dep("//tensorflow:chromiumos"): a,
        "//conditions:default": otherwise,
    })

def if_macos(a, otherwise = []):
    return select({
        clean_dep("//tensorflow:macos"): a,
        "//conditions:default": otherwise,
    })

def if_ios(a, otherwise = []):
    return select({
        clean_dep("//tensorflow:ios"): a,
        "//conditions:default": otherwise,
    })

def if_ios_x86_64(a):
    return select({
        clean_dep("//tensorflow:ios_x86_64"): a,
        "//conditions:default": [],
    })

def if_mobile(a):
    return select({
        clean_dep("//tensorflow:mobile"): a,
        "//conditions:default": [],
    })

def if_not_mobile(a):
    return select({
        clean_dep("//tensorflow:mobile"): [],
        "//conditions:default": a,
    })

# Config setting selector used when building for products
# which requires restricted licenses to be avoided.
def if_not_mobile_or_arm_or_lgpl_restricted(a):
    _ = (a,)
    return select({
        "//conditions:default": [],
    })

def if_not_windows(a):
    return select({
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": a,
    })

def if_windows(a, otherwise = []):
    return select({
        clean_dep("//tensorflow:windows"): a,
        "//conditions:default": otherwise,
    })

def if_windows_cuda(a, otherwise = []):
    return select({
        clean_dep("//tensorflow:is_cuda_enabled_and_windows"): a,
        "//conditions:default": otherwise,
    })

def if_not_fuchsia(a):
    return select({
        clean_dep("//tensorflow:fuchsia"): [],
        "//conditions:default": a,
    })

def if_linux_x86_64(a):
    return select({
        clean_dep("//tensorflow:linux_x86_64"): a,
        "//conditions:default": [],
    })

def if_override_eigen_strong_inline(a):
    return select({
        clean_dep("//tensorflow:override_eigen_strong_inline"): a,
        "//conditions:default": [],
    })

if_nccl = _if_nccl

def if_zendnn(if_true, if_false = []):
    return select({
        clean_dep("//tensorflow:linux_x86_64"): if_true,
        "//conditions:default": if_false,
    })

def if_libtpu(if_true, if_false = []):
    """Shorthand for select()ing whether to build backend support for TPUs when building libtpu.so"""
    return select({
        # copybara:uncomment_begin(different config setting in OSS)
        # "//tools/cc_target_os:gce": if_true,
        # copybara:uncomment_end_and_comment_begin
        clean_dep("//tensorflow:with_tpu_support"): if_true,
        # copybara:comment_end
        "//conditions:default": if_false,
    })

def if_with_tpu_support(if_true, if_false = []):
    """Shorthand for select()ing whether to build API support for TPUs when building TensorFlow"""
    return select({
        "//tensorflow:with_tpu_support": if_true,
        "//conditions:default": if_false,
    })

def if_registration_v2(if_true, if_false = []):
    return select({
        "//tensorflow:registration_v2": if_true,
        "//conditions:default": if_false,
    })

def if_portable(if_true, if_false = []):
    return if_true

ADDITIONAL_API_INDEXABLE_SETTINGS = []

# We are never indexing generated code in the OSS build, but still
# return a select() for consistency.
def if_indexing_source_code(
        if_true,  # @unused
        if_false):
    """Return a select() on whether or not we are building for source code indexing."""
    return select({
        "//conditions:default": if_false,
    })

# Linux systems may required -lrt linker flag for e.g. clock_gettime
# see https://github.com/tensorflow/tensorflow/issues/15129
def lrt_if_needed():
    lrt = ["-lrt"]
    return select({
        clean_dep("//tensorflow:linux_aarch64"): lrt,
        clean_dep("//tensorflow:linux_x86_64"): lrt,
        clean_dep("//tensorflow:linux_ppc64le"): lrt,
        "//conditions:default": [],
    })

def get_win_copts(is_external = False):
    WINDOWS_COPTS = [
        # copybara:uncomment_begin(no MSVC flags in google)
        # "-DPLATFORM_WINDOWS",
        # "-DEIGEN_HAS_C99_MATH",
        # "-DTENSORFLOW_USE_EIGEN_THREADPOOL",
        # "-DEIGEN_AVOID_STL_ARRAY",
        # "-Iexternal/gemmlowp",
        # "-Wno-sign-compare",
        # "-DNOGDI",
        # copybara:uncomment_end_and_comment_begin
        "/DPLATFORM_WINDOWS",
        "/DEIGEN_HAS_C99_MATH",
        "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
        "/DEIGEN_AVOID_STL_ARRAY",
        "/Iexternal/gemmlowp",
        "/wd4018",  # -Wno-sign-compare
        # Bazel's CROSSTOOL currently pass /EHsc to enable exception by
        # default. We can't pass /EHs-c- to disable exception, otherwise
        # we will get a waterfall of flag conflict warnings. Wait for
        # Bazel to fix this.
        # "/D_HAS_EXCEPTIONS=0",
        # "/EHs-c-",
        "/wd4577",
        "/DNOGDI",
        # Also see build:windows lines in tensorflow/opensource_only/.bazelrc
        # where we set some other options globally.
        # copybara:comment_end
    ]

    if is_external:
        return WINDOWS_COPTS + [if_oss(
            "/UTF_COMPILE_LIBRARY",
            "-UTF_COMPILE_LIBRARY",
        )]
    else:
        return WINDOWS_COPTS + [if_oss(
            "/DTF_COMPILE_LIBRARY",
            "-DTF_COMPILE_LIBRARY",
        )]

def tf_copts(
        android_optimization_level_override = "-O2",
        is_external = False,
        allow_exceptions = False):
    # For compatibility reasons, android_optimization_level_override
    # is currently only being set for Android.
    # To clear this value, and allow the CROSSTOOL default
    # to be used, pass android_optimization_level_override=None
    android_copts = [
        "-DTF_LEAN_BINARY",
        "-Wno-narrowing",
        "-fomit-frame-pointer",
    ]
    if android_optimization_level_override:
        android_copts.append(android_optimization_level_override)
    return (
        if_not_windows([
            "-DEIGEN_AVOID_STL_ARRAY",
            "-Iexternal/gemmlowp",
            "-Wno-sign-compare",
            "-ftemplate-depth=900",
        ]) +
        (if_not_windows(["-fno-exceptions"]) if not allow_exceptions else []) +
        if_cuda(["-DGOOGLE_CUDA=1"]) +
        if_nvcc(["-DTENSORFLOW_USE_NVCC=1"]) +
        if_libtpu(["-DLIBTPU_ON_GCE"], []) +
        if_xla_available(["-DTENSORFLOW_USE_XLA=1"]) +
        if_tensorrt(["-DGOOGLE_TENSORRT=1"]) +
        if_rocm(["-DTENSORFLOW_USE_ROCM=1"]) +
        # Compile in oneDNN based ops when building for x86 platforms
        if_mkl(["-DINTEL_MKL"]) +
        # Enable additional ops (e.g., ops with non-NHWC data layout) and
        # optimizations for Intel builds using oneDNN if configured
        if_enable_mkl(["-DENABLE_MKL"]) +
        if_mkldnn_openmp(["-DENABLE_ONEDNN_OPENMP"]) +
        if_onednn_v3(["-DENABLE_ONEDNN_V3"]) +
        if_mkldnn_aarch64_acl(["-DDNNL_AARCH64_USE_ACL=1"]) +
        if_mkldnn_aarch64_acl_openmp(["-DENABLE_ONEDNN_OPENMP"]) +
        if_zendnn(["-DAMD_ZENDNN"]) +
        if_enable_acl(["-DXLA_CPU_USE_ACL=1", "-fexceptions"]) +
        if_android_arm(["-mfpu=neon"]) +
        if_linux_x86_64(["-msse3"]) +
        if_ios_x86_64(["-msse4.1"]) +
        if_no_default_logger(["-DNO_DEFAULT_LOGGER"]) +
        select({
            clean_dep("//tensorflow:framework_shared_object"): [],
            "//conditions:default": ["-DTENSORFLOW_MONOLITHIC_BUILD"],
        }) +
        select({
            clean_dep("//tensorflow:android"): android_copts,
            clean_dep("//tensorflow:emscripten"): [],
            clean_dep("//tensorflow:macos"): [],
            clean_dep("//tensorflow:windows"): get_win_copts(is_external),
            clean_dep("//tensorflow:ios"): [],
            clean_dep("//tensorflow:no_lgpl_deps"): ["-D__TENSORFLOW_NO_LGPL_DEPS__", "-pthread"],
            "//conditions:default": ["-pthread"],
        })
    )

def tf_openmp_copts():
    # We assume when compiling on Linux gcc/clang will be used and MSVC on Windows
    return select({
        # copybara:uncomment_begin
        # "//third_party/mkl:build_with_mkl_lnx_openmp": ["-fopenmp"],
        # "//third_party/mkl:build_with_mkl_windows_openmp": ["/openmp"],
        # copybara:uncomment_end_and_comment_begin
        "@org_tensorflow//third_party/mkl:build_with_mkl_lnx_openmp": ["-fopenmp"],
        "@org_tensorflow//third_party/mkl:build_with_mkl_windows_openmp": ["/openmp:llvm"],
        # copybara:comment_end
        "//conditions:default": [],
    })

def tf_openmp_lopts():
    # When compiling on Windows, force MSVC to use libiomp that was compiled
    # as part of this build.
    return select({
        "//third_party/mkl:build_with_mkl_windows_openmp": [windows_llvm_openmp_linkopts()],
        "//conditions:default": [],
    })

def tf_opts_nortti():
    return [
        "-fno-rtti",
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
    ]

def tf_opts_nortti_if_android():
    return if_android(tf_opts_nortti())

def tf_opts_nortti_if_mobile():
    return if_mobile(tf_opts_nortti())

def tf_defines_nortti():
    return [
        "GOOGLE_PROTOBUF_NO_RTTI",
        "GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
    ]

def tf_defines_nortti_if_android():
    return if_android(tf_defines_nortti())

def tf_features_nomodules_if_android():
    return if_android(["-use_header_modules"])

def tf_features_nomodules_if_mobile():
    return if_mobile(["-use_header_modules"])

# portable_tensorflow_lib_lite does not export the headers needed to
# use it.  Thus anything that depends on it needs to disable layering
# check.
def tf_features_nolayering_check_if_ios():
    return select({
        clean_dep("//tensorflow:ios"): ["-layering_check"],
        "//conditions:default": [],
    })

def tf_opts_nortti_if_lite_protos():
    return tf_portable_full_lite_protos(
        full = [],
        lite = tf_opts_nortti(),
    )

def tf_defines_nortti_if_lite_protos():
    return tf_portable_full_lite_protos(
        full = [],
        lite = tf_defines_nortti(),
    )

# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate a library for that file.
def tf_gen_op_libs(
        op_lib_names,
        sub_directory = "ops/",
        deps = None,
        is_external = True,
        compatible_with = None,
        features = []):
    # Make library out of each op so it can also be used to generate wrappers
    # for various languages.
    if not deps:
        deps = []
    for n in op_lib_names:
        cc_library(
            name = n + "_op_lib",
            copts = tf_copts(is_external = is_external),
            features = features,
            srcs = [sub_directory + n + ".cc"],
            deps = deps + [clean_dep("//tensorflow/core:framework")],
            compatible_with = compatible_with,
            visibility = ["//visibility:public"],
            alwayslink = 1,
            linkstatic = 1,
        )

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        clean_dep("//tensorflow:macos"): [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
        ],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
        ],
    })

def _rpath_user_link_flags(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        clean_dep("//tensorflow:macos"): [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
        ],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$ORIGIN", levels_to_root),),
        ],
    })

# Bazel-generated shared objects which must be linked into TensorFlow binaries
# to define symbols from //tensorflow/core:framework and //tensorflow/core:lib.
def tf_binary_additional_srcs(fullversion = False):
    if fullversion:
        suffix = "." + VERSION
    else:
        suffix = "." + VERSION_MAJOR

    return if_static(
        extra_deps = [],
        macos = [
            clean_dep("//tensorflow:libtensorflow_framework%s.dylib" % suffix),
        ],
        otherwise = [
            clean_dep("//tensorflow:libtensorflow_framework.so%s" % suffix),
        ],
    )

def tf_binary_additional_data_deps():
    return if_static(
        extra_deps = [],
        macos = [
            clean_dep("//tensorflow:libtensorflow_framework.dylib"),
            clean_dep("//tensorflow:libtensorflow_framework.%s.dylib" % VERSION_MAJOR),
            clean_dep("//tensorflow:libtensorflow_framework.%s.dylib" % VERSION),
        ],
        otherwise = [
            clean_dep("//tensorflow:libtensorflow_framework.so"),
            clean_dep("//tensorflow:libtensorflow_framework.so.%s" % VERSION_MAJOR),
            clean_dep("//tensorflow:libtensorflow_framework.so.%s" % VERSION),
        ],
    )

def tf_binary_pybind_deps():
    return select({
        clean_dep("//tensorflow:macos"): [
            clean_dep(
                "//tensorflow/python:_pywrap_tensorflow_internal_macos",
            ),
        ],
        clean_dep("//tensorflow:windows"): [
            clean_dep(
                "//tensorflow/python:_pywrap_tensorflow_internal_windows",
            ),
        ],
        "//conditions:default": [
            clean_dep(
                "//tensorflow/python:_pywrap_tensorflow_internal_linux",
            ),
        ],
    })

# Helper function for the per-OS tensorflow libraries and their version symlinks
def tf_shared_library_deps():
    return select({
        clean_dep("//tensorflow:macos_with_framework_shared_object"): [
            clean_dep("//tensorflow:libtensorflow.dylib"),
            clean_dep("//tensorflow:libtensorflow.%s.dylib" % VERSION_MAJOR),
            clean_dep("//tensorflow:libtensorflow.%s.dylib" % VERSION),
        ],
        clean_dep("//tensorflow:macos"): [],
        clean_dep("//tensorflow:windows"): [
            clean_dep("//tensorflow:tensorflow.dll"),
            clean_dep("//tensorflow:tensorflow_dll_import_lib"),
        ],
        clean_dep("//tensorflow:framework_shared_object"): [
            clean_dep("//tensorflow:libtensorflow.so"),
            clean_dep("//tensorflow:libtensorflow.so.%s" % VERSION_MAJOR),
            clean_dep("//tensorflow:libtensorflow.so.%s" % VERSION),
        ],
        "//conditions:default": [],
    }) + tf_binary_additional_srcs()

# Helper functions to add kernel dependencies to tf binaries when using dynamic
# kernel linking.
def tf_binary_dynamic_kernel_dsos():
    return if_dynamic_kernels(
        extra_deps = [
            # TODO(gunan): Remove dependencies on these, and make them load dynamically.
            # "//tensorflow/core/kernels:libtfkernel_all_kernels.so",
        ],
        otherwise = [],
    )

# Helper functions to add kernel dependencies to tf binaries when using static
# kernel linking.
def tf_binary_dynamic_kernel_deps(kernels):
    return if_dynamic_kernels(
        extra_deps = [],
        otherwise = kernels,
    )

# Shared libraries have different name pattern on different platforms,
# but cc_binary cannot output correct artifact name yet,
# so we generate multiple cc_binary targets with all name patterns when necessary.
# TODO(pcloudy): Remove this workaround when https://github.com/bazelbuild/bazel/issues/4570
# is done and cc_shared_library is available.
SHARED_LIBRARY_NAME_PATTERN_LINUX = "lib%s.so%s"
SHARED_LIBRARY_NAME_PATTERN_MACOS = "lib%s%s.dylib"
SHARED_LIBRARY_NAME_PATTERN_WINDOWS = "%s%s.dll"
SHARED_LIBRARY_NAME_PATTERNS = [
    SHARED_LIBRARY_NAME_PATTERN_LINUX,
    SHARED_LIBRARY_NAME_PATTERN_MACOS,
    SHARED_LIBRARY_NAME_PATTERN_WINDOWS,
]

def tf_cc_shared_object(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = lrt_if_needed(),
        framework_so = tf_binary_additional_srcs(),
        soversion = None,
        kernels = [],
        per_os_targets = False,  # Generate targets with SHARED_LIBRARY_NAME_PATTERNS
        visibility = None,
        **kwargs):
    """Configure the shared object (.so) file for TensorFlow."""
    if soversion != None:
        suffix = "." + str(soversion).split(".")[0]
        longsuffix = "." + str(soversion)
    else:
        suffix = ""
        longsuffix = ""

    if per_os_targets:
        names = [
            (
                pattern % (name, ""),
                pattern % (name, suffix),
                pattern % (name, longsuffix),
            )
            for pattern in SHARED_LIBRARY_NAME_PATTERNS
        ]
    else:
        names = [(
            name,
            name + suffix,
            name + longsuffix,
        )]

    testonly = kwargs.pop("testonly", False)

    for name_os, name_os_major, name_os_full in names:
        # Windows DLLs cant be versioned
        if name_os.endswith(".dll"):
            name_os_major = name_os
            name_os_full = name_os

        if name_os != name_os_major:
            native.genrule(
                name = name_os + "_sym",
                outs = [name_os],
                srcs = [name_os_major],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )
            native.genrule(
                name = name_os_major + "_sym",
                outs = [name_os_major],
                srcs = [name_os_full],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )

        soname = name_os_major.split("/")[-1]

        data_extra = []
        if framework_so != []:
            data_extra = tf_binary_additional_data_deps()

        cc_binary(
            exec_properties = if_google({"cpp_link.mem": "16g"}, {}),
            name = name_os_full,
            srcs = srcs + framework_so,
            deps = deps,
            linkshared = 1,
            data = data + data_extra,
            linkopts = linkopts + _rpath_linkopts(name_os_full) + select({
                clean_dep("//tensorflow:ios"): [
                    "-Wl,-install_name,@rpath/" + soname,
                ],
                clean_dep("//tensorflow:macos"): [
                    "-Wl,-install_name,@rpath/" + soname,
                ],
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-Wl,-soname," + soname,
                ],
            }),
            testonly = testonly,
            visibility = visibility,
            **kwargs
        )

    flat_names = [item for sublist in names for item in sublist]
    if name not in flat_names:
        native.filegroup(
            name = name,
            srcs = select({
                clean_dep("//tensorflow:windows"): [":%s.dll" % (name)],
                clean_dep("//tensorflow:macos"): [":lib%s%s.dylib" % (name, longsuffix)],
                "//conditions:default": [":lib%s.so%s" % (name, longsuffix)],
            }),
            visibility = visibility,
            testonly = testonly,
        )

# buildozer: disable=function-docstring-args
def tf_cc_shared_library_opensource(
        name,
        srcs = [],
        dynamic_deps = [],
        static_deps = [],
        deps = [],
        roots = [],
        exports_filter = [],
        data = [],
        copts = [],
        linkopts = lrt_if_needed(),
        additional_linker_inputs = [],
        linkstatic = True,
        framework_so = [clean_dep("//tensorflow:libtensorflow_framework_import_lib")],
        soversion = None,
        per_os_targets = False,  # TODO(rostam): Should be deprecated.
        win_def_file = None,
        visibility = None):
    """Configures the shared object file for TensorFlow."""
    names = _get_shared_library_name_os_version_matrix(
        name,
        per_os_targets = per_os_targets,
        version = soversion,
    )
    for name_os, name_os_major, name_os_full in names:
        soname = name_os_major.split("/")[-1]  # Uses major version for soname.
        user_link_flags = linkopts + _rpath_user_link_flags(name_os_full) + select({
            clean_dep("//tensorflow:ios"): [
                "-Wl,-install_name,@rpath/" + soname,
            ],
            clean_dep("//tensorflow:macos"): [
                "-Wl,-install_name,@rpath/" + soname,
            ],
            clean_dep("//tensorflow:windows"): [],
            "//conditions:default": [
                "-Wl,-soname," + soname,
            ],
        })
        _tf_cc_shared_library_opensource(
            name_os_full,
            additional_linker_inputs = additional_linker_inputs,
            copts = copts,
            data = data,
            deps = deps + framework_so,
            dynamic_deps = dynamic_deps,
            exports_filter = exports_filter,
            linkstatic = linkstatic,
            roots = roots,
            shared_lib_name = name_os_full,
            srcs = srcs,
            static_deps = static_deps,
            user_link_flags = user_link_flags,
            visibility = visibility,
            win_def_file = win_def_file,
        )

        if name_os != name_os_major:
            filegroup_name = name_os_full + "_filegroup"
            filegroup(
                name = filegroup_name,
                srcs = [name_os_full],
                output_group = "main_shared_library_output",
                visibility = visibility,
            )
            _create_symlink(name_os, name_os_major, visibility = visibility)
            _create_symlink(name_os_major, filegroup_name, visibility = visibility)

    if name not in [item for sublist in names for item in sublist]:
        native.filegroup(
            name = name,
            srcs = select({
                clean_dep("//tensorflow:windows"): [":%s" % get_versioned_shared_library_name_windows(name, soversion)],
                clean_dep("//tensorflow:macos"): [":%s" % get_versioned_shared_library_name_macos(name, soversion)],
                "//conditions:default": [":%s" % get_versioned_shared_library_name_linux(name, soversion)],
            }),
            visibility = visibility,
        )

def _tf_cc_shared_library_opensource(
        name,
        additional_linker_inputs = None,
        copts = None,
        data = None,
        deps = None,
        dynamic_deps = None,
        exports_filter = None,
        linkstatic = False,
        roots = None,
        shared_lib_name = None,
        srcs = None,
        static_deps = None,
        user_link_flags = None,
        visibility = None,
        win_def_file = None):
    cc_library_name = name + "_cclib"
    cc_library(
        name = cc_library_name,
        srcs = srcs,
        data = data,
        deps = deps,
        copts = copts,
        linkstatic = linkstatic,
    )
    cc_shared_library(
        name = name,
        roots = [cc_library_name] + roots,
        exports_filter = exports_filter,
        dynamic_deps = dynamic_deps,
        static_deps = static_deps,
        shared_lib_name = shared_lib_name,
        user_link_flags = user_link_flags,
        additional_linker_inputs = additional_linker_inputs,
        visibility = visibility,
        win_def_file = if_windows(win_def_file, otherwise = None),
    )

def _create_symlink(src, dest, visibility = None):
    native.genrule(
        name = src + "_sym",
        outs = [src],
        srcs = [dest],
        output_to_bindir = 1,
        cmd = "ln -sf $$(realpath --relative-to=$(RULEDIR) $<) $@",
        visibility = visibility,
    )

def _get_shared_library_name_os_version_matrix(name, per_os_targets = False, version = None):
    if per_os_targets:
        names = [
            (get_versioned_shared_library_name_linux(name), get_versioned_shared_library_name_linux(name, version, True), get_versioned_shared_library_name_linux(name, version)),
            (get_versioned_shared_library_name_macos(name), get_versioned_shared_library_name_macos(name, version, True), get_versioned_shared_library_name_macos(name, version)),
            (get_versioned_shared_library_name_windows(name), get_versioned_shared_library_name_windows(name, version, True), get_versioned_shared_library_name_windows(name, version)),
        ]
    else:
        names = [(name, name + get_suffix_major_version(version), name + get_suffix_version(version))]
    return names

def get_versioned_shared_library_name_linux(name, version = None, major = False):
    if major:
        name = SHARED_LIBRARY_NAME_PATTERN_LINUX % (name, get_suffix_major_version(version))
    else:
        name = SHARED_LIBRARY_NAME_PATTERN_LINUX % (name, get_suffix_version(version))
    return name

def get_versioned_shared_library_name_macos(name, version = None, major = False):
    if major:
        name = SHARED_LIBRARY_NAME_PATTERN_MACOS % (name, get_suffix_major_version(version))
    else:
        name = SHARED_LIBRARY_NAME_PATTERN_MACOS % (name, get_suffix_version(version))
    return name

def get_versioned_shared_library_name_windows(name, version = None, major = False):
    _ = version  # buildifier: disable=unused-variable
    _ = major  # buildifier: disable=unused-variable
    return SHARED_LIBRARY_NAME_PATTERN_WINDOWS % (name, "")

def get_suffix_version(version):
    return "." + str(version) if version else ""

def get_suffix_major_version(version):
    return "." + extract_major_version(version) if version else ""

def extract_major_version(version):
    return str(version).split(".", 1)[0]

# Export open source version of tf_cc_shared_library under base name as well.
tf_cc_shared_library = tf_cc_shared_library_opensource

# Links in the framework shared object
# (//third_party/tensorflow:libtensorflow_framework.so) when not building
# statically. Also adds linker options (rpaths) so that the framework shared
# object can be found.
def tf_cc_binary(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = lrt_if_needed(),
        copts = tf_copts(),
        kernels = [],
        per_os_targets = False,  # Generate targets with SHARED_LIBRARY_NAME_PATTERNS
        visibility = None,
        default_copts = [],
        **kwargs):
    if kernels:
        added_data_deps = tf_binary_dynamic_kernel_dsos()
    else:
        added_data_deps = []

    if per_os_targets:
        names = [pattern % (name, "") for pattern in SHARED_LIBRARY_NAME_PATTERNS]
    else:
        names = [name]

    # Optional MKL dependency, we also tell buildcleaner to ignore this dep using a tag.
    mkl_dep = if_mkl_ml([clean_dep("//third_party/mkl:intel_binary_blob")])
    tags = kwargs.pop("tags", []) + ["req_dep=" + clean_dep("//third_party/mkl:intel_binary_blob")]

    for name_os in names:
        cc_binary(
            name = name_os,
            copts = default_copts + copts,
            srcs = srcs + tf_binary_additional_srcs(),
            deps = deps + tf_binary_dynamic_kernel_deps(kernels) + mkl_dep + if_static(
                extra_deps = [],
                otherwise = [
                    clean_dep("//tensorflow:libtensorflow_framework_import_lib"),
                ],
            ),
            tags = tags,
            data = depset(data + added_data_deps),
            linkopts = linkopts + _rpath_linkopts(name_os),
            visibility = visibility,
            **kwargs
        )
    if name not in names:
        native.filegroup(
            name = name,
            srcs = select({
                "//tensorflow:windows": [":%s.dll" % name],
                "//tensorflow:macos": [":lib%s.dylib" % name],
                "//conditions:default": [":lib%s.so" % name],
            }),
            visibility = visibility,
        )

register_extension_info(
    extension = tf_cc_binary,
    label_regex_for_dep = "{extension_name}",
)

# A simple wrap around native.cc_binary rule.
# When using this rule, you should realize it doesn't link to any tensorflow
# dependencies by default.
def tf_native_cc_binary(
        name,
        copts = tf_copts(),
        linkopts = [],
        **kwargs):
    cc_binary(
        name = name,
        copts = copts,
        linkopts = select({
            clean_dep("//tensorflow:windows"): [],
            clean_dep("//tensorflow:macos"): [
                "-lm",
            ],
            "//conditions:default": [
                "-lpthread",
                "-lm",
            ],
        }) + linkopts + _rpath_linkopts(name) + lrt_if_needed(),
        **kwargs
    )

def tf_gen_op_wrapper_cc(
        name,
        out_ops_file,
        pkg = "",
        op_gen = clean_dep("//tensorflow/cc:cc_op_gen_main"),
        deps = None,
        include_internal_ops = 0,
        # ApiDefs will be loaded in the order specified in this list.
        api_def_srcs = [],
        compatible_with = []):
    # Construct an op generator binary for these ops.
    tool = out_ops_file + "_gen_cc"
    if deps == None:
        deps = [pkg + ":" + name + "_op_lib"]
    tf_cc_binary(
        name = tool,
        copts = tf_copts(),
        linkopts = if_not_windows(["-lm", "-Wl,-ldl"]) + lrt_if_needed(),
        linkstatic = 1,  # Faster to link this one-time-use binary dynamically
        deps = [op_gen] + deps,
    )

    srcs = api_def_srcs[:]

    if not api_def_srcs:
        api_def_args_str = ","
    else:
        api_def_args = []
        for api_def_src in api_def_srcs:
            # Add directory of the first ApiDef source to args.
            # We are assuming all ApiDefs in a single api_def_src are in the
            # same directory.
            api_def_args.append(
                " $$(dirname $$(echo $(locations " + api_def_src +
                ") | cut -d\" \" -f1))",
            )
        api_def_args_str = ",".join(api_def_args)

    native.genrule(
        name = name + "_genrule",
        outs = [
            out_ops_file + ".h",
            out_ops_file + ".cc",
            out_ops_file + "_internal.h",
            out_ops_file + "_internal.cc",
        ],
        srcs = srcs,
        tools = [":" + tool] + tf_binary_additional_srcs(),
        cmd = ("$(location :" + tool + ") $(location :" + out_ops_file + ".h) " +
               "$(location :" + out_ops_file + ".cc) " +
               str(include_internal_ops) + " " + api_def_args_str),
        compatible_with = compatible_with,
    )

# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate individual C++ .cc and .h
# files for each of the ops files mentioned, and then generate a
# single cc_library called "name" that combines all the
# generated C++ code.
#
# For example, for:
#  tf_gen_op_wrappers_cc("tf_ops_lib", [ "array_ops", "math_ops" ])
#
#
# This will ultimately generate ops/* files and a library like:
#
# cc_library(name = "tf_ops_lib",
#            srcs = [ "ops/array_ops.cc",
#                     "ops/math_ops.cc" ],
#            hdrs = [ "ops/array_ops.h",
#                     "ops/math_ops.h" ],
#            deps = [ ... ])
#
# Plus a private library for the "hidden" ops.
# cc_library(name = "tf_ops_lib_internal",
#            srcs = [ "ops/array_ops_internal.cc",
#                     "ops/math_ops_internal.cc" ],
#            hdrs = [ "ops/array_ops_internal.h",
#                     "ops/math_ops_internal.h" ],
#            deps = [ ... ])
# TODO(joshl): Cleaner approach for hidden ops.
def tf_gen_op_wrappers_cc(
        name,
        op_lib_names = [],
        other_srcs = [],
        other_hdrs = [],
        other_srcs_internal = [],
        other_hdrs_internal = [],
        pkg = "",
        deps = [
            clean_dep("//tensorflow/cc:ops"),
            clean_dep("//tensorflow/cc:scope"),
            clean_dep("//tensorflow/cc:const_op"),
        ],
        deps_internal = [],
        op_gen = clean_dep("//tensorflow/cc:cc_op_gen_main"),
        include_internal_ops = 0,
        visibility = None,
        # ApiDefs will be loaded in the order specified in this list.
        api_def_srcs = [],
        # Any extra dependencies that the wrapper generator might need.
        extra_gen_deps = [],
        compatible_with = []):
    subsrcs = other_srcs[:]
    subhdrs = other_hdrs[:]
    internalsrcs = other_srcs_internal[:]
    internalhdrs = other_hdrs_internal[:]
    for n in op_lib_names:
        tf_gen_op_wrapper_cc(
            n,
            "ops/" + n,
            api_def_srcs = api_def_srcs,
            include_internal_ops = include_internal_ops,
            op_gen = op_gen,
            pkg = pkg,
            deps = [pkg + ":" + n + "_op_lib"] + extra_gen_deps,
            compatible_with = compatible_with,
        )
        subsrcs += ["ops/" + n + ".cc"]
        subhdrs += ["ops/" + n + ".h"]
        internalsrcs += ["ops/" + n + "_internal.cc"]
        internalhdrs += ["ops/" + n + "_internal.h"]

    cc_library(
        name = name,
        srcs = subsrcs,
        hdrs = subhdrs,
        deps = deps + if_not_android([
            clean_dep("//tensorflow/core:core_cpu"),
            clean_dep("//tensorflow/core:framework"),
            clean_dep("//tensorflow/core:lib"),
            clean_dep("//tensorflow/core:ops"),
            clean_dep("//tensorflow/core:protos_all_cc"),
        ]) + if_android([
            clean_dep("//tensorflow/core:portable_tensorflow_lib"),
        ]),
        copts = tf_copts(),
        alwayslink = 1,
        visibility = visibility,
        compatible_with = compatible_with,
    )
    cc_library(
        name = name + "_internal",
        srcs = internalsrcs,
        hdrs = internalhdrs,
        deps = deps + deps_internal + if_not_android([
            clean_dep("//tensorflow/core:core_cpu"),
            clean_dep("//tensorflow/core:framework"),
            clean_dep("//tensorflow/core:lib"),
            clean_dep("//tensorflow/core:ops"),
            clean_dep("//tensorflow/core:protos_all_cc"),
        ]) + if_android([
            clean_dep("//tensorflow/core:portable_tensorflow_lib"),
        ]),
        copts = tf_copts(),
        alwayslink = 1,
        visibility = [clean_dep("//tensorflow:internal")],
        compatible_with = compatible_with,
    )

OpRegistrationSrcInfo = provider(
    "Info needed to extract op registration sources.",
    fields = {
        "srcs": "depset of source Files that contains op registrations.",
    },
)

def _collect_op_reg_srcs_aspect_impl(_target, ctx):
    """Aspect implementation function for collect_op_reg_srcs_aspect.

    This aspect will traverse the dependency graph along the "deps" attribute of the target
    and return an OpRegistrationSrcInfo provider.

    OpRegistrationSrcInfo will have the union of the srcs of the C++ dependencies
    with filename end with "_ops.cc" or "_op.cc".
    """
    direct, transitive = [], []
    if ctx.rule.kind == "cc_library" and hasattr(ctx.rule.attr, "srcs"):
        # Assuming the filename of op registration source files ends with "_ops.cc" or "_op.cc"
        direct += [
            src
            for src in ctx.rule.files.srcs
            if src.path.endswith("_op.cc") or src.path.endswith("_ops.cc")
        ]
    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if OpRegistrationSrcInfo in dep:
                transitive.append(dep[OpRegistrationSrcInfo].srcs)
    if not direct and not transitive:
        return []
    return [OpRegistrationSrcInfo(srcs = depset(direct = direct, transitive = transitive))]

collect_op_reg_srcs_aspect = aspect(
    attr_aspects = ["deps"],
    required_providers = [CcInfo],
    implementation = _collect_op_reg_srcs_aspect_impl,
)

def _generate_op_reg_offsets_impl(ctx):
    op_reg_srcs = []
    for dep in ctx.attr.deps:
        if OpRegistrationSrcInfo in dep:
            for src in dep[OpRegistrationSrcInfo].srcs.to_list():
                op_reg_srcs.append(src)

    args = ctx.actions.args()
    args.add(ctx.outputs.out.path, format = "--out_path=%s")
    args.add_all(op_reg_srcs)

    ctx.actions.run(
        outputs = [ctx.outputs.out],
        inputs = op_reg_srcs + ctx.files.tf_binary_additional_srcs,
        tools = [ctx.executable._offset_counter],
        executable = ctx.executable._offset_counter,
        arguments = [args],
    )

generate_op_reg_offsets = rule(
    attrs = {
        "out": attr.output(),
        "deps": attr.label_list(
            aspects = [collect_op_reg_srcs_aspect],
            mandatory = True,
            allow_files = True,
            providers = [CcInfo],
        ),
        # This is for carrying the required files for _offset_counter to execute.
        "tf_binary_additional_srcs": attr.label_list(
            cfg = "exec",
            mandatory = True,
            allow_files = True,
        ),
        "_offset_counter": attr.label(
            cfg = "exec",
            executable = True,
            allow_files = True,
            default = "//tensorflow/python/framework:offset_counter",
        ),
    },
    implementation = _generate_op_reg_offsets_impl,
)

# Generates a Python library target wrapping the ops registered in "deps".
#
# Args:
#   name: used as the name of the generated target and as a name component of
#     the intermediate files.
#   out: name of the python file created by this rule. If None, then
#     "ops/gen_{name}.py" is used.
#   hidden: Optional list of ops names to make private in the Python module.
#     It is invalid to specify both "hidden" and "op_allowlist".
#   visibility: passed to py_library.
#   deps: list of dependencies for the intermediate tool used to generate the
#     python target. NOTE these `deps` are not applied to the final python
#     library target itself.
#   require_shape_functions: Unused. Leave this as False.
#   hidden_file: optional file that contains a list of op names to make private
#     in the generated Python module. Each op name should be on a line by
#     itself. Lines that start with characters that are invalid op name
#     starting characters are treated as comments and ignored.
#   generated_target_name: name of the generated target (overrides the
#     "name" arg)
#   op_whitelist: [DEPRECATED] an older spelling for "op_allowlist"
#   op_allowlist: if not empty, only op names in this list will be wrapped. It
#     is invalid to specify both "hidden" and "op_allowlist".
#   cc_linkopts: Optional linkopts to be added to tf_cc_binary that contains the
#     specified ops.

def tf_gen_op_wrapper_py(
        name,
        out = None,
        hidden = None,
        visibility = None,
        deps = [],
        require_shape_functions = False,
        hidden_file = None,
        generated_target_name = None,
        op_whitelist = [],
        op_allowlist = [],
        cc_linkopts = lrt_if_needed(),
        api_def_srcs = [],
        compatible_with = [],
        testonly = False,
        copts = []):
    _ = require_shape_functions  # Unused.
    if op_whitelist and op_allowlist:
        fail("op_whitelist is deprecated. Only use op_allowlist.")
    if op_whitelist:
        full_target_name = "//" + native.package_name() + ":" + name
        print("op_whitelist is deprecated. Please migrate to the preferred " +
              "`op_allowlist` spelling. Offending target: " +
              full_target_name)  # buildifier: disable=print
        op_allowlist = op_whitelist

    if (hidden or hidden_file) and op_allowlist:
        fail("Cannot pass specify both hidden and op_allowlist.")

    # Construct a cc_binary containing the specified ops.
    tool_name = "gen_" + name + "_py_wrappers_cc"
    if not deps:
        deps = [str(Label("//tensorflow/core:" + name + "_op_lib"))]
    tf_cc_binary(
        name = tool_name,
        copts = copts + tf_copts(),
        linkopts = if_not_windows(["-lm", "-Wl,-ldl"]) + cc_linkopts,
        linkstatic = 1,  # Faster to link this one-time-use binary dynamically
        visibility = [clean_dep("//tensorflow:internal")],
        deps = ([
            clean_dep("//tensorflow/core:framework"),
            clean_dep("//tensorflow/python/framework:python_op_gen_main"),
        ] + deps),
        testonly = testonly,
    )

    pygen_args = []

    # Invoke the previous cc_binary to generate a python file.
    if not out:
        out = "ops/gen_" + name + ".py"

    extra_srcs = []
    if hidden:
        pygen_args.append("--hidden_op_list=" + ",".join(hidden))
    elif hidden_file:
        # `hidden_file` is file containing a list of op names to be hidden in the
        # generated module.
        pygen_args.append("--hidden_op_list_filename=$(location " + hidden_file + ")")
        extra_srcs.append(hidden_file)
    elif op_allowlist:
        pygen_args.append("--op_allowlist=" + ",".join(["'%s'" % op for op in op_allowlist]))

    # Prepare ApiDef directories to pass to the genrule.
    if api_def_srcs:
        api_def_args = []
        for api_def_src in api_def_srcs:
            # Add directory of the first ApiDef source to args.
            # We are assuming all ApiDefs in a single api_def_src are in the
            # same directory.
            api_def_args.append(
                "$$(dirname $$(echo $(locations " + api_def_src +
                ") | cut -d\" \" -f1))",
            )
        pygen_args.append("--api_def_dirs=" + ",".join(api_def_args))

    op_reg_offset_out = "gen_" + name + "_reg_offsets.pb"
    generate_op_reg_offsets(
        name = name + "_reg_offsets",
        out = op_reg_offset_out,
        # Feed an empty dep list if not indexing to skip unnecessary aspect propagation.
        deps = select({
            clean_dep("//tensorflow:api_indexable"): deps,
            "//conditions:default": [],
        }),
        tf_binary_additional_srcs = tf_binary_additional_srcs(),
        testonly = testonly,
    )
    extra_srcs.append(op_reg_offset_out)
    pygen_args.append("--op_reg_offset_filename=$(location " + op_reg_offset_out + ")")

    native.genrule(
        name = name + "_pygenrule",
        outs = [out],
        srcs = api_def_srcs + extra_srcs,
        tools = [tool_name] + tf_binary_additional_srcs(),
        cmd = ("$(location " + tool_name + ") " + " ".join(pygen_args) + " > $@"),
        compatible_with = compatible_with,
        testonly = testonly,
    )

    # Make a py_library out of the generated python file.
    if not generated_target_name:
        generated_target_name = name
    native.py_library(
        name = generated_target_name,
        srcs = [out],
        srcs_version = "PY3",
        visibility = visibility,
        deps = [
            clean_dep("//tensorflow/python:framework_for_generated_wrappers_v2"),
        ],
        # Instruct build_cleaner to try to avoid using this rule; typically ops
        # creators will provide their own tf_custom_op_py_library based target
        # that wraps this one.
        tags = ["avoid_dep"],
        compatible_with = compatible_with,
        testonly = testonly,
    )

# Define a bazel macro that creates cc_test for tensorflow.
#
# Links in the framework shared object
# (//third_party/tensorflow:libtensorflow_framework.so) when not building
# statically. Also adds linker options (rpaths) so that the framework shared
# object can be found.
#
# TODO(opensource): we need to enable this to work around the hidden symbol
# __cudaRegisterFatBinary error. Need more investigations.
def tf_cc_test(
        name,
        srcs,
        deps,
        data = [],
        extra_copts = [],
        suffix = "",
        linkopts = lrt_if_needed(),
        kernels = [],
        **kwargs):
    cc_test(
        name = "%s%s" % (name, suffix),
        srcs = srcs + tf_binary_additional_srcs(),
        copts = tf_copts() + extra_copts,
        linkopts = select({
            clean_dep("//tensorflow:android"): [
                "-pie",
            ],
            clean_dep("//tensorflow:windows"): [],
            clean_dep("//tensorflow:macos"): [
                "-lm",
            ],
            "//conditions:default": [
                "-lpthread",
                "-lm",
            ],
            clean_dep("//third_party/compute_library:build_with_acl"): [
                "-fopenmp",
                "-lm",
            ],
        }) + linkopts + _rpath_linkopts(name),
        deps = deps + tf_binary_dynamic_kernel_deps(kernels) + if_mkl_ml(
            [
                clean_dep("//third_party/mkl:intel_binary_blob"),
            ],
        ),
        data = data +
               tf_binary_dynamic_kernel_dsos() +
               tf_binary_additional_srcs(),
        exec_properties = tf_exec_properties(kwargs),
        **kwargs
    )

def tf_cc_shared_test(
        name,
        srcs,
        deps,
        data = [],
        extra_copts = [],
        suffix = "",
        linkopts = lrt_if_needed(),
        kernels = [],
        **kwargs):
    cc_test(
        name = "%s%s" % (name, suffix),
        srcs = srcs,
        copts = tf_copts() + extra_copts,
        linkopts = select({
            clean_dep("//tensorflow:android"): [
                "-pie",
            ],
            clean_dep("//tensorflow:windows"): [],
            clean_dep("//tensorflow:macos"): [
                "-lm",
            ],
            "//conditions:default": [
                "-lpthread",
                "-lm",
            ],
            clean_dep("//third_party/compute_library:build_with_acl"): [
                "-fopenmp",
                "-lm",
            ],
        }) + linkopts + _rpath_linkopts(name),
        deps = deps + tf_binary_dynamic_kernel_deps(kernels) + if_mkl_ml(
            [
                clean_dep("//third_party/mkl:intel_binary_blob"),
            ],
        ),
        dynamic_deps = if_static(
            extra_deps = [],
            macos = ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
            otherwise = ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
        ),
        data = data + tf_binary_dynamic_kernel_dsos(),
        exec_properties = tf_exec_properties(kwargs),
        **kwargs
    )

register_extension_info(
    extension = tf_cc_test,
    label_regex_for_dep = "{extension_name}",
)

# TODO(jakeharmon): Replace with or implement in terms of tsl_gpu_cc_test, which doesn't add a
# dependency on core:common_runtime
def tf_gpu_cc_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        data = [],
        size = "medium",
        extra_copts = [],
        linkstatic = 0,
        args = [],
        kernels = [],
        linkopts = [],
        **kwargs):
    targets = []
    tf_cc_test(
        name = name,
        size = size,
        srcs = srcs,
        args = args,
        data = data,
        extra_copts = extra_copts + if_cuda(["-DNV_CUDNN_DISABLE_EXCEPTION"]),
        kernels = kernels,
        linkopts = linkopts,
        linkstatic = linkstatic,
        suffix = "_cpu",
        tags = tags,
        deps = deps,
        **kwargs
    )
    targets.append(name + "_cpu")
    tf_cc_test(
        name = name,
        size = size,
        srcs = srcs,
        args = args,
        data = data,
        extra_copts = extra_copts + if_cuda(["-DNV_CUDNN_DISABLE_EXCEPTION"]),
        kernels = kernels,
        linkopts = linkopts,
        linkstatic = select({
            # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
            clean_dep("//tensorflow:macos"): 1,
            "@local_config_cuda//cuda:using_nvcc": 1,
            "@local_config_cuda//cuda:using_clang": 1,
            "//conditions:default": 0,
        }),
        suffix = "_gpu",
        tags = tags + tf_gpu_tests_tags(),
        deps = deps + if_cuda_or_rocm([
            clean_dep("//tensorflow/core:gpu_runtime"),
        ]),
        **kwargs
    )
    targets.append(name + "_gpu")
    if "multi_gpu" in tags or "multi_and_single_gpu" in tags:
        cleaned_tags = tags + two_gpu_tags
        if "requires-gpu-nvidia" in cleaned_tags:
            cleaned_tags.remove("requires-gpu-nvidia")
        tf_cc_test(
            name = name,
            size = size,
            srcs = srcs,
            args = args,
            data = data,
            extra_copts = extra_copts,
            kernels = kernels,
            linkopts = linkopts,
            linkstatic = select({
                # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
                clean_dep("//tensorflow:macos"): 1,
                "@local_config_cuda//cuda:using_nvcc": 1,
                "@local_config_cuda//cuda:using_clang": 1,
                "//conditions:default": 0,
            }),
            suffix = "_2gpu",
            tags = cleaned_tags,
            deps = deps + if_cuda_or_rocm([
                clean_dep("//tensorflow/core:gpu_runtime"),
            ]),
            **kwargs
        )
        targets.append(name + "_2gpu")

    native.test_suite(name = name, tests = targets, tags = tags)

# terminology changes: saving tf_cuda_* definition for compatibility
def tf_cuda_cc_test(*args, **kwargs):
    tf_gpu_cc_test(*args, **kwargs)

def tf_gpu_only_cc_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        data = [],
        size = "medium",
        args = [],
        kernels = [],
        linkopts = []):
    tags = tags + tf_gpu_tests_tags()

    gpu_lib_name = "%s%s" % (name, "_gpu_lib")
    tf_gpu_kernel_library(
        name = gpu_lib_name,
        srcs = srcs + tf_binary_additional_srcs(),
        deps = deps,
        testonly = 1,
    )
    cc_test(
        name = "%s%s" % (name, "_gpu"),
        size = size,
        args = args,
        features = if_cuda(["-use_header_modules"]),
        data = data + tf_binary_dynamic_kernel_dsos(),
        deps = [":" + gpu_lib_name],
        linkopts = if_not_windows(["-lpthread", "-lm"]) + linkopts + _rpath_linkopts(name),
        tags = tags,
        exec_properties = tf_exec_properties({"tags": tags}),
    )

# terminology changes: saving tf_cuda_* definition for compatibility
def tf_cuda_only_cc_test(*args, **kwargs):
    tf_gpu_only_cc_test(*args, **kwargs)

# Create a cc_test for each of the tensorflow tests listed in "tests", along
# with a test suite of the given name, if provided.
def tf_cc_tests(
        srcs,
        deps,
        name = "",
        linkstatic = 0,
        tags = [],
        size = "medium",
        args = None,
        linkopts = lrt_if_needed(),
        kernels = [],
        create_named_test_suite = False,
        visibility = None):
    test_names = []
    for src in srcs:
        test_name = src_to_test_name(src)
        tf_cc_test(
            name = test_name,
            size = size,
            srcs = [src],
            args = args,
            kernels = kernels,
            linkopts = linkopts,
            linkstatic = linkstatic,
            tags = tags,
            deps = deps,
            visibility = visibility,
        )
        test_names.append(test_name)

    # Add a test suite with the generated tests if a name was provided and
    # it does not conflict any of the test names.
    if create_named_test_suite:
        native.test_suite(
            name = name,
            tests = test_names,
            visibility = visibility,
            tags = tags,
        )

register_extension_info(
    extension = tf_cc_tests,
    label_regex_for_dep = "{extension_name}",
)

def tf_cc_test_mkl(
        srcs,
        deps,
        name = "",
        data = [],
        linkstatic = 0,
        tags = [],
        size = "medium",
        kernels = [],
        args = None):
    # -fno-exceptions in nocopts breaks compilation if header modules are enabled.
    disable_header_modules = ["-use_header_modules"]

    for src in srcs:
        cc_test(
            name = src_to_test_name(src),
            srcs = if_mkl([src]) + tf_binary_additional_srcs(),
            # Adding an explicit `-fexceptions` because `allow_exceptions = True`
            # in `tf_copts` doesn't work internally.
            copts = tf_copts() + ["-fexceptions"] + tf_openmp_copts(),
            linkopts = select({
                clean_dep("//tensorflow:android"): [
                    "-pie",
                ],
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-lpthread",
                    "-lm",
                ],
            }) + _rpath_linkopts(src_to_test_name(src)) + tf_openmp_lopts(),
            deps = deps + tf_binary_dynamic_kernel_deps(kernels) + if_mkl_ml(["//third_party/mkl:intel_binary_blob"]),
            data = data + tf_binary_dynamic_kernel_dsos(),
            exec_properties = tf_exec_properties({"tags": tags}),
            linkstatic = linkstatic,
            tags = tags,
            size = size,
            args = args,
            features = disable_header_modules,
        )

def tf_gpu_cc_tests(
        srcs,
        deps,
        name = "",
        tags = [],
        size = "medium",
        linkstatic = 0,
        args = None,
        kernels = [],
        linkopts = []):
    for src in srcs:
        tf_gpu_cc_test(
            name = src_to_test_name(src),
            size = size,
            srcs = [src],
            args = args,
            kernels = kernels,
            linkopts = linkopts,
            linkstatic = linkstatic,
            tags = tags,
            deps = deps,
        )

# terminology changes: saving tf_cuda_* definition for compatibility
def tf_cuda_cc_tests(*args, **kwargs):
    tf_gpu_cc_tests(*args, **kwargs)

def tf_java_test(
        name,
        srcs = [],
        deps = [],
        kernels = [],
        *args,
        **kwargs):
    cc_library_name = name + "_cclib"
    cc_library(
        # TODO(b/183579145): Remove when cc_shared_library supports CcInfo or JavaInfo providers .
        name = cc_library_name,
        srcs = tf_binary_additional_srcs(fullversion = True) + tf_binary_dynamic_kernel_dsos() + tf_binary_dynamic_kernel_deps(kernels),
    )
    native.java_test(
        name = name,
        srcs = srcs,
        deps = deps + [cc_library_name],
        *args,
        **kwargs
    )

def _cuda_copts(opts = []):
    """Gets the appropriate set of copts for (maybe) CUDA compilation.

        If we're doing CUDA compilation, returns copts for our particular CUDA
        compiler.  If we're not doing CUDA compilation, returns an empty list.

        """
    return select({
        "//conditions:default": [],
        "@local_config_cuda//cuda:using_nvcc": [
            "-nvcc_options=relaxed-constexpr",
            "-nvcc_options=ftz=true",
        ] + opts,
        "@local_config_cuda//cuda:using_clang": [
            "-fcuda-flush-denormals-to-zero",
        ] + opts,
    })

# Build defs for TensorFlow kernels

# When this target is built using --config=cuda, a cc_library is built
# that passes -DGOOGLE_CUDA=1 and '-x cuda', linking in additional
# libraries needed by GPU kernels.
#
# When this target is built using --config=rocm, a cc_library is built
# that passes -DTENSORFLOW_USE_ROCM and '-x rocm', linking in additional
# libraries needed by GPU kernels.
def tf_gpu_kernel_library(
        srcs,
        copts = [],
        cuda_copts = [],
        deps = [],
        hdrs = [],
        **kwargs):
    copts = copts + tf_copts() + _cuda_copts(opts = cuda_copts) + rocm_copts(opts = cuda_copts)
    kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]

    cuda_library(
        srcs = srcs,
        hdrs = hdrs,
        copts = copts,
        deps = deps + if_cuda([
            clean_dep("//tensorflow/tsl/cuda:cudart_stub"),
        ]) + if_cuda_or_rocm([
            clean_dep("//tensorflow/core:gpu_lib"),
        ]),
        alwayslink = 1,
        **kwargs
    )

tf_gpu_library = tsl_gpu_library

# terminology changes: saving tf_cuda_* definition for compatibility
tf_cuda_library = tsl_gpu_library

def tf_kernel_library(
        name,
        prefix = None,
        srcs = None,
        gpu_srcs = None,
        hdrs = None,
        deps = None,
        gpu_deps = None,
        alwayslink = 1,
        copts = None,
        gpu_copts = None,
        is_external = False,
        compatible_with = None,
        **kwargs):
    """A rule to build a TensorFlow OpKernel.

      May either specify srcs/hdrs or prefix.  Similar to tf_gpu_library,
      but with alwayslink=1 by default.  If prefix is specified:
        * prefix*.cc (except *.cu.cc) is added to srcs
        * prefix*.h (except *.cu.h) is added to hdrs
        * prefix*.cu.cc and prefix*.h (including *.cu.h) are added to gpu_srcs.
      With the exception that test files are excluded.
      For example, with prefix = "cast_op",
        * srcs = ["cast_op.cc"]
        * hdrs = ["cast_op.h"]
        * gpu_srcs = ["cast_op_gpu.cu.cc", "cast_op.h"]
        * "cast_op_test.cc" is excluded
      With prefix = "cwise_op"
        * srcs = ["cwise_op_abs.cc", ..., "cwise_op_tanh.cc"],
        * hdrs = ["cwise_ops.h", "cwise_ops_common.h"],
        * gpu_srcs = ["cwise_op_gpu_abs.cu.cc", ..., "cwise_op_gpu_tanh.cu.cc",
                      "cwise_ops.h", "cwise_ops_common.h",
                      "cwise_ops_gpu_common.cu.h"]
        * "cwise_ops_test.cc" is excluded
      """
    if not srcs:
        srcs = []
    if not hdrs:
        hdrs = []
    if not deps:
        deps = []
    if not gpu_deps:
        gpu_deps = []
    if not copts:
        copts = []
    if not gpu_copts:
        gpu_copts = []
    textual_hdrs = []
    copts = copts + tf_copts(is_external = is_external) + if_cuda(["-DNV_CUDNN_DISABLE_EXCEPTION"])

    # Override EIGEN_STRONG_INLINE to inline when
    # --define=override_eigen_strong_inline=true to avoid long compiling time.
    # See https://github.com/tensorflow/tensorflow/issues/10521
    copts = copts + if_override_eigen_strong_inline(["/DEIGEN_STRONG_INLINE=inline"])
    if prefix:
        if native.glob([prefix + "*.cu.cc"], exclude = ["*test*"]):
            if not gpu_srcs:
                gpu_srcs = []
            gpu_srcs = gpu_srcs + native.glob(
                [prefix + "*.cu.cc", prefix + "*.h"],
                exclude = [prefix + "*test*"],
            )
        srcs = srcs + native.glob(
            [prefix + "*.cc"],
            exclude = [prefix + "*test*", prefix + "*.cu.cc"],
        )
        hdrs = hdrs + native.glob(
            [prefix + "*.h"],
            exclude = [prefix + "*test*", prefix + "*.cu.h", prefix + "*impl.h"],
        )
        textual_hdrs = native.glob(
            [prefix + "*impl.h"],
            exclude = [prefix + "*test*", prefix + "*.cu.h"],
        )
    cuda_deps = [clean_dep("//tensorflow/core:gpu_lib")]
    if gpu_srcs:
        for gpu_src in gpu_srcs:
            if gpu_src.endswith(".cc") and not gpu_src.endswith(".cu.cc"):
                fail("{} not allowed in gpu_srcs. .cc sources must end with .cu.cc"
                    .format(gpu_src))
        tf_gpu_kernel_library(
            name = name + "_gpu",
            srcs = gpu_srcs,
            deps = deps + gpu_deps,
            copts = gpu_copts,
            **kwargs
        )
        cuda_deps.extend([":" + name + "_gpu"])
    kwargs["tags"] = kwargs.get("tags", []) + [
        "req_dep=%s" % clean_dep("//tensorflow/core:gpu_lib"),
        "req_dep=@local_config_cuda//cuda:cuda_headers",
    ]
    tf_gpu_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        textual_hdrs = textual_hdrs,
        copts = copts,
        cuda_deps = cuda_deps + gpu_deps,
        linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
        alwayslink = alwayslink,
        deps = deps,
        compatible_with = compatible_with,
        **kwargs
    )

    # TODO(gunan): CUDA dependency not clear here. Fix it.
    tf_cc_shared_object(
        name = "libtfkernel_%s.so" % name,
        srcs = srcs + hdrs,
        copts = copts,
        tags = ["manual", "notap"],
        deps = deps,
    )

register_extension_info(
    extension = tf_kernel_library,
    label_regex_for_dep = "{extension_name}",
)

def tf_mkl_kernel_library(
        name,
        prefix = None,
        srcs = None,
        hdrs = None,
        deps = None,
        alwayslink = 1,
        # Adding an explicit `-fexceptions` because `allow_exceptions = True`
        # in `tf_copts` doesn't work internally.
        copts = tf_copts() + ["-fexceptions"] + tf_openmp_copts(),
        linkopts = tf_openmp_lopts()):
    """A rule to build MKL-based TensorFlow kernel libraries."""

    if not bool(srcs):
        srcs = []
    if not bool(hdrs):
        hdrs = []

    if prefix:
        srcs = srcs + native.glob(
            [prefix + "*.cc"],
            exclude = [prefix + "*test*"],
        )
        hdrs = hdrs + native.glob(
            [prefix + "*.h"],
            exclude = [prefix + "*test*"],
        )

    # -fno-exceptions in nocopts breaks compilation if header modules are enabled.
    disable_header_modules = ["-use_header_modules"]

    cc_library(
        name = name,
        srcs = if_mkl(srcs),
        hdrs = hdrs,
        deps = deps,
        linkopts = linkopts,
        alwayslink = alwayslink,
        copts = copts + if_override_eigen_strong_inline(["/DEIGEN_STRONG_INLINE=inline"]),
        features = disable_header_modules,
    )

def _get_transitive_headers(hdrs, deps):
    """Obtain the header files for a target and its transitive dependencies.

      Args:
        hdrs: a list of header files
        deps: a list of targets that are direct dependencies

      Returns:
        a collection of the transitive headers
      """
    return depset(
        hdrs,
        transitive = [dep[CcInfo].compilation_context.headers for dep in deps],
    )

def _get_repository_roots(ctx, files):
    """Returns abnormal root directories under which files reside.

      When running a ctx.action, source files within the main repository are all
      relative to the current directory; however, files that are generated or exist
      in remote repositories will have their root directory be a subdirectory,
      e.g. bazel-out/local-fastbuild/genfiles/external/jpeg_archive. This function
      returns the set of these devious directories, ranked and sorted by popularity
      in order to hopefully minimize the number of I/O system calls within the
      compiler, because includes have quadratic complexity.
      """
    result = {}
    for f in files.to_list():
        root = f.root.path
        if root:
            if root not in result:
                result[root] = 0
            result[root] -= 1
        work = f.owner.workspace_root
        if work:
            if root:
                root += "/"
            root += work
        if root:
            if root not in result:
                result[root] = 0
            result[root] -= 1
    return [k for v, k in sorted([(v, k) for k, v in result.items()])]

def tf_custom_op_library_additional_deps():
    return [
        "@com_google_protobuf//:protobuf_headers",  # copybara:comment
        clean_dep("//third_party/eigen3"),
        clean_dep("//tensorflow/core:framework_headers_lib"),
    ]

# A list of targets that contains the implementation of
# tf_custom_op_library_additional_deps. It's used to generate a DEF file for
# exporting symbols from _pywrap_tensorflow.dll on Windows.
def tf_custom_op_library_additional_deps_impl():
    return [
        # copybara:comment_begin
        "@com_google_protobuf//:protobuf",
        "@nsync//:nsync_cpp",
        # copybara:comment_end

        # for //third_party/eigen3
        clean_dep("//third_party/eigen3"),

        # for //tensorflow/core:framework_headers_lib
        clean_dep("//tensorflow/core:framework"),
        clean_dep("//tensorflow/core:reader_base"),
    ]

# Traverse the dependency graph along the "deps" attribute of the
# target and return a struct with one field called 'tf_collected_deps'.
# tf_collected_deps will be the union of the deps of the current target
# and the tf_collected_deps of the dependencies of this target.
def _collect_deps_aspect_impl(target, ctx):
    direct, transitive = [], []
    all_deps = []
    if hasattr(ctx.rule.attr, "deps"):
        all_deps += ctx.rule.attr.deps
    if hasattr(ctx.rule.attr, "data"):
        all_deps += ctx.rule.attr.data
    if hasattr(ctx.rule.attr, "roots"):
        all_deps += ctx.rule.attr.roots
    for dep in all_deps:
        direct.append(dep.label)
        if hasattr(dep, "tf_collected_deps"):
            transitive.append(dep.tf_collected_deps)
    return struct(tf_collected_deps = depset(direct = direct, transitive = transitive))

collect_deps_aspect = aspect(
    attr_aspects = ["deps", "data", "roots"],
    implementation = _collect_deps_aspect_impl,
)

def _dep_label(dep):
    label = dep.label
    return label.package + ":" + label.name

# This rule checks that transitive dependencies don't depend on the targets
# listed in the 'disallowed_deps' attribute, but do depend on the targets listed
# in the 'required_deps' attribute. Dependencies considered are targets in the
# 'deps' attribute or the 'data' attribute.
def _check_deps_impl(ctx):
    required_deps = ctx.attr.required_deps
    disallowed_deps = ctx.attr.disallowed_deps
    for input_dep in ctx.attr.deps:
        if not hasattr(input_dep, "tf_collected_deps"):
            continue
        collected_deps = sets.make(input_dep.tf_collected_deps.to_list())
        for disallowed_dep in disallowed_deps:
            if sets.contains(collected_deps, disallowed_dep.label):
                fail(
                    "{src} cannot depend on {dep}. See: bazel query 'somepath(//{src}, //{dep})'".format(
                        src = _dep_label(input_dep),
                        dep = _dep_label(disallowed_dep),
                    ),
                )
        for required_dep in required_deps:
            if not sets.contains(collected_deps, required_dep.label):
                fail(
                    _dep_label(input_dep) + " must depend on " +
                    _dep_label(required_dep),
                )
    return struct()

check_deps = rule(
    _check_deps_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [collect_deps_aspect],
            mandatory = True,
            allow_files = True,
        ),
        "disallowed_deps": attr.label_list(
            default = [],
            allow_files = True,
        ),
        "required_deps": attr.label_list(
            default = [],
            allow_files = True,
        ),
    },
)

def tf_custom_op_library(
        name,
        srcs = [],
        gpu_srcs = [],
        deps = [],
        gpu_deps = None,
        linkopts = [],
        copts = [],
        **kwargs):
    """Helper to build a dynamic library (.so) from the sources containing implementations of custom ops and kernels."""

    if not gpu_deps:
        gpu_deps = []

    deps = deps + if_cuda_or_rocm([
        clean_dep("//tensorflow/core:stream_executor_headers_lib"),
    ]) + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ]) + if_windows([
        clean_dep("//tensorflow/python:pywrap_tensorflow_import_lib"),
    ]) + tf_custom_op_library_additional_deps()

    # Override EIGEN_STRONG_INLINE to inline when
    # --define=override_eigen_strong_inline=true to avoid long compiling time.
    # See https://github.com/tensorflow/tensorflow/issues/10521
    copts = copts + if_override_eigen_strong_inline(["/DEIGEN_STRONG_INLINE=inline"])

    if gpu_srcs:
        basename = name.split(".")[0]
        cuda_library(
            name = basename + "_gpu",
            srcs = gpu_srcs,
            copts = copts + tf_copts() + _cuda_copts() + rocm_copts() +
                    if_tensorrt(["-DGOOGLE_TENSORRT=1"]),
            deps = deps + gpu_deps,
            **kwargs
        )
        deps = deps + [":" + basename + "_gpu"]

    check_deps(
        name = name + "_check_deps",
        disallowed_deps = [
            clean_dep("//tensorflow/core:framework"),
            clean_dep("//tensorflow/core:lib"),
        ],
        deps = deps,
    )
    tf_cc_shared_object(
        name = name,
        srcs = srcs,
        deps = deps,
        data = if_static([name + "_check_deps"]),
        copts = copts + tf_copts(is_external = True),
        features = ["windows_export_all_symbols"],
        linkopts = linkopts + select({
            "//conditions:default": [
                "-lm",
            ],
            clean_dep("//tensorflow:windows"): [],
            clean_dep("//tensorflow:macos"): [],
        }),
        **kwargs
    )

# Placeholder to use until bazel supports py_strict_binary.
def py_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_library.
def py_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_binary.
def pytype_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_library.
def pytype_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_test.
def py_strict_test(name, **kwargs):
    py_test(name = name, **kwargs)

def tf_custom_op_py_library(
        name,
        srcs = [],
        dso = [],
        kernels = [],
        srcs_version = "PY3",
        visibility = None,
        deps = [],
        **kwargs):
    _ignore = [kernels]
    native.py_library(
        name = name,
        data = dso,
        srcs = srcs,
        srcs_version = srcs_version,
        visibility = visibility,
        deps = deps,
        **kwargs
    )

# In tf_py_wrap_cc_opensource generated libraries
# module init functions are not exported unless
# they contain one of the keywords in the version file
# this prevents custom python modules.
# This function attempts to append init_module_name to list of
# exported functions in version script
def _append_init_to_versionscript_impl(ctx):
    mod_name = ctx.attr.module_name
    if ctx.attr.is_version_script:
        ctx.actions.expand_template(
            template = ctx.file.template_file,
            output = ctx.outputs.versionscript,
            substitutions = {
                "global:": "global:\n     init_%s;\n     _init_%s;\n     PyInit_*;\n     _PyInit_*;" % (mod_name, mod_name),
            },
            is_executable = False,
        )
    else:
        ctx.actions.expand_template(
            template = ctx.file.template_file,
            output = ctx.outputs.versionscript,
            substitutions = {
                "*tensorflow*": "*tensorflow*\ninit_%s\n_init_%s\nPyInit_*\n_PyInit_*\n" % (mod_name, mod_name),
            },
            is_executable = False,
        )

_append_init_to_versionscript = rule(
    attrs = {
        "module_name": attr.string(mandatory = True),
        "template_file": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "is_version_script": attr.bool(
            default = True,
            doc = "whether target is a ld version script or exported symbol list",
            mandatory = False,
        ),
    },
    outputs = {"versionscript": "%{name}.lds"},
    implementation = _append_init_to_versionscript_impl,
)

# This macro should only be used for pywrap_tensorflow_internal.so.
# It was copied and refined from the original tf_py_wrap_cc_opensource rule.
# buildozer: disable=function-docstring-args
def pywrap_tensorflow_macro_opensource(
        name,
        srcs = [],
        roots = [],
        deps = [],
        dynamic_deps = [],
        static_deps = [],
        exports_filter = [],
        copts = [],
        version_script = None,
        win_def_file = None):
    """Builds the pywrap_tensorflow_internal shared object."""
    module_name = name.split("/")[-1]

    # Convert a rule name such as foo/bar/baz to foo/bar/_baz.so
    # and use that as the name for the rule producing the .so file.
    cc_library_base = "/".join(name.split("/")[:-1] + ["_" + module_name])

    # TODO(b/137885063): tf_cc_shared_object needs to be cleaned up; we really
    # shouldn't be passing a name qualified with .so here.
    cc_shared_library_name = cc_library_base + ".so"
    cc_library_pyd_name = "/".join(
        name.split("/")[:-1] + ["_" + module_name + ".pyd"],
    )

    # We need pybind11 to export the shared object PyInit symbol only in OSS.
    extra_deps = [clean_dep("@pybind11")]

    if not version_script:
        version_script = select({
            "//tensorflow:macos": clean_dep("//tensorflow:tf_exported_symbols.lds"),
            "//conditions:default": clean_dep("//tensorflow:tf_version_script.lds"),
        })
    vscriptname = name + "_versionscript"
    _append_init_to_versionscript(
        name = vscriptname,
        is_version_script = select({
            "//tensorflow:macos": False,
            "//conditions:default": True,
        }),
        module_name = module_name,
        template_file = version_script,
    )
    extra_linkopts = select({
        clean_dep("//tensorflow:macos"): [
            # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
            # not being exported.  There should be a better way to deal with this.
            "-Wl,-w",
            "-Wl,-exported_symbols_list,$(location %s.lds)" % vscriptname,
        ],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,--version-script",
            "$(location %s.lds)" % vscriptname,
        ],
    })
    additional_linker_inputs = if_windows([], otherwise = ["%s.lds" % vscriptname])

    # This is needed so that libtensorflow_cc is included in the pip package.
    srcs += select({
        clean_dep("//tensorflow:macos"): [clean_dep("//tensorflow:libtensorflow_cc.%s.dylib" % VERSION_MAJOR)],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [clean_dep("//tensorflow:libtensorflow_cc.so.%s" % VERSION_MAJOR)],
    })

    tf_cc_shared_library_opensource(
        name = cc_shared_library_name,
        srcs = srcs,
        # framework_so is no longer needed as libtf.so is included via the extra_deps.
        framework_so = [],
        copts = copts + if_not_windows([
            "-Wno-self-assign",
            "-Wno-sign-compare",
            "-Wno-write-strings",
        ]),
        linkopts = extra_linkopts,
        linkstatic = 1,
        roots = roots,
        deps = deps + extra_deps,
        dynamic_deps = dynamic_deps,
        static_deps = static_deps,
        exports_filter = exports_filter,
        win_def_file = win_def_file,
        additional_linker_inputs = additional_linker_inputs,
    )

    # When a non-versioned .so is added as a 'src' to a bazel target, it uses
    # -l%(so_name) instead of -l:%(so_file) during linking.  When -l%(so_name)
    # is passed to ld, it will look for an associated file with the schema
    # lib%(so_name).so.  Since pywrap_tensorflow is not explicitly versioned
    # and is not prefixed with lib_, we add a rule for the creation of an .so
    # file with the canonical lib schema (e.g. libNAME.so), so that
    # -l%(so_name) is resolved during linking.
    #
    # See: https://github.com/bazelbuild/bazel/blob/7a6808260a733d50983c1adf0cf5a7493472267f/src/main/java/com/google/devtools/build/lib/rules/cpp/LibrariesToLinkCollector.java#L319
    for pattern in SHARED_LIBRARY_NAME_PATTERNS:
        name_os = pattern % (cc_library_base, "")
        native.genrule(
            name = name_os + "_rule",
            srcs = [":" + cc_shared_library_name],
            outs = [name_os],
            cmd = "cp $< $@",
        )

    native.genrule(
        name = "gen_" + cc_library_pyd_name,
        srcs = [":" + cc_shared_library_name],
        outs = [cc_library_pyd_name],
        cmd = "cp $< $@",
    )

    # TODO(amitpatankar): Remove this py_library reference and
    # move the dependencies to pywrap_tensorflow. This can
    # eliminate one layer of Python import redundancy. We would
    # have to change all pywrap_tensorflow imports to
    # pywrap_tensorflow_internal.

    # Bazel requires an empty .py file for pywrap_tensorflow_internal.py.
    empty_py_file = [name + ".py"]
    native.genrule(
        name = "empty_py_file_rule",
        outs = empty_py_file,
        cmd = "touch $@",
    )

    # TODO(b/271333181): This should be done more generally on Windows for every dll dependency
    # (there is only one currently) that is not in the same directory, otherwise Python will fail to
    # link the pyd (which is just a dll) because of missing dependencies.
    _create_symlink("bfloat16.so", "//tensorflow/tsl/python/lib/core:bfloat16.so")

    native.py_library(
        name = name,
        srcs = [":" + name + ".py"],
        srcs_version = "PY3",
        data = select({
            clean_dep("//tensorflow:windows"): [
                ":" + cc_library_pyd_name,
                ":bfloat16.so",
                "//tensorflow/tsl/python/lib/core:bfloat16.so",
            ],
            "//conditions:default": [
                ":" + cc_shared_library_name,
            ],
        }),
    )

# Export open source version of pywrap_tensorflow_macro under base name as well.
pywrap_tensorflow_macro = pywrap_tensorflow_macro_opensource

# This macro is for running python tests against system installed pip package
# on Windows.
#
# py_test is built as an executable python zip file on Windows, which contains all
# dependencies of the target. Because of the C++ extensions, it would be very
# inefficient if the py_test zips all runfiles, plus we don't need them when running
# tests against system installed pip package. So we'd like to get rid of the deps
# of py_test in this case.
#
# In order to trigger the tests without bazel clean after getting rid of deps,
# we introduce the following :
# 1. When --define=no_tensorflow_py_deps=true, the py_test depends on a marker
#    file of the pip package, the test gets to rerun when the pip package change.
#    Note that this only works on Windows. See the definition of
#    //third_party/tensorflow/tools/pip_package:win_pip_package_marker for specific reasons.
# 2. When --define=no_tensorflow_py_deps=false (by default), it's a normal py_test.
def py_test(deps = [], data = [], kernels = [], exec_properties = None, test_rule = native.py_test, **kwargs):
    if not exec_properties:
        exec_properties = tf_exec_properties(kwargs)

    test_rule(
        # TODO(jlebar): Ideally we'd use tcmalloc here.,
        deps = select({
            "//conditions:default": deps,
            clean_dep("//tensorflow:no_tensorflow_py_deps"): [],
        }),
        data = data + select({
            "//conditions:default": kernels,
            clean_dep("//tensorflow:no_tensorflow_py_deps"): ["//tensorflow/tools/pip_package:win_pip_package_marker"],
        }),
        exec_properties = exec_properties,
        **kwargs
    )

register_extension_info(
    extension = py_test,
    label_regex_for_dep = "{extension_name}",
)

# Similar to py_test above, this macro is used to exclude dependencies for some py_binary
# targets in order to reduce the size of //tensorflow/tools/pip_package:simple_console_windows.
# See https://github.com/tensorflow/tensorflow/issues/22390
def py_binary(name, deps = [], **kwargs):
    # Add an extra target for dependencies to avoid nested select statement.
    native.py_library(
        name = name + "_deps",
        deps = deps,
    )

    # Python version placeholder
    native.py_binary(
        name = name,
        deps = select({
            "//conditions:default": [":" + name + "_deps"],
            clean_dep("//tensorflow:no_tensorflow_py_deps"): [],
        }),
        **kwargs
    )

def pytype_library(name, pytype_deps = [], pytype_srcs = [], **kwargs):
    # Types not enforced in OSS.
    native.py_library(name = name, **kwargs)

def tf_py_test(
        name,
        srcs,
        size = "medium",
        data = [],
        main = None,
        args = [],
        tags = [],
        shard_count = 1,
        additional_visibility = [],
        kernels = [],
        flaky = 0,
        xla_enable_strict_auto_jit = False,
        xla_enabled = False,
        grpc_enabled = False,
        tfrt_enabled = False,
        # `tfrt_enabled` is set for some test targets, and if we enable
        # TFRT tests just by that, this will enable TFRT builds for open source.
        # TFRT open source is not fully integrated yet so we need a temporary
        # workaround to enable TFRT only for internal builds. `tfrt_enabled_internal`
        # will be set by `tensorflow.google.bzl`'s `tf_py_test` target, which is
        # only applied for internal builds.
        # TODO(b/156911178): Revert this temporary workaround once TFRT open source
        # is fully integrated with TF.
        tfrt_enabled_internal = False,
        **kwargs):
    """Create one or more python tests with extra tensorflow dependencies."""
    xla_test_true_list = []
    if "additional_deps" in kwargs:
        fail("Use `deps` to specify dependencies. `additional_deps` has been replaced with the standard pattern of `deps`.")
    deps = kwargs.pop("deps", [])

    # xla_enable_strict_auto_jit is used to run Tensorflow unit tests with all XLA compilable
    # kernels compiled with XLA.
    if xla_enable_strict_auto_jit:
        xla_enabled = True
        xla_test_true_list.append("//tensorflow/python/framework:is_xla_test_true")
    if xla_enabled:
        deps = deps + tf_additional_xla_deps_py()
    if grpc_enabled:
        deps = deps + tf_additional_grpc_deps_py()

    # NOTE(ebrevdo): This is a workaround for depset() not being able to tell
    # the difference between 'dep' and 'clean_dep(dep)'.
    for to_add in [
        "//tensorflow/python:extra_py_tests_deps",
    ]:
        if to_add not in deps and clean_dep(to_add) not in deps:
            deps.append(clean_dep(to_add))

    env = kwargs.pop("env", {})

    # Python version placeholder
    kwargs.setdefault("srcs_version", "PY3")
    py_test(
        name = name,
        size = size,
        srcs = srcs,
        args = args,
        data = data,
        flaky = flaky,
        kernels = kernels,
        main = main,
        shard_count = shard_count,
        tags = tags,
        visibility = [clean_dep("//tensorflow:internal")] +
                     additional_visibility,
        deps = depset(deps + xla_test_true_list),
        env = env,
        **kwargs
    )
    if tfrt_enabled_internal:
        tfrt_env = {}
        tfrt_env.update(env)
        tfrt_env["EXPERIMENTAL_ENABLE_TFRT"] = "1"

        # None `main` defaults to `name` + ".py" in `py_test` target. However, since we
        # are appending _tfrt. it becomes `name` + "_tfrt.py" effectively. So force
        # set `main` argument without `_tfrt`.
        if main == None:
            main = name + ".py"

        py_test(
            env = tfrt_env,
            name = name + "_tfrt",
            size = size,
            srcs = srcs,
            args = args,
            data = data,
            flaky = flaky,
            kernels = kernels,
            main = main,
            shard_count = shard_count,
            tags = tags + ["tfrt"],
            visibility = [clean_dep("//tensorflow:internal")] +
                         additional_visibility,
            deps = depset(deps + xla_test_true_list),
            **kwargs
        )

register_extension_info(
    extension = tf_py_test,
    label_regex_for_dep = "{extension_name}(_tfrt)?",
)

def gpu_py_test(
        name,
        srcs,
        size = "medium",
        data = [],
        main = None,
        args = [],
        shard_count = 1,
        kernels = [],
        tags = [],
        flaky = 0,
        xla_enable_strict_auto_jit = False,
        xla_enabled = False,
        grpc_enabled = False,
        xla_tags = [],  # additional tags for xla_gpu tests
        **kwargs):
    if main == None:
        main = name + ".py"
    if "additional_deps" in kwargs:
        fail("Use `deps` to specify dependencies. `additional_deps` has been replaced with the standard pattern of `deps`.")
    configs = ["cpu", "gpu"]
    if "multi_gpu" in tags or "multi_and_single_gpu" in tags:
        configs = configs + ["2gpu"]

    targets = []

    for config in configs:
        test_name = name
        test_tags = tags
        if config == "gpu":
            test_tags = test_tags + tf_gpu_tests_tags()
        if config == "2gpu":
            test_tags = test_tags + two_gpu_tags
            if "requires-gpu-nvidia" in test_tags:
                test_tags.remove("requires-gpu-nvidia")

        # TODO(b/215751004): CPU on XLA tests are skipped intentionally.
        if config != "cpu" and xla_enable_strict_auto_jit:
            strict_auto_jit_test_name = test_name + "_xla_" + config
            tf_py_test(
                name = strict_auto_jit_test_name,
                size = size,
                srcs = srcs,
                args = args,
                data = data,
                flaky = flaky,
                grpc_enabled = grpc_enabled,
                kernels = kernels,
                main = main,
                shard_count = shard_count,
                tags = test_tags + xla_tags + ["xla", "manual"],
                xla_enabled = xla_enabled,
                xla_enable_strict_auto_jit = True,
                **kwargs
            )
            targets.append(strict_auto_jit_test_name)

        test_name = test_name + "_" + config

        tf_py_test(
            name = test_name,
            size = size,
            srcs = srcs,
            args = args,
            data = data,
            flaky = flaky,
            grpc_enabled = grpc_enabled,
            kernels = kernels,
            main = main,
            shard_count = shard_count,
            tags = test_tags,
            xla_enabled = xla_enabled,
            xla_enable_strict_auto_jit = False,
            **kwargs
        )
        targets.append(test_name)

    native.test_suite(name = name, tests = targets, tags = tags)

# terminology changes: saving cuda_* definition for compatibility
def cuda_py_test(*args, **kwargs):
    gpu_py_test(*args, **kwargs)

register_extension_info(
    extension = gpu_py_test,
    label_regex_for_dep = "{extension_name}_cpu",
)

def py_tests(
        name,
        srcs,
        size = "medium",
        kernels = [],
        data = [],
        tags = [],
        shard_count = 1,
        prefix = "",
        xla_enable_strict_auto_jit = False,
        xla_enabled = False,
        grpc_enabled = False,
        tfrt_enabled = False,
        **kwargs):
    if "additional_deps" in kwargs:
        fail("Use `deps` to specify dependencies. `additional_deps` has been replaced with the standard pattern of `deps`.")
    for src in srcs:
        test_name = src.split("/")[-1].split(".")[0]
        if prefix:
            test_name = "%s_%s" % (prefix, test_name)
        tf_py_test(
            name = test_name,
            size = size,
            srcs = [src],
            data = data,
            grpc_enabled = grpc_enabled,
            kernels = kernels,
            main = src,
            shard_count = shard_count,
            tags = tags,
            xla_enabled = xla_enabled,
            xla_enable_strict_auto_jit = xla_enable_strict_auto_jit,
            tfrt_enabled = tfrt_enabled,
            **kwargs
        )

# Creates a genrule named <name> for running tools/proto_text's generator to
# make the proto_text functions, for the protos passed in <srcs>.
#
# Return a struct with fields (hdrs, srcs) containing the names of the
# generated files.
def tf_generate_proto_text_sources(name, srcs_relative_dir, srcs, protodeps = [], deps = [], visibility = None, compatible_with = None):
    out_hdrs = (
        [
            p.replace(".proto", ".pb_text.h")
            for p in srcs
        ] + [p.replace(".proto", ".pb_text-impl.h") for p in srcs]
    )
    out_srcs = [p.replace(".proto", ".pb_text.cc") for p in srcs]
    native.genrule(
        name = name + "_srcs",
        srcs = srcs + protodeps + [clean_dep("//tensorflow/tools/proto_text:placeholder.txt")],
        outs = out_hdrs + out_srcs,
        visibility = visibility,
        cmd =
            "$(location //tensorflow/tools/proto_text:gen_proto_text_functions) " +
            "$(@D) " + srcs_relative_dir + " $(SRCS)",
        tools = [
            clean_dep("//tensorflow/tools/proto_text:gen_proto_text_functions"),
        ],
        compatible_with = compatible_with,
    )

    native.filegroup(
        name = name + "_hdrs",
        srcs = out_hdrs,
        visibility = visibility,
        compatible_with = compatible_with,
    )

    cc_library(
        compatible_with = compatible_with,
        name = name,
        srcs = out_srcs,
        hdrs = out_hdrs,
        visibility = visibility,
        deps = deps,
        alwayslink = 1,
    )

def tf_genrule_cmd_append_to_srcs(to_append):
    return ("cat $(SRCS) > $(@) && " + "echo >> $(@) && " + "echo " + to_append +
            " >> $(@)")

def _local_exec_transition_impl(settings, attr):
    return {
        # Force all targets in the subgraph to build on the local machine.
        "//command_line_option:modify_execution_info": ".*=+no-remote-exec",
    }

# A transition that forces all targets in the subgraph to be built locally.
_local_exec_transition = transition(
    implementation = _local_exec_transition_impl,
    inputs = [],
    outputs = [
        "//command_line_option:modify_execution_info",
    ],
)

def _local_genrule_impl(ctx):
    ctx.actions.run_shell(
        outputs = [ctx.outputs.out],
        inputs = [f for t in ctx.attr.srcs for f in t.files.to_list()],
        tools = [ctx.executable.exec_tool],
        arguments = [f.path for t in ctx.attr.srcs for f in t.files.to_list()] +
                    [ctx.outputs.out.path],
        command = "%s %s" % (ctx.executable.exec_tool.path, ctx.attr.arguments),
        execution_requirements = {"no-remote-exec": ""},
        use_default_shell_env = True,
    )

# A genrule that executes locally and forces the tool it runs to be built locally.
# For python, we want to build all py_binary rules locally that we also want
# to execute locally, as the remote image might use a different python version.
# TODO(klimek): Currently we still need to annotate the py_binary rules to use
# the local platform when building. When we know how to change the platform
# (https://github.com/bazelbuild/bazel/issues/11035) use this to not require
# annotating the py_binary rules.
_local_genrule_internal = rule(
    implementation = _local_genrule_impl,
    attrs = {
        "out": attr.output(),
        "exec_tool": attr.label(
            executable = True,
            cfg = _local_exec_transition,
            allow_files = True,
        ),
        "arguments": attr.string(),
        "srcs": attr.label_list(
            allow_files = True,
        ),
        "_whitelist_function_transition": attr.label(default = "@bazel_tools//tools/whitelists/function_transition_whitelist"),
    },
)

# Wrap the rule in a macro so we can pass in exec_compatible_with.
def _local_genrule(**kwargs):
    _local_genrule_internal(
        exec_compatible_with = [
            "@local_execution_config_platform//:platform_constraint",
        ],
        **kwargs
    )

def tf_version_info_genrule(name, out, compatible_with = None):
    # TODO(gunan): Investigate making this action hermetic so we do not need
    # to run it locally.
    _local_genrule(
        name = name,
        out = out,
        compatible_with = compatible_with,
        exec_tool = "//tensorflow/tools/git:gen_git_source",
        srcs = [
            "@local_config_git//:gen/spec.json",
            "@local_config_git//:gen/head",
            "@local_config_git//:gen/branch_ref",
        ],
        arguments = "--generate \"$@\" --git_tag_override=${GIT_TAG_OVERRIDE:-}",
    )

def _dict_to_kv(d):
    """Convert a dictionary to a space-joined list of key=value pairs."""
    return " " + " ".join(["%s=%s" % (k, v) for k, v in d.items()])

def tf_py_build_info_genrule(name, out):
    _local_genrule(
        name = name,
        out = out,
        exec_tool = "//tensorflow/tools/build_info:gen_build_info",
        arguments =
            "--raw_generate \"$@\" " +
            " --key_value" +
            " is_rocm_build=" + if_rocm("True", "False") +
            " is_cuda_build=" + if_cuda("True", "False") +
            " is_tensorrt_build=" + if_tensorrt("True", "False") +
            if_windows(_dict_to_kv({
                "msvcp_dll_names": "msvcp140.dll,msvcp140_1.dll",
            }), "") + if_windows_cuda(_dict_to_kv({
                "nvcuda_dll_name": "nvcuda.dll",
                "cudart_dll_name": "cudart{cuda_version}.dll",
                "cudnn_dll_name": "cudnn{cudnn_version}.dll",
            }), ""),
    )

def cc_library_with_android_deps(
        deps,
        android_deps = [],
        common_deps = [],
        copts = tf_copts(),
        **kwargs):
    deps = if_not_android(deps) + if_android(android_deps) + common_deps
    cc_library(deps = deps, copts = copts, **kwargs)

def tensorflow_opensource_extra_deps():
    return []

# Builds a pybind11 compatible library.
def pybind_library(
        name,
        copts = [],
        features = [],
        tags = [],
        deps = [],
        **kwargs):
    # Mark common dependencies as required for build_cleaner.
    tags = tags + ["req_dep=" + clean_dep("//third_party/pybind11"), "req_dep=@local_config_python//:python_headers"]

    native.cc_library(
        name = name,
        copts = copts + ["-fexceptions"],
        features = features + [
            "-use_header_modules",  # Required for pybind11.
            "-parse_headers",
        ],
        tags = tags,
        deps = deps + [clean_dep("//third_party/pybind11"), "@local_config_python//:python_headers"],
        **kwargs
    )

# buildozer: disable=function-docstring-args
def pybind_extension_opensource(
        name,
        srcs,
        module_name = None,
        hdrs = [],
        dynamic_deps = [],
        static_deps = [],
        deps = [],
        additional_exported_symbols = [],
        compatible_with = None,
        copts = [],
        data = [],
        defines = [],
        deprecation = None,
        features = [],
        link_in_framework = False,
        licenses = None,
        linkopts = [],
        pytype_deps = [],
        pytype_srcs = [],
        restricted_to = None,
        srcs_version = "PY3",
        testonly = None,
        visibility = None,
        win_def_file = None):
    """Builds a generic Python extension module."""
    _ignore = [module_name]
    p = name.rfind("/")
    if p == -1:
        sname = name
        prefix = ""
    else:
        sname = name[p + 1:]
        prefix = name[:p + 1]
    so_file = "%s%s.so" % (prefix, sname)
    filegroup_name = "%s_filegroup" % name
    pyd_file = "%s%s.pyd" % (prefix, sname)
    exported_symbols = [
        "init%s" % sname,
        "init_%s" % sname,
        "PyInit_%s" % sname,
    ] + additional_exported_symbols

    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name

    exported_symbols_output = "\n".join(["_%s" % symbol for symbol in exported_symbols])
    version_script_output = "\n".join([" %s;" % symbol for symbol in exported_symbols])

    native.genrule(
        name = name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '%s' >$@" % exported_symbols_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    native.genrule(
        name = name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n%s\n local: *;};' >$@" % version_script_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    if static_deps:
        cc_library_name = so_file + "_cclib"
        cc_library(
            name = cc_library_name,
            hdrs = hdrs,
            srcs = srcs + hdrs,
            data = data,
            deps = deps,
            compatible_with = compatible_with,
            copts = copts + [
                "-fno-strict-aliasing",
                "-fexceptions",
            ] + select({
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-fvisibility=hidden",
                ],
            }),
            defines = defines,
            features = features + ["-use_header_modules"],
            restricted_to = restricted_to,
            testonly = testonly,
            visibility = visibility,
        )

        cc_shared_library(
            name = so_file,
            roots = [cc_library_name],
            dynamic_deps = dynamic_deps,
            static_deps = static_deps,
            additional_linker_inputs = [exported_symbols_file, version_script_file],
            compatible_with = compatible_with,
            deprecation = deprecation,
            features = features + ["-use_header_modules"],
            licenses = licenses,
            restricted_to = restricted_to,
            shared_lib_name = so_file,
            testonly = testonly,
            user_link_flags = linkopts + _rpath_user_link_flags(name) + select({
                clean_dep("//tensorflow:macos"): [
                    # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                    # not being exported.  There should be a better way to deal with this.
                    "-Wl,-w",
                    "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
                ],
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-Wl,--version-script",
                    "$(location %s)" % version_script_file,
                ],
            }),
            visibility = visibility,
        )

        # cc_shared_library can generate more than one file.
        # Solution to avoid the error "variable '$<' : more than one input file."
        filegroup(
            name = filegroup_name,
            srcs = [so_file],
            output_group = "main_shared_library_output",
            testonly = testonly,
        )
    else:
        if link_in_framework:
            srcs += tf_binary_additional_srcs()

        cc_binary(
            name = so_file,
            srcs = srcs + hdrs,
            data = data,
            copts = copts + [
                "-fno-strict-aliasing",
                "-fexceptions",
            ] + select({
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-fvisibility=hidden",
                ],
            }),
            linkopts = linkopts + _rpath_linkopts(name) + select({
                clean_dep("//tensorflow:macos"): [
                    # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                    # not being exported.  There should be a better way to deal with this.
                    "-Wl,-w",
                    "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
                ],
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-Wl,--version-script",
                    "$(location %s)" % version_script_file,
                ],
            }),
            deps = deps + [
                exported_symbols_file,
                version_script_file,
            ],
            defines = defines,
            features = features + ["-use_header_modules"],
            linkshared = 1,
            testonly = testonly,
            licenses = licenses,
            visibility = visibility,
            deprecation = deprecation,
            restricted_to = restricted_to,
            compatible_with = compatible_with,
        )

        # For Windows, emulate the above filegroup with the shared object.
        native.alias(
            name = filegroup_name,
            actual = so_file,
        )

    # For Windows only.
    native.genrule(
        name = name + "_pyd_copy",
        srcs = [filegroup_name],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
        testonly = testonly,
    )

    native.py_library(
        name = name,
        data = select({
            clean_dep("//tensorflow:windows"): [pyd_file],
            "//conditions:default": [so_file],
        }) + pytype_srcs,
        deps = pytype_deps,
        srcs_version = srcs_version,
        licenses = licenses,
        testonly = testonly,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )

# Export open source version of pybind_extension under base name as well.
pybind_extension = pybind_extension_opensource

# Note: we cannot add //third_party/tf_runtime:__subpackages__ here,
# because that builds all of tf_runtime's packages, and some of them
# are known not to build on big endian systems.
# See b/148087476 and
# https://github.com/tensorflow/tensorflow/issues/57844.
# TODO(b/254083070): remove this definition once the packages move to TSL.
def tsl_async_value_deps():
    return [
        "@tf_runtime//:async_value",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
        "@tf_runtime//:concurrent_vector",
        "@tf_runtime//:ref_count",
        "@tf_runtime//third_party/llvm_derived:unique_any",
        "@tf_runtime//third_party/llvm_derived:in_place",
    ]

def tf_python_pybind_static_deps(testonly = False):
    # TODO(b/146808376): Reduce the dependencies to those that are really needed.
    static_deps = [
        "//:__subpackages__",
        "@FP16//:__subpackages__",
        "@FXdiv//:__subpackages__",
        "@XNNPACK//:__subpackages__",
        "@arm_neon_2_x86_sse//:__subpackages__",
        "@bazel_tools//:__subpackages__",
        "@boringssl//:__subpackages__",
        "@clog//:__subpackages__",
        "@com_github_cares_cares//:__subpackages__",
        "@com_github_googlecloudplatform_tensorflow_gcp_tools//:__subpackages__",
        "@com_github_grpc_grpc//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_googleapis//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@com_googlesource_code_re2//:__subpackages__",
        "@compute_library//:__subpackages__",
        "@cpuinfo//:__subpackages__",
        "@cudnn_frontend_archive//:__subpackages__",  #  TFRT integration for TensorFlow.
        "@curl//:__subpackages__",
        "@dlpack//:__subpackages__",
        "@double_conversion//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@farmhash_archive//:__subpackages__",
        "@farmhash_gpu_archive//:__subpackages__",
        "@fft2d//:__subpackages__",
        "@flatbuffers//:__subpackages__",
        "@gemmlowp//:__subpackages__",
        "@gif//:__subpackages__",
        "@highwayhash//:__subpackages__",
        "@hwloc//:__subpackages__",
        "@icu//:__subpackages__",
        "@jsoncpp_git//:__subpackages__",
        "@libjpeg_turbo//:__subpackages__",
        "@llvm-project//:__subpackages__",
        "@llvm_openmp//:__subpackages__",
        "@llvm_terminfo//:__subpackages__",
        "@llvm_zlib//:__subpackages__",
        "@local_config_cuda//:__subpackages__",
        "@local_config_git//:__subpackages__",
        "@local_config_nccl//:__subpackages__",
        "@local_config_python//:__subpackages__",
        "@local_config_rocm//:__subpackages__",
        "@local_config_tensorrt//:__subpackages__",
        "@local_execution_config_platform//:__subpackages__",
        "@mkl_dnn_acl_compatible//:__subpackages__",
        "@mkl_dnn_v1//:__subpackages__",
        "@nsync//:__subpackages__",
        "@nccl_archive//:__subpackages__",
        "@org_sqlite//:__subpackages__",
        "@platforms//:__subpackages__",
        "@png//:__subpackages__",
        "@pthreadpool//:__subpackages__",
        "@pybind11//:__subpackages__",
        "@ruy//:__subpackages__",
        "@snappy//:__subpackages__",
        "@sobol_data//:__subpackages__",
        "@stablehlo//:__subpackages__",
        "@tf_runtime//:__subpackages__",
        "@upb//:__subpackages__",
        "@zlib//:__subpackages__",
    ]
    static_deps += tsl_async_value_deps()
    static_deps += [] if not testonly else [
        "@com_google_benchmark//:__subpackages__",
        "@com_google_googletest//:__subpackages__",
    ]
    static_deps += if_onednn_v3(["@onednn_v3//:__subpackages__"])
    return if_oss(static_deps)

# buildozer: enable=function-docstring-args
def tf_python_pybind_extension_opensource(
        name,
        srcs,
        module_name = None,
        hdrs = [],  # TODO(b/264128506): Drop after migration to cc_shared_library.
        deps = [],
        dynamic_deps = [],
        static_deps = [],
        compatible_with = None,
        copts = [],
        defines = [],
        features = [],
        testonly = False,
        visibility = None,
        win_def_file = None):
    """A wrapper macro for pybind_extension_opensource that is used in tensorflow/python/BUILD.

    Please do not use it anywhere else as it may behave unexpectedly. b/146445820

    It is used for targets under //third_party/tensorflow/python that link
    against libtensorflow_framework.so and pywrap_tensorflow_internal.so.
    """
    extended_deps = deps + if_mkl_ml(["//third_party/mkl:intel_binary_blob"])
    extended_deps += [] if dynamic_deps else if_windows([], ["//tensorflow:libtensorflow_framework_import_lib"]) + tf_binary_pybind_deps()
    pybind_extension_opensource(
        name,
        srcs,
        module_name = module_name,
        hdrs = hdrs,
        dynamic_deps = dynamic_deps,
        static_deps = static_deps,
        deps = extended_deps,
        compatible_with = compatible_with,
        copts = copts,
        defines = defines,
        features = features,
        testonly = testonly,
        visibility = visibility,
        win_def_file = win_def_file,
    )

# Export open source version of tf_python_pybind_extension under base name as well.
tf_python_pybind_extension = tf_python_pybind_extension_opensource

def tf_pybind_cc_library_wrapper_opensource(name, deps, visibility = None, **kwargs):
    """Wrapper for cc_library and proto dependencies used by tf_python_pybind_extension_opensource.

    This wrapper ensures that cc libraries' and protos' headers are made
    available to pybind code, without creating ODR violations in the dynamically
    linked case.  The symbols in these deps symbols should be linked to, and
    exported by, the core pywrap_tensorflow_internal.so
    """
    cc_header_only_library(name = name, deps = deps, visibility = visibility, **kwargs)

# Export open source version of tf_pybind_cc_library_wrapper under base name as well.
tf_pybind_cc_library_wrapper = tf_pybind_cc_library_wrapper_opensource

if_cuda_or_rocm = _if_cuda_or_rocm

def tf_monitoring_framework_deps(link_to_tensorflow_framework = True):
    """Get the monitoring libs that will be linked to the tensorflow framework.

      Currently in OSS, the protos must be statically linked to the tensorflow
      framework, whereas the grpc should not be linked here.
    """
    return select({
        "//tensorflow:stackdriver_support": [
            "@com_github_googlecloudplatform_tensorflow_gcp_tools//monitoring:stackdriver_exporter_protos",
        ],
        "//conditions:default": [],
    })

def tf_monitoring_python_deps():
    """Get the monitoring libs that will be linked to the python wrapper.

      Currently in OSS, the grpc must be statically linked to the python wrapper
      whereas the protos should not be linked here.
    """
    return select({
        "//tensorflow:stackdriver_support": [
            "@com_github_googlecloudplatform_tensorflow_gcp_tools//monitoring:stackdriver_exporter",
        ],
        "//conditions:default": [],
    })

# Teams sharing the same repo can provide their own ops_to_register.h file using
# this function, and pass in -Ipath/to/repo flag when building the target.
def tf_selective_registration_deps():
    return []

def tf_jit_compilation_passes_extra_deps():
    return []

def if_mlir(if_true, if_false = []):
    return select({
        str(Label("//tensorflow:with_mlir_support")): if_true,
        "//conditions:default": if_false,
    })

def tf_enable_mlir_bridge():
    return select({
        str(Label("//tensorflow:enable_mlir_bridge")): [
            "//tensorflow/python:is_mlir_bridge_test_true",
        ],
        str(Label("//tensorflow:disable_mlir_bridge")): [
            "//tensorflow/python:is_mlir_bridge_test_false",
        ],
        "//conditions:default": [],
    })

def tfcompile_target_cpu(name = ""):
    return ""

def tfcompile_dfsan_enabled():
    return False

def tfcompile_dfsan_abilists():
    return []

def tf_external_workspace_visible(visibility):
    # External workspaces can see this target.
    return ["//visibility:public"]

def _filegroup_as_file_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(
        output = out,
        content = "\n".join([f.short_path for f in ctx.files.dep]),
    )
    return DefaultInfo(files = depset([out]))

_filegroup_as_file = rule(
    implementation = _filegroup_as_file_impl,
    attrs = {
        "dep": attr.label(),
    },
)

def filegroup_as_file(name, dep, visibility = []):
    """Creates a filegroup ${name}_file which contains the file ${name}."""
    _filegroup_as_file(name = name, dep = dep)
    native.filegroup(
        name = name + "_file",
        srcs = [name],
        visibility = visibility,
    )

def tf_grpc_dependencies():
    return ["//tensorflow:grpc"]

def tf_grpc_cc_dependencies():
    return ["//tensorflow:grpc++"]

def get_compatible_with_portable():
    return []

def get_compatible_with_cloud():
    return []

def filegroup(**kwargs):
    native.filegroup(**kwargs)

def genrule(**kwargs):
    native.genrule(**kwargs)

def internal_tfrt_deps():
    return []

def _tf_gen_options_header_impl(ctx):
    header_depset = depset([ctx.outputs.output_header])

    define_vals = {True: "true", False: "false"}
    substitutions = {}
    for target, identifier in ctx.attr.build_settings.items():
        setting_val = target[BuildSettingInfo].value
        lines = [
            "// %s" % target.label,
            "#define TF_OPTION_%s() %s" % (identifier, define_vals[setting_val]),
        ]
        substitutions["#define_option %s" % identifier] = "\n".join(lines)

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.output_header,
        substitutions = substitutions,
    )

    return [
        DefaultInfo(files = header_depset),
    ]

tf_gen_options_header = rule(
    attrs = {
        "output_header": attr.output(
            doc = "File path for the generated header (output)",
            mandatory = True,
        ),
        "template": attr.label(
            doc = """Template for the header.
            For each option name 'X' (see build_settings attribute),
            '#define_option X' results in a macro 'TF_OPTION_X()'
            """,
            allow_single_file = True,
            mandatory = True,
        ),
        "build_settings": attr.label_keyed_string_dict(
            doc = """Dictionary from build-setting labels to option names. Example:
               {"//tensorflow:x_setting" : "X"}
            """,
            providers = [BuildSettingInfo],
        ),
    },
    implementation = _tf_gen_options_header_impl,
    doc = """
    Generates a header file for Bazel build settings.

    This is an alternative to setting preprocessor defines on the compiler
    command line. It has a few advantages:
      - Usage of the options requires #include-ing the header, and thus a
        Bazel-level dependency.
      - Each option has a definition site in source code, which mentions the
        corresponding Bazel setting. This is particularly useful when
        navigating code with the assistance of static analysis (e.g.
        https://cs.opensource.google/tensorflow).
      - Each option is represented as a FUNCTION()-style macro, which is always
        defined (i.e. one uses #if instead of #ifdef). This allows forms like
        'if constexpr (TF_OPTION_FOO()) { ... }', and helps catch missing
        dependencies (if 'F' is undefined, '#if F()' results in an error).
    """,
)

# These flags are used selectively to disable benign ptxas warnings for some
# build targets.  On clang "-Xcuda-ptxas --disable-warnings" is sufficient, but
# that does not work on some versions of GCC.  So for now this is empty in the
# open source build.
def tf_disable_ptxas_warning_flags():
    return []

# Use this to replace the `non_portable_tf_deps` (i.e., tensorflow/core/...) with
# tensorflow/core:portable_tensorflow_lib_lite when building portably.
def replace_with_portable_tf_lib_when_required(non_portable_tf_deps, use_lib_with_runtime = False):
    portable_tf_lib = "//tensorflow/core:portable_tensorflow_lib_lite"

    return select({
        "//tensorflow:android": [portable_tf_lib],
        "//tensorflow:ios": [portable_tf_lib],
        "//conditions:default": non_portable_tf_deps,
    })
