# Return the options to use for a C++ library or binary build.
# Uses the ":optmode" config_setting to pick the options.
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_dynamic_kernels",
    "if_static",
    "register_extension_info",
    "tf_additional_grpc_deps_py",
    "tf_additional_xla_deps_py",
    "tf_exec_compatible_with",
    "tf_gpu_tests_tags",
    "tf_sycl_tests_tags",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_binary",
    "cc_library",
    "cc_test",
)
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_library",
    "if_cuda",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
    "if_rocm_is_configured",
    "rocm_copts",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_enable_mkl",
    "if_mkl",
    "if_mkl_lnx_x64",
    "if_mkl_ml",
    "mkl_deps",
)
load(
    "//third_party/mkl_dnn:build_defs.bzl",
    "if_mkl_open_source_only",
    "if_mkl_v1_open_source_only",
)
load(
    "//third_party/ngraph:build_defs.bzl",
    "if_ngraph",
)

# version for the shared libraries, can
# not contain rc or alpha, only numbers.
# Also update tensorflow/core/public/version.h
# and tensorflow/tools/pip_package/setup.py
VERSION = "2.1.0"
VERSION_MAJOR = VERSION.split(".")[0]

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

def if_cuda_is_configured_compat(x):
    return if_cuda_is_configured(x)

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

# List of proto files for android builds
def tf_android_core_proto_sources(core_proto_sources_relative):
    return [
        _add_tfcore_prefix(p)
        for p in core_proto_sources_relative
    ]

# Returns the list of pb.h and proto.h headers that are generated for
# tf_android_core_proto_sources().
def tf_android_core_proto_headers(core_proto_sources_relative):
    return ([
        _add_tfcore_prefix(p).replace(":", "/").replace(".proto", ".pb.h")
        for p in core_proto_sources_relative
    ] + [
        _add_tfcore_prefix(p).replace(":", "/").replace(".proto", ".proto.h")
        for p in core_proto_sources_relative
    ])

# Wrapper for portable protos which currently just creates an empty rule.
def tf_portable_proto_library(name, proto_deps, deps = [], **kwargs):
    _ignore = [kwargs]
    cc_library(name = name, deps = deps + [dep + "_cc" for dep in proto_deps])

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

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

def if_ios(a):
    return select({
        clean_dep("//tensorflow:ios"): a,
        "//conditions:default": [],
    })

def if_ios_x86_64(a):
    return select({
        clean_dep("//tensorflow:ios_x86_64"): a,
        "//conditions:default": [],
    })

def if_mobile(a):
    return select({
        clean_dep("//tensorflow:android"): a,
        clean_dep("//tensorflow:ios"): a,
        "//conditions:default": [],
    })

def if_not_mobile(a):
    return select({
        clean_dep("//tensorflow:android"): [],
        clean_dep("//tensorflow:ios"): [],
        "//conditions:default": a,
    })

# Config setting selector used when building for products
# which requires restricted licenses to be avoided.
def if_not_lgpl_restricted(a):
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
        clean_dep("//tensorflow:with_cuda_support_windows_override"): a,
        "//conditions:default": otherwise,
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

def if_nccl(if_true, if_false = []):
    return select({
        "//tensorflow:no_nccl_support": if_false,
        "//tensorflow:windows": if_false,
        "//conditions:default": if_true,
    })

def get_win_copts(is_external = False):
    WINDOWS_COPTS = [
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
    ]
    if is_external:
        return WINDOWS_COPTS + ["/UTF_COMPILE_LIBRARY"]
    else:
        return WINDOWS_COPTS + ["/DTF_COMPILE_LIBRARY"]

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
        if_tensorrt(["-DGOOGLE_TENSORRT=1"]) +
        if_mkl(["-DINTEL_MKL=1", "-DEIGEN_USE_VML"]) +
        if_mkl_open_source_only(["-DINTEL_MKL_DNN_ONLY"]) +
        if_mkl_v1_open_source_only(["-DENABLE_MKLDNN_V1"]) +
        if_enable_mkl(["-DENABLE_MKL"]) +
        if_ngraph(["-DINTEL_NGRAPH=1"]) +
        if_android_arm(["-mfpu=neon"]) +
        if_linux_x86_64(["-msse3"]) +
        if_ios_x86_64(["-msse4.1"]) +
        select({
            clean_dep("//tensorflow:framework_shared_object"): [],
            "//conditions:default": ["-DTENSORFLOW_MONOLITHIC_BUILD"],
        }) +
        select({
            clean_dep("//tensorflow:android"): android_copts,
            clean_dep("//tensorflow:macos"): [],
            clean_dep("//tensorflow:windows"): get_win_copts(is_external),
            clean_dep("//tensorflow:ios"): [],
            clean_dep("//tensorflow:no_lgpl_deps"): ["-D__TENSORFLOW_NO_LGPL_DEPS__", "-pthread"],
            "//conditions:default": ["-pthread"],
        })
    )

def tf_openmp_copts():
    return if_mkl_lnx_x64(["-fopenmp"])

def tfe_xla_copts():
    return select({
        "//tensorflow:with_xla_support": ["-DTENSORFLOW_EAGER_USE_XLA"],
        "//conditions:default": [],
    })

def tf_opts_nortti_if_android():
    return if_android([
        "-fno-rtti",
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
    ])

def tf_opts_nortti_if_emscripten():
    return if_emscripten([
        "-fno-rtti",
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
    ])

def tf_features_nomodules_if_android():
    return if_android(["-use_header_modules"])

def tf_features_nomodules_if_emscripten():
    return if_emscripten(["-use_header_modules"])

# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate a library for that file.
def tf_gen_op_libs(op_lib_names, deps = None, is_external = True):
    # Make library out of each op so it can also be used to generate wrappers
    # for various languages.
    if not deps:
        deps = []
    for n in op_lib_names:
        cc_library(
            name = n + "_op_lib",
            copts = tf_copts(is_external = is_external),
            srcs = ["ops/" + n + ".cc"],
            deps = deps + [clean_dep("//tensorflow/core:framework")],
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
        ],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
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
SHARED_LIBRARY_NAME_PATTERNS = [
    "lib%s.so%s",  # On Linux, shared libraries are usually named as libfoo.so
    "lib%s%s.dylib",  # On macos, shared libraries are usually named as libfoo.dylib
    "%s%s.dll",  # On Windows, shared libraries are usually named as foo.dll
]

def tf_cc_shared_object(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = [],
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
            name = name_os_full,
            srcs = srcs + framework_so,
            deps = deps,
            linkshared = 1,
            data = data + data_extra,
            linkopts = linkopts + _rpath_linkopts(name_os_full) + select({
                clean_dep("//tensorflow:macos"): [
                    "-Wl,-install_name,@rpath/" + soname,
                ],
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-Wl,-soname," + soname,
                ],
            }),
            visibility = visibility,
            **kwargs
        )

    flat_names = [item for sublist in names for item in sublist]
    if name not in flat_names:
        native.filegroup(
            name = name,
            srcs = select({
                "//tensorflow:windows": [":%s.dll" % (name)],
                "//tensorflow:macos": [":lib%s%s.dylib" % (name, longsuffix)],
                "//conditions:default": [":lib%s.so%s" % (name, longsuffix)],
            }),
            visibility = visibility,
        )

register_extension_info(
    extension_name = "tf_cc_shared_object",
    label_regex_for_dep = "{extension_name}",
)

# Links in the framework shared object
# (//third_party/tensorflow:libtensorflow_framework.so) when not building
# statically. Also adds linker options (rpaths) so that the framework shared
# object can be found.
def tf_cc_binary(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = [],
        copts = tf_copts(),
        kernels = [],
        per_os_targets = False,  # Generate targets with SHARED_LIBRARY_NAME_PATTERNS
        visibility = None,
        **kwargs):
    if kernels:
        added_data_deps = tf_binary_dynamic_kernel_dsos()
    else:
        added_data_deps = []

    if per_os_targets:
        names = [pattern % (name, "") for pattern in SHARED_LIBRARY_NAME_PATTERNS]
    else:
        names = [name]
    for name_os in names:
        cc_binary(
            name = name_os,
            copts = copts,
            srcs = srcs + tf_binary_additional_srcs(),
            deps = deps + tf_binary_dynamic_kernel_deps(kernels) + if_mkl_ml(
                [
                    clean_dep("//third_party/mkl:intel_binary_blob"),
                ],
            ) + if_static(
                extra_deps = [],
                otherwise = [
                    clean_dep("//tensorflow:libtensorflow_framework_import_lib"),
                ],
            ),
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
    extension_name = "tf_cc_binary",
    label_regex_for_dep = "{extension_name}.*",
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
        }) + linkopts + _rpath_linkopts(name),
        **kwargs
    )

register_extension_info(
    extension_name = "tf_native_cc_binary",
    label_regex_for_dep = "{extension_name}.*",
)

def tf_gen_op_wrapper_cc(
        name,
        out_ops_file,
        pkg = "",
        op_gen = clean_dep("//tensorflow/cc:cc_op_gen_main"),
        deps = None,
        include_internal_ops = 0,
        # ApiDefs will be loaded in the order specified in this list.
        api_def_srcs = []):
    # Construct an op generator binary for these ops.
    tool = out_ops_file + "_gen_cc"
    if deps == None:
        deps = [pkg + ":" + name + "_op_lib"]
    tf_cc_binary(
        name = tool,
        copts = tf_copts(),
        linkopts = if_not_windows(["-lm", "-Wl,-ldl"]),
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
        extra_gen_deps = []):
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
            clean_dep("//tensorflow/core:android_tensorflow_lib"),
        ]),
        copts = tf_copts(),
        alwayslink = 1,
        visibility = visibility,
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
            clean_dep("//tensorflow/core:android_tensorflow_lib"),
        ]),
        copts = tf_copts(),
        alwayslink = 1,
        visibility = [clean_dep("//tensorflow:internal")],
    )

# Generates a Python library target wrapping the ops registered in "deps".
#
# Args:
#   name: used as the name of the generated target and as a name component of
#     the intermediate files.
#   out: name of the python file created by this rule. If None, then
#     "ops/gen_{name}.py" is used.
#   hidden: Optional list of ops names to make private in the Python module.
#     It is invalid to specify both "hidden" and "op_whitelist".
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
#   op_whitelist: if not empty, only op names in this list will be wrapped. It
#     is invalid to specify both "hidden" and "op_whitelist".
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
        cc_linkopts = [],
        api_def_srcs = []):
    _ = require_shape_functions  # Unused.

    if (hidden or hidden_file) and op_whitelist:
        fail("Cannot pass specify both hidden and op_whitelist.")

    # Construct a cc_binary containing the specified ops.
    tool_name = "gen_" + name + "_py_wrappers_cc"
    if not deps:
        deps = [str(Label("//tensorflow/core:" + name + "_op_lib"))]
    tf_cc_binary(
        name = tool_name,
        copts = tf_copts(),
        linkopts = if_not_windows(["-lm", "-Wl,-ldl"]) + cc_linkopts,
        linkstatic = 1,  # Faster to link this one-time-use binary dynamically
        visibility = [clean_dep("//tensorflow:internal")],
        deps = ([
            clean_dep("//tensorflow/core:framework"),
            clean_dep("//tensorflow/python:python_op_gen_main"),
        ] + deps),
    )

    # Invoke the previous cc_binary to generate a python file.
    if not out:
        out = "ops/gen_" + name + ".py"

    if hidden:
        op_list_arg = ",".join(hidden)
        op_list_is_whitelist = False
    elif op_whitelist:
        op_list_arg = ",".join(op_whitelist)
        op_list_is_whitelist = True
    else:
        op_list_arg = "''"
        op_list_is_whitelist = False

    # Prepare ApiDef directories to pass to the genrule.
    if not api_def_srcs:
        api_def_args_str = ","
    else:
        api_def_args = []
        for api_def_src in api_def_srcs:
            # Add directory of the first ApiDef source to args.
            # We are assuming all ApiDefs in a single api_def_src are in the
            # same directory.
            api_def_args.append(
                "$$(dirname $$(echo $(locations " + api_def_src +
                ") | cut -d\" \" -f1))",
            )
        api_def_args_str = ",".join(api_def_args)

    if hidden_file:
        # `hidden_file` is file containing a list of op names to be hidden in the
        # generated module.
        native.genrule(
            name = name + "_pygenrule",
            outs = [out],
            srcs = api_def_srcs + [hidden_file],
            tools = [tool_name] + tf_binary_additional_srcs(),
            cmd = ("$(location " + tool_name + ") " + api_def_args_str +
                   " @$(location " + hidden_file + ") > $@"),
        )
    else:
        native.genrule(
            name = name + "_pygenrule",
            outs = [out],
            srcs = api_def_srcs,
            tools = [tool_name] + tf_binary_additional_srcs(),
            cmd = ("$(location " + tool_name + ") " + api_def_args_str + " " +
                   op_list_arg + " " +
                   ("1" if op_list_is_whitelist else "0") + " > $@"),
        )

    # Make a py_library out of the generated python file.
    if not generated_target_name:
        generated_target_name = name
    native.py_library(
        name = generated_target_name,
        srcs = [out],
        srcs_version = "PY2AND3",
        visibility = visibility,
        deps = [
            clean_dep("//tensorflow/python:framework_for_generated_wrappers_v2"),
        ],
        # Instruct build_cleaner to try to avoid using this rule; typically ops
        # creators will provide their own tf_custom_op_py_library based target
        # that wraps this one.
        tags = ["avoid_dep"],
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
        linkstatic = 0,
        extra_copts = [],
        suffix = "",
        linkopts = [],
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
        }) + linkopts + _rpath_linkopts(name),
        deps = deps + tf_binary_dynamic_kernel_deps(kernels) + if_mkl_ml(
            [
                clean_dep("//third_party/mkl:intel_binary_blob"),
            ],
        ),
        data = data +
               tf_binary_dynamic_kernel_dsos() +
               tf_binary_additional_srcs(),
        exec_compatible_with = tf_exec_compatible_with(kwargs),
        # Nested select() statements seem not to be supported when passed to
        # linkstatic, and we already have a cuda select() passed in to this
        # function.
        linkstatic = linkstatic or select({
            # cc_tests with ".so"s in srcs incorrectly link on Darwin unless
            # linkstatic=1 (https://github.com/bazelbuild/bazel/issues/3450).
            # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
            clean_dep("//tensorflow:macos"): 1,
            "//conditions:default": 0,
        }),
        **kwargs
    )

register_extension_info(
    extension_name = "tf_cc_test",
    label_regex_for_dep = "{extension_name}.*",
)

# Part of the testing workflow requires a distinguishable name for the build
# rules that involve a GPU, even if otherwise identical to the base rule.
def tf_cc_test_gpu(
        name,
        srcs,
        deps,
        linkstatic = 0,
        tags = [],
        data = [],
        size = "medium",
        suffix = "",
        args = None):
    tf_cc_test(
        name,
        srcs,
        deps,
        size = size,
        args = args,
        data = data,
        linkstatic = linkstatic,
        suffix = suffix,
        tags = tags,
    )

register_extension_info(
    extension_name = "tf_cc_test_gpu",
    label_regex_for_dep = "{extension_name}",
)

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
        linkopts = []):
    tf_cc_test(
        name = name,
        size = size,
        srcs = srcs,
        args = args,
        data = data,
        extra_copts = extra_copts,
        kernels = kernels,
        linkopts = linkopts,
        linkstatic = linkstatic,
        tags = tags + ["manual"],
        deps = deps,
    )
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
        suffix = "_gpu",
        tags = tags + tf_gpu_tests_tags(),
        deps = deps + if_cuda_is_configured([
            clean_dep("//tensorflow/core:gpu_runtime"),
        ]) + if_rocm_is_configured([
            clean_dep("//tensorflow/core:gpu_runtime"),
        ]),
    )

register_extension_info(
    extension_name = "tf_gpu_cc_test",
    label_regex_for_dep = "{extension_name}",
)

# terminology changes: saving tf_cuda_* definition for compatibility
def tf_cuda_cc_test(*args, **kwargs):
    tf_gpu_cc_test(*args, **kwargs)

register_extension_info(
    extension_name = "tf_cuda_cc_test",
    label_regex_for_dep = "{extension_name}",
)

def tf_gpu_only_cc_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        data = [],
        size = "medium",
        linkstatic = 0,
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
        linkstatic = linkstatic or select({
            # cc_tests with ".so"s in srcs incorrectly link on Darwin
            # unless linkstatic=1.
            # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
            clean_dep("//tensorflow:macos"): 1,
            "//conditions:default": 0,
        }),
        tags = tags,
        exec_compatible_with = tf_exec_compatible_with({"tags": tags}),
    )

register_extension_info(
    extension_name = "tf_gpu_only_cc_test",
    label_regex_for_dep = "{extension_name}_gpu",
)

# terminology changes: saving tf_cuda_* definition for compatibility
def tf_cuda_only_cc_test(*args, **kwargs):
    tf_gpu_only_cc_test(*args, **kwargs)

register_extension_info(
    extension_name = "tf_cuda_only_cc_test",
    label_regex_for_dep = "{extension_name}_gpu",
)

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
        linkopts = [],
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
            copts = tf_copts(allow_exceptions = True) + tf_openmp_copts(),
            linkopts = select({
                clean_dep("//tensorflow:android"): [
                    "-pie",
                ],
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-lpthread",
                    "-lm",
                ],
            }) + _rpath_linkopts(src_to_test_name(src)),
            deps = deps + tf_binary_dynamic_kernel_deps(kernels) + mkl_deps(),
            data = data + tf_binary_dynamic_kernel_dsos(),
            exec_compatible_with = tf_exec_compatible_with({"tags": tags}),
            linkstatic = linkstatic,
            tags = tags,
            size = size,
            args = args,
            features = disable_header_modules,
        )

def tf_cc_tests_gpu(
        srcs,
        deps,
        name = "",
        linkstatic = 0,
        tags = [],
        size = "medium",
        kernels = [],
        args = None):
    tf_cc_tests(srcs, deps, linkstatic, size = size, args = args, kernels = kernels, tags = tags)

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
    native.java_test(
        name = name,
        srcs = srcs,
        deps = deps + tf_binary_additional_srcs(fullversion = True) + tf_binary_dynamic_kernel_dsos() + tf_binary_dynamic_kernel_deps(kernels),
        *args,
        **kwargs
    )

register_extension_info(
    extension_name = "tf_java_test",
    label_regex_for_dep = "{extension_name}",
)

def _cuda_copts(opts = []):
    """Gets the appropriate set of copts for (maybe) CUDA compilation.

        If we're doing CUDA compilation, returns copts for our particular CUDA
        compiler.  If we're not doing CUDA compilation, returns an empty list.

        """
    return select({
        "//conditions:default": [],
        "@local_config_cuda//cuda:using_nvcc": ([
            "-nvcc_options=relaxed-constexpr",
            "-nvcc_options=ftz=true",
        ]),
        "@local_config_cuda//cuda:using_clang": ([
            "-fcuda-flush-denormals-to-zero",
        ]),
    }) + if_cuda_is_configured_compat(opts)

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
        deps = deps + if_cuda_is_configured_compat([
            clean_dep("//tensorflow/stream_executor/cuda:cudart_stub"),
            clean_dep("//tensorflow/core:gpu_lib"),
        ]) + if_rocm_is_configured([
            clean_dep("//tensorflow/core:gpu_lib"),
        ]),
        alwayslink = 1,
        **kwargs
    )

register_extension_info(
    extension_name = "tf_gpu_kernel_library",
    label_regex_for_dep = "{extension_name}",
)

def tf_gpu_library(deps = None, cuda_deps = None, copts = tf_copts(), **kwargs):
    """Generate a cc_library with a conditional set of CUDA dependencies.

      When the library is built with --config=cuda:

      - Both deps and cuda_deps are used as dependencies.
      - The cuda runtime is added as a dependency (if necessary).
      - The library additionally passes -DGOOGLE_CUDA=1 to the list of copts.
      - In addition, when the library is also built with TensorRT enabled, it
          additionally passes -DGOOGLE_TENSORRT=1 to the list of copts.

      Args:
      - cuda_deps: BUILD dependencies which will be linked if and only if:
          '--config=cuda' is passed to the bazel command line.
      - deps: dependencies which will always be linked.
      - copts: copts always passed to the cc_library.
      - kwargs: Any other argument to cc_library.
      """
    if not deps:
        deps = []
    if not cuda_deps:
        cuda_deps = []

    kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]
    cc_library(
        deps = deps + if_cuda_is_configured_compat(cuda_deps + [
            clean_dep("//tensorflow/stream_executor/cuda:cudart_stub"),
            "@local_config_cuda//cuda:cuda_headers",
        ]) + if_rocm_is_configured(cuda_deps + [
            "@local_config_rocm//rocm:rocm_headers",
        ]),
        copts = (copts + if_cuda(["-DGOOGLE_CUDA=1"]) + if_rocm(["-DTENSORFLOW_USE_ROCM=1"]) + if_mkl(["-DINTEL_MKL=1"]) + if_mkl_open_source_only(["-DINTEL_MKL_DNN_ONLY"]) + if_enable_mkl(["-DENABLE_MKL"]) + if_tensorrt(["-DGOOGLE_TENSORRT=1"])),
        **kwargs
    )

register_extension_info(
    extension_name = "tf_gpu_library",
    label_regex_for_dep = "{extension_name}",
)

# terminology changes: saving tf_cuda_* definition for compatibility
def tf_cuda_library(*args, **kwargs):
    tf_gpu_library(*args, **kwargs)

register_extension_info(
    extension_name = "tf_cuda_library",
    label_regex_for_dep = "{extension_name}",
)

def tf_kernel_library(
        name,
        prefix = None,
        srcs = None,
        gpu_srcs = None,
        hdrs = None,
        deps = None,
        alwayslink = 1,
        copts = None,
        gpu_copts = None,
        is_external = False,
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
    if not copts:
        copts = []
    if not gpu_copts:
        gpu_copts = []
    textual_hdrs = []
    copts = copts + tf_copts(is_external = is_external)

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
            deps = deps,
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
        cuda_deps = cuda_deps,
        linkstatic = 1,  # Needed since alwayslink is broken in bazel b/27630669
        alwayslink = alwayslink,
        deps = deps,
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
    extension_name = "tf_kernel_library",
    label_regex_for_dep = "({extension_name}(_gpu)?|libtfkernel_{extension_name}\\.so)",
)

def tf_mkl_kernel_library(
        name,
        prefix = None,
        srcs = None,
        hdrs = None,
        deps = None,
        alwayslink = 1,
        copts = tf_copts(allow_exceptions = True) + tf_openmp_copts()):
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
        alwayslink = alwayslink,
        copts = copts,
        features = disable_header_modules,
    )

register_extension_info(
    extension_name = "tf_mkl_kernel_library",
    label_regex_for_dep = "{extension_name}",
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

# Bazel rules for building swig files.
def _py_wrap_cc_impl(ctx):
    srcs = ctx.files.srcs
    if len(srcs) != 1:
        fail("Exactly one SWIG source file label must be specified.", "srcs")
    module_name = ctx.attr.module_name
    src = ctx.files.srcs[0]
    inputs = _get_transitive_headers([src] + ctx.files.swig_includes, ctx.attr.deps)
    inputs = depset(ctx.files._swiglib, transitive = [inputs])
    inputs = depset(ctx.files.toolchain_deps, transitive = [inputs])
    swig_include_dirs = depset(_get_repository_roots(ctx, inputs))
    swig_include_dirs = depset(sorted([f.dirname for f in ctx.files._swiglib]), transitive = [swig_include_dirs])
    args = [
        "-c++",
        "-python",
        "-module",
        module_name,
        "-o",
        ctx.outputs.cc_out.path,
        "-outdir",
        ctx.outputs.py_out.dirname,
    ]
    args += ["-l" + f.path for f in ctx.files.swig_includes]
    args += ["-I" + i for i in swig_include_dirs.to_list()]
    args += [src.path]
    outputs = [ctx.outputs.cc_out, ctx.outputs.py_out]
    ctx.actions.run(
        executable = ctx.executable._swig,
        arguments = args,
        inputs = inputs.to_list(),
        outputs = outputs,
        mnemonic = "PythonSwig",
        progress_message = "SWIGing " + src.path,
    )
    return struct(files = depset(outputs))

_py_wrap_cc = rule(
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "swig_includes": attr.label_list(
            allow_files = True,
        ),
        "deps": attr.label_list(
            allow_files = True,
            providers = [CcInfo],
        ),
        "toolchain_deps": attr.label_list(
            allow_files = True,
        ),
        "module_name": attr.string(mandatory = True),
        "py_module_name": attr.string(mandatory = True),
        "_swig": attr.label(
            default = Label("@swig//:swig"),
            executable = True,
            cfg = "host",
        ),
        "_swiglib": attr.label(
            default = Label("@swig//:templates"),
            allow_files = True,
        ),
    },
    outputs = {
        "cc_out": "%{module_name}.cc",
        "py_out": "%{py_module_name}.py",
    },
    implementation = _py_wrap_cc_impl,
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

# Bazel rule for collecting the header files that a target depends on.
def _transitive_hdrs_impl(ctx):
    outputs = _get_transitive_headers([], ctx.attr.deps)
    return struct(files = outputs)

_transitive_hdrs = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            providers = [CcInfo],
        ),
    },
    implementation = _transitive_hdrs_impl,
)

def transitive_hdrs(name, deps = [], **kwargs):
    _transitive_hdrs(name = name + "_gather", deps = deps)
    native.filegroup(name = name, srcs = [":" + name + "_gather"])

# Create a header only library that includes all the headers exported by
# the libraries in deps.
def cc_header_only_library(name, deps = [], includes = [], extra_deps = [], **kwargs):
    _transitive_hdrs(name = name + "_gather", deps = deps)
    cc_library(
        name = name,
        hdrs = [":" + name + "_gather"],
        includes = includes,
        deps = extra_deps,
        **kwargs
    )

def tf_custom_op_library_additional_deps():
    return [
        "@com_google_protobuf//:protobuf_headers",
        clean_dep("//third_party/eigen3"),
        clean_dep("//tensorflow/core:framework_headers_lib"),
    ] + if_windows([clean_dep("//tensorflow/python:pywrap_tensorflow_import_lib")])

# A list of targets that contains the implemenation of
# tf_custom_op_library_additional_deps. It's used to generate a DEF file for
# exporting symbols from _pywrap_tensorflow.dll on Windows.
def tf_custom_op_library_additional_deps_impl():
    return [
        "@com_google_protobuf//:protobuf",
        "@nsync//:nsync_cpp",
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
    alldeps = depset()
    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            alldeps = depset([dep.label], transitive = [alldeps])
            if hasattr(dep, "tf_collected_deps"):
                alldeps = depset(transitive = [alldeps, dep.tf_collected_deps])
    return struct(tf_collected_deps = alldeps)

collect_deps_aspect = aspect(
    attr_aspects = ["deps"],
    implementation = _collect_deps_aspect_impl,
)

def _dep_label(dep):
    label = dep.label
    return label.package + ":" + label.name

# This rule checks that the transitive dependencies of targets listed
# in the 'deps' attribute don't depend on the targets listed in
# the 'disallowed_deps' attribute.
def _check_deps_impl(ctx):
    disallowed_deps = ctx.attr.disallowed_deps
    for input_dep in ctx.attr.deps:
        if not hasattr(input_dep, "tf_collected_deps"):
            continue
        for dep in input_dep.tf_collected_deps.to_list():
            for disallowed_dep in disallowed_deps:
                if dep == disallowed_dep.label:
                    fail(
                        _dep_label(input_dep) + " cannot depend on " + _dep_label(
                            disallowed_dep,
                        ),
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
            mandatory = True,
            allow_files = True,
        ),
    },
)

def tf_custom_op_library(name, srcs = [], gpu_srcs = [], deps = [], linkopts = [], copts = [], **kwargs):
    """Helper to build a dynamic library (.so) from the sources containing implementations of custom ops and kernels.
      """
    cuda_deps = [
        clean_dep("//tensorflow/core:stream_executor_headers_lib"),
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ]
    rocm_deps = [
        clean_dep("//tensorflow/core:stream_executor_headers_lib"),
    ]
    deps = deps + tf_custom_op_library_additional_deps()

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
            deps = deps + if_cuda_is_configured_compat(cuda_deps) + if_rocm_is_configured(rocm_deps),
            **kwargs
        )
        cuda_deps.extend([":" + basename + "_gpu"])
        rocm_deps.extend([":" + basename + "_gpu"])

    check_deps(
        name = name + "_check_deps",
        disallowed_deps = [
            clean_dep("//tensorflow/core:framework"),
            clean_dep("//tensorflow/core:lib"),
        ],
        deps = deps + if_cuda_is_configured_compat(cuda_deps) + if_rocm_is_configured(rocm_deps),
    )
    tf_cc_shared_object(
        name = name,
        srcs = srcs,
        deps = deps + if_cuda_is_configured_compat(cuda_deps) + if_rocm_is_configured(rocm_deps),
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

register_extension_info(
    extension_name = "tf_custom_op_library",
    label_regex_for_dep = "{extension_name}",
)

def tf_custom_op_py_library(
        name,
        srcs = [],
        dso = [],
        kernels = [],
        srcs_version = "PY2AND3",
        visibility = None,
        deps = []):
    _ignore = [kernels]
    native.py_library(
        name = name,
        data = dso,
        srcs = srcs,
        srcs_version = srcs_version,
        visibility = visibility,
        deps = deps,
    )

register_extension_info(
    extension_name = "tf_custom_op_py_library",
    label_regex_for_dep = "{extension_name}",
)

# In tf_py_wrap_cc generated libraries
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

def tf_py_wrap_cc(
        name,
        srcs,
        swig_includes = [],
        deps = [],
        copts = [],
        version_script = None,
        **kwargs):
    """Builds a Python extension module."""
    module_name = name.split("/")[-1]

    # Convert a rule name such as foo/bar/baz to foo/bar/_baz.so
    # and use that as the name for the rule producing the .so file.
    cc_library_base = "/".join(name.split("/")[:-1] + ["_" + module_name])

    # TODO(b/137885063): tf_cc_shared_object needs to be cleaned up; we really
    # shouldn't be passing a name qualified with .so here.
    cc_library_name = cc_library_base + ".so"
    cc_library_pyd_name = "/".join(
        name.split("/")[:-1] + ["_" + module_name + ".pyd"],
    )
    extra_deps = []
    _py_wrap_cc(
        name = name + "_py_wrap",
        srcs = srcs,
        module_name = module_name,
        py_module_name = name,
        swig_includes = swig_includes,
        toolchain_deps = ["@bazel_tools//tools/cpp:current_cc_toolchain"],
        deps = deps + extra_deps,
    )
    if not version_script:
        version_script = select({
            "@local_config_cuda//cuda:darwin": clean_dep("//tensorflow:tf_exported_symbols.lds"),
            "//conditions:default": clean_dep("//tensorflow:tf_version_script.lds"),
        })
    vscriptname = name + "_versionscript"
    _append_init_to_versionscript(
        name = vscriptname,
        is_version_script = select({
            "@local_config_cuda//cuda:darwin": False,
            "//conditions:default": True,
        }),
        module_name = module_name,
        template_file = version_script,
    )
    extra_linkopts = select({
        "@local_config_cuda//cuda:darwin": [
            "-Wl,-exported_symbols_list,$(location %s.lds)" % vscriptname,
        ],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,--version-script",
            "$(location %s.lds)" % vscriptname,
        ],
    })
    extra_deps += select({
        "@local_config_cuda//cuda:darwin": [
            "%s.lds" % vscriptname,
        ],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "%s.lds" % vscriptname,
        ],
    })

    tf_cc_shared_object(
        name = cc_library_name,
        srcs = [module_name + ".cc"],
        copts = copts + if_not_windows([
            "-Wno-self-assign",
            "-Wno-sign-compare",
            "-Wno-write-strings",
        ]),
        linkopts = extra_linkopts,
        linkstatic = 1,
        deps = deps + extra_deps,
        **kwargs
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
            srcs = [":" + cc_library_name],
            outs = [name_os],
            cmd = "cp $< $@",
        )

    native.genrule(
        name = "gen_" + cc_library_pyd_name,
        srcs = [":" + cc_library_name],
        outs = [cc_library_pyd_name],
        cmd = "cp $< $@",
    )
    native.py_library(
        name = name,
        srcs = [":" + name + ".py"],
        srcs_version = "PY2AND3",
        data = select({
            clean_dep("//tensorflow:windows"): [":" + cc_library_pyd_name],
            "//conditions:default": [":" + cc_library_name],
        }),
    )

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
def py_test(deps = [], data = [], kernels = [], **kwargs):
    # Python version placeholder
    native.py_test(
        # TODO(jlebar): Ideally we'd use tcmalloc here.,
        deps = select({
            "//conditions:default": deps,
            clean_dep("//tensorflow:no_tensorflow_py_deps"): [],
        }),
        data = data + select({
            "//conditions:default": kernels,
            clean_dep("//tensorflow:no_tensorflow_py_deps"): ["//tensorflow/tools/pip_package:win_pip_package_marker"],
        }),
        exec_compatible_with = tf_exec_compatible_with(kwargs),
        **kwargs
    )

register_extension_info(
    extension_name = "py_test",
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

register_extension_info(
    extension_name = "py_binary",
    label_regex_for_dep = "{extension_name}",
)

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
        xla_test_true_list += ["//tensorflow/python:is_xla_test_true"]
    if xla_enabled:
        deps = deps + tf_additional_xla_deps_py()
    if grpc_enabled:
        deps = deps + tf_additional_grpc_deps_py()

    # Python version placeholder
    kwargs.setdefault("srcs_version", "PY2AND3")
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
        deps = depset([
            clean_dep("//tensorflow/python:extra_py_tests_deps"),
            clean_dep("//tensorflow/python:gradient_checker"),
        ] + deps + xla_test_true_list),
        **kwargs
    )

register_extension_info(
    extension_name = "tf_py_test",
    label_regex_map = {"deps": "deps:{extension_name}"},
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
        **kwargs):
    # TODO(b/122522101): Don't ignore xla_enable_strict_auto_jit and enable additional
    # XLA tests once enough compute resources are available.
    _ignored = [xla_enable_strict_auto_jit]
    if main == None:
        main = name + ".py"
    if "additional_deps" in kwargs:
        fail("Use `deps` to specify dependencies. `additional_deps` has been replaced with the standard pattern of `deps`.")
    for config in ["cpu", "gpu"]:
        test_name = name
        test_tags = tags
        if config == "gpu":
            test_name += "_gpu"
            test_tags = test_tags + tf_gpu_tests_tags()
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

register_extension_info(
    extension_name = "gpu_py_test",
    label_regex_map = {"deps": "deps:{extension_name}"},
)

# terminology changes: saving cuda_* definition for compatibility
def cuda_py_test(*args, **kwargs):
    gpu_py_test(*args, **kwargs)

register_extension_info(
    extension_name = "cuda_py_test",
    label_regex_map = {"deps": "deps:{extension_name}"},
)

def sycl_py_test(
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
        xla_enabled = False,
        grpc_enabled = False,
        **kwargs):
    test_tags = tags + tf_sycl_tests_tags()
    if "additional_deps" in kwargs:
        fail("Use `deps` to specify dependencies. `additional_deps` has been replaced with the standard pattern of `deps`.")
    tf_py_test(
        name = name,
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
        **kwargs
    )

register_extension_info(
    extension_name = "sycl_py_test",
    label_regex_map = {"deps": "deps:{extension_name}"},
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
            **kwargs
        )

def gpu_py_tests(
        name,
        srcs,
        size = "medium",
        kernels = [],
        data = [],
        shard_count = 1,
        tags = [],
        prefix = "",
        xla_enable_strict_auto_jit = False,
        xla_enabled = False,
        grpc_enabled = False,
        **kwargs):
    # TODO(b/122522101): Don't ignore xla_enable_strict_auto_jit and enable additional
    # XLA tests once enough compute resources are available.
    _ignored = [xla_enable_strict_auto_jit]
    test_tags = tags + tf_gpu_tests_tags()
    if "additional_deps" in kwargs:
        fail("Use `deps` to specify dependencies. `additional_deps` has been replaced with the standard pattern of `deps`.")
    py_tests(
        name = name,
        size = size,
        srcs = srcs,
        data = data,
        grpc_enabled = grpc_enabled,
        kernels = kernels,
        prefix = prefix,
        shard_count = shard_count,
        tags = test_tags,
        xla_enabled = xla_enabled,
        xla_enable_strict_auto_jit = False,
        **kwargs
    )

# terminology changes: saving cuda_* definition for compatibility
def cuda_py_tests(*args, **kwargs):
    gpu_py_tests(*args, **kwargs)

# Creates a genrule named <name> for running tools/proto_text's generator to
# make the proto_text functions, for the protos passed in <srcs>.
#
# Return a struct with fields (hdrs, srcs) containing the names of the
# generated files.
def tf_generate_proto_text_sources(name, srcs_relative_dir, srcs, protodeps = [], deps = [], visibility = None):
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
    )

    native.filegroup(
        name = name + "_hdrs",
        srcs = out_hdrs,
        visibility = visibility,
    )

    cc_library(
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

def tf_version_info_genrule(name, out):
    native.genrule(
        name = name,
        srcs = [
            clean_dep("@local_config_git//:gen/spec.json"),
            clean_dep("@local_config_git//:gen/head"),
            clean_dep("@local_config_git//:gen/branch_ref"),
        ],
        outs = [out],
        cmd =
            "$(location //tensorflow/tools/git:gen_git_source) --generate $(SRCS) \"$@\" --git_tag_override=$${GIT_TAG_OVERRIDE:-}",
        local = 1,
        tools = [clean_dep("//tensorflow/tools/git:gen_git_source")],
    )

def tf_py_build_info_genrule(name, out, **kwargs):
    native.genrule(
        name = name,
        outs = [out],
        cmd =
            "$(location //tensorflow/tools/build_info:gen_build_info) --raw_generate \"$@\" " +
            " --is_config_cuda " + if_cuda("True", "False") +
            " --is_config_rocm " + if_rocm("True", "False") +
            " --key_value " +
            if_cuda(" cuda_version_number=$${TF_CUDA_VERSION:-} cudnn_version_number=$${TF_CUDNN_VERSION:-} ", "") +
            if_windows(" msvcp_dll_names=msvcp140.dll,msvcp140_1.dll ", "") +
            if_windows_cuda(" ".join([
                "nvcuda_dll_name=nvcuda.dll",
                "cudart_dll_name=cudart64_$$(echo $${TF_CUDA_VERSION:-} | sed \"s/\\.//\").dll",
                "cudnn_dll_name=cudnn64_$${TF_CUDNN_VERSION:-}.dll",
            ]), ""),
        local = 1,
        tools = [clean_dep("//tensorflow/tools/build_info:gen_build_info")],
        **kwargs
    )

def cc_library_with_android_deps(
        deps,
        android_deps = [],
        common_deps = [],
        copts = tf_copts(),
        **kwargs):
    deps = if_not_android(deps) + if_android(android_deps) + common_deps
    cc_library(deps = deps, copts = copts, **kwargs)

register_extension_info(
    extension_name = "cc_library_with_android_deps",
    label_regex_for_dep = "{extension_name}",
)

def tensorflow_opensource_extra_deps():
    return []

# buildozer: disable=function-docstring-args
def pybind_extension(
        name,
        srcs,
        module_name,
        hdrs = [],
        features = [],
        srcs_version = "PY2AND3",
        data = [],
        copts = [],
        linkopts = [],
        deps = [],
        visibility = None,
        testonly = None,
        licenses = None,
        compatible_with = None,
        restricted_to = None,
        deprecation = None):
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
    pyd_file = "%s%s.pyd" % (prefix, sname)
    symbol = "init%s" % sname
    symbol2 = "init_%s" % sname
    symbol3 = "PyInit_%s" % sname
    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name
    native.genrule(
        name = name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '_%s\n_%s\n_%s' >$@" % (symbol, symbol2, symbol3),
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    native.genrule(
        name = name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n %s;\n %s;\n %s;\n local: *;};' >$@" % (symbol, symbol2, symbol3),
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )
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
            "@local_config_cuda//cuda:darwin": [
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
        features = features + ["-use_header_modules"],
        linkshared = 1,
        testonly = testonly,
        licenses = licenses,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )
    native.genrule(
        name = name + "_pyd_copy",
        srcs = [so_file],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )
    native.py_library(
        name = name,
        data = select({
            "@org_tensorflow//tensorflow:windows": [pyd_file],
            "//conditions:default": [so_file],
        }),
        srcs_version = srcs_version,
        licenses = licenses,
        testonly = testonly,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )

# buildozer: enable=function-docstring-args

def tf_python_pybind_extension(
        name,
        srcs,
        module_name,
        features = [],
        copts = [],
        hdrs = [],
        deps = [],
        visibility = None):
    """A wrapper macro for pybind_extension that is used in tensorflow/python/BUILD.

    It is used for targets under //third_party/tensorflow/python that link
    against libtensorflow_framework.so and pywrap_tensorflow_internal.so.
    """
    pybind_extension(
        name,
        srcs + tf_binary_additional_srcs(),
        module_name,
        features = features,
        copts = copts,
        hdrs = hdrs,
        deps = deps + tf_binary_pybind_deps() + mkl_deps(),
        visibility = visibility,
    )

def if_cuda_or_rocm(if_true, if_false = []):
    """Shorthand for select()'ing whether to build for either CUDA or ROCm.

      Returns a select statement which evaluates to
         if_true if we're building with either CUDA or ROCm enabled.
         if_false, otherwise.

      Sometimes a target has additional CUDa or ROCm specific dependencies.
      The `if_cuda` / `if_rocm` functions are used to specify these additional
      dependencies. For eg, see the `//tensorflow/core/kernels:bias_op` target

      If the same additional dependency is needed for both CUDA and ROCm
      (for eg. `reduction_ops` dependency for the `bias_op` target above),
      then specifying that dependency in both  both `if_cuda` and `if_rocm` will
      result in both those functions returning a select statement, which contains
      the same dependency, which then leads to a duplicate dependency bazel error.

      In order to work around this error, any additional dependency that is common
      to both the CUDA and ROCm platforms, should be specified using this function.
      Doing so will eliminate the cause of the bazel error (i.e. the  same
      dependency showing up in two different select statements)

      """
    return select({
        "@local_config_cuda//cuda:using_nvcc": if_true,
        "@local_config_cuda//cuda:using_clang": if_true,
        "@local_config_rocm//rocm:using_hipcc": if_true,
        "//conditions:default": if_false,
    })

def tf_jit_compilation_passes_extra_deps():
    return []

def if_mlir(if_true, if_false = []):
    return select({
        "//conditions:default": if_false,
        "//tensorflow:with_mlir_support": if_true,
    })

def tfcompile_extra_flags():
    return ""

def tf_external_workspace_visible(visibility):
    # External workspaces can see this target.
    return ["//visibility:public"]
