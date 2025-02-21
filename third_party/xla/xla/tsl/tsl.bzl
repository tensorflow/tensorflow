"""Provides build configuration for TSL"""

load("@bazel_skylib//lib:new_sets.bzl", "sets")
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)
load(
    "//xla/tsl/mkl:build_defs.bzl",
    "if_enable_mkl",
    "if_mkl",
    "if_mkldnn_aarch64_acl",
    "if_mkldnn_aarch64_acl_openmp",
    "if_mkldnn_openmp",
    "onednn_v3_define",
)
load(
    "//third_party/compute_library:build_defs.bzl",
    "if_enable_acl",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_binary",
    "cc_library",
    "cc_shared_library",
)
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)
load(
    "@local_tsl//third_party/py/rules_pywrap:pywrap.bzl",
    "use_pywrap_rules",
)

# Internally this loads a macro, but in OSS this is a function
# buildifier: disable=out-of-order-load
def register_extension_info(**_kwargs):
    pass

two_gpu_tags = ["requires-gpu-nvidia:2", "notap", "manual", "no_pip"]

def clean_dep(target):
    """Returns string to 'target' in the TSL repository.

    Use this function when referring to targets in the TSL
    repository from macros that may be called from external repositories.

    Args:
      target: the target to produce a canonicalized label for.
    Returns:
      The canonical label of the target.
    """

    # A repo-relative label is resolved relative to the file in which the
    # Label() call appears, e.g. @local_tsl or tsl.
    # TODO(ddunleavy): update this during and after go/moving-tsl-into-xla-lsc
    label = Label(target)
    not_yet_moved = ["concurrency", "framework", "lib", "platform", "profiler", "protobuf"]

    if any([label.package.startswith("tsl/" + dirname) for dirname in not_yet_moved]):
        return "@local_tsl//" + label.package + ":" + label.name
    else:
        return str(label)

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
      then specifying that dependency in both `if_cuda` and `if_rocm` will
      result in both those functions returning a select statement, which contains
      the same dependency, which then leads to a duplicate dependency bazel error.

      In order to work around this error, any additional dependency that is common
      to both the CUDA and ROCm platforms, should be specified using this function.
      Doing so will eliminate the cause of the bazel error (i.e. the  same
      dependency showing up in two different select statements)

      """
    return select({
        clean_dep("//xla/tsl:is_cuda_enabled"): if_true,
        "@local_config_rocm//rocm:using_hipcc": if_true,
        "//conditions:default": if_false,
    })

def if_oss(oss_value, google_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    _ = (google_value, oss_value)  # buildifier: disable=unused-variable
    return oss_value  # copybara:comment_replace return google_value

def if_google(google_value, oss_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    _ = (google_value, oss_value)  # buildifier: disable=unused-variable
    return oss_value  # copybara:comment_replace return google_value

def internal_visibility(internal_targets):
    """Returns internal_targets in g3, but returns public in OSS.

    Useful for targets that are part of the XLA/TSL API surface but want finer-grained visibilites
    internally.
    """
    return if_google(internal_targets, ["//visibility:public"])

# TODO(jakeharmon): Use this to replace if_static
# TODO(b/356020232): remove completely after migration is done
def if_tsl_link_protobuf(if_true, if_false = []):
    if use_pywrap_rules():
        return if_true
    return select({
        "//conditions:default": if_true,
        clean_dep("//xla/tsl:tsl_protobuf_header_only"): if_false,
    })

def if_libtpu(if_true, if_false = []):
    """Shorthand for select()ing whether to build backend support for TPUs when building libtpu.so"""
    return select({
        # copybara:uncomment_begin(different config setting in OSS)
        # "//tools/cc_target_os:gce": if_true,
        # copybara:uncomment_end_and_comment_begin
        clean_dep("//xla/tsl:with_tpu_support"): if_true,
        # copybara:comment_end
        "//conditions:default": if_false,
    })

def if_macos(a, otherwise = []):
    return select({
        clean_dep("//xla/tsl:macos"): a,
        "//conditions:default": otherwise,
    })

def if_windows(a, otherwise = []):
    return select({
        clean_dep("//xla/tsl:windows"): a,
        "//conditions:default": otherwise,
    })

def if_not_windows(a):
    return select({
        clean_dep("//xla/tsl:windows"): [],
        "//conditions:default": a,
    })

def if_not_fuchsia(a):
    return select({
        clean_dep("//xla/tsl:fuchsia"): [],
        "//conditions:default": a,
    })

def if_nvcc(a):
    return select({
        clean_dep("//xla/tsl:is_cuda_nvcc"): a,
        "//conditions:default": [],
    })

def if_xla_available(if_true, if_false = []):
    return select({
        clean_dep("//xla/tsl:with_xla_support"): if_true,
        "//conditions:default": if_false,
    })

def if_android_arm(a):
    return select({
        clean_dep("//xla/tsl:android_arm"): a,
        "//conditions:default": [],
    })

def if_not_android(a):
    return select({
        clean_dep("//xla/tsl:android"): [],
        "//conditions:default": a,
    })

def if_linux_x86_64(a):
    return select({
        clean_dep("//xla/tsl:linux_x86_64"): a,
        "//conditions:default": [],
    })

def if_ios_x86_64(a):
    return select({
        clean_dep("//xla/tsl:ios_x86_64"): a,
        "//conditions:default": [],
    })

def if_no_default_logger(a):
    return select({
        clean_dep("//xla/tsl:no_default_logger"): a,
        "//conditions:default": [],
    })

# Enabled unless Windows or actively disabled, even without --config=cuda.
# Combine with 'if_gpu_is_configured' (XLA) or 'if_cuda_or_rocm' (otherwise).
def if_nccl(if_true, if_false = []):
    return select({
        clean_dep("//xla/tsl:no_nccl_support"): if_false,
        clean_dep("//xla/tsl:windows"): if_false,
        clean_dep("//xla/tsl:arm"): if_false,
        "//conditions:default": if_true,
    })

def if_with_tpu_support(if_true, if_false = []):
    """Shorthand for select()ing whether to build API support for TPUs when building TSL"""
    return select({
        clean_dep("//xla/tsl:with_tpu_support"): if_true,
        "//conditions:default": if_false,
    })

# These configs are used to determine whether we should use CUDA tools and libs in cc_libraries.
# They are intended for the OSS builds only.
def if_cuda_tools(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we're building with hCUDA tools."""
    return select({"@local_config_cuda//cuda:cuda_tools": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false

def if_cuda_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic CUDA libraries."""
    return select({"@local_config_cuda//cuda:cuda_tools_and_libs": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false

def get_win_copts(is_external = False):
    WINDOWS_COPTS = [
        # copybara:uncomment_begin(no MSVC flags in google)
        # "-DPLATFORM_WINDOWS",
        # "-DEIGEN_HAS_C99_MATH",
        # "-DTENSORFLOW_USE_EIGEN_THREADPOOL",
        # "-DEIGEN_AVOID_STL_ARRAY",
        # "-Iexternal/gemmlowp",
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
        # copybara:comment_end
        # Also see build:windows lines in tensorflow/opensource_only/.bazelrc
        # where we set some other options globally.
    ]

    if is_external:
        # copybara:uncomment_begin(no MSVC flags in google)
        # return WINDOWS_COPTS + ["-UTF_COMPILE_LIBRARY"]
        # copybara:uncomment_end_and_comment_begin
        return WINDOWS_COPTS + ["/UTF_COMPILE_LIBRARY"]
        # copybara:comment_end

    else:
        # copybara:uncomment_begin(no MSVC flags in google)
        # return WINDOWS_COPTS + ["-DTF_COMPILE_LIBRARY"]
        # copybara:uncomment_end_and_comment_begin
        return WINDOWS_COPTS + ["/DTF_COMPILE_LIBRARY"]
        # copybara:comment_end

# TODO(b/356020232): cleanup non-use_pywrap_rules part once migration is done
# buildozer: disable=function-docstring-args
def tsl_copts(
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
    ]
    if android_optimization_level_override:
        android_copts.append(android_optimization_level_override)

    framework_deps = []
    if use_pywrap_rules():
        pass
    else:
        framework_deps = select({
            clean_dep("//xla/tsl:framework_shared_object"): [],
            "//conditions:default": ["-DTENSORFLOW_MONOLITHIC_BUILD"],
        })

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
        onednn_v3_define() +
        if_mkldnn_aarch64_acl(["-DDNNL_AARCH64_USE_ACL=1"]) +
        if_mkldnn_aarch64_acl_openmp(["-DENABLE_ONEDNN_OPENMP"]) +
        if_enable_acl(["-DXLA_CPU_USE_ACL=1", "-fexceptions"]) +
        if_android_arm(["-mfpu=neon", "-fomit-frame-pointer"]) +
        if_linux_x86_64(["-msse3"]) +
        if_ios_x86_64(["-msse4.1"]) +
        if_no_default_logger(["-DNO_DEFAULT_LOGGER"]) +
        framework_deps +
        select({
            clean_dep("//xla/tsl:android"): android_copts,
            clean_dep("//xla/tsl:emscripten"): [],
            clean_dep("//xla/tsl:macos"): [],
            clean_dep("//xla/tsl:windows"): get_win_copts(is_external),
            clean_dep("//xla/tsl:ios"): [],
            clean_dep("//xla/tsl:no_lgpl_deps"): ["-D__TENSORFLOW_NO_LGPL_DEPS__", "-pthread"],
            "//conditions:default": ["-pthread"],
        })
    )

def tf_openmp_copts():
    # We assume when compiling on Linux gcc/clang will be used and MSVC on Windows
    return select({
        clean_dep("//xla/tsl/mkl:build_with_mkl_lnx_openmp"): ["-fopenmp"],
        # copybara:uncomment_begin
        # "//xla/tsl/mkl:build_with_mkl_windows_openmp": ["/openmp"],
        # copybara:uncomment_end_and_comment_begin
        clean_dep("//xla/tsl/mkl:build_with_mkl_windows_openmp"): ["/openmp:llvm"],
        # copybara:comment_end
        "//conditions:default": [],
    })

def tsl_gpu_library(
        deps = None,
        cuda_deps = None,
        copts = tsl_copts(),
        add_gpu_deps_for_oss = True,
        **kwargs):
    """Generate a cc_library with a conditional set of CUDA dependencies.

    When the library is built with --config=cuda:

    - Both deps and cuda_deps are used as dependencies.
    - The cuda runtime is added as a dependency (if necessary).
    - The library additionally passes -DGOOGLE_CUDA=1 to the list of copts.
    - In addition, when the library is also built with TensorRT enabled, it
        additionally passes -DGOOGLE_TENSORRT=1 to the list of copts.

    Args:
      cuda_deps: BUILD dependencies which will be linked if and only if:
        '--config=cuda' is passed to the bazel command line.
      deps: dependencies which will always be linked.
      copts: copts always passed to the cc_library.
      add_gpu_deps_for_oss: Whether to add gpu deps for OSS too.
      **kwargs: Any other argument to cc_library.
    """
    if not deps:
        deps = []
    if not cuda_deps:
        cuda_deps = []

    kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]
    deps = deps + (if_cuda(cuda_deps) if add_gpu_deps_for_oss else if_google(if_cuda(cuda_deps)))
    if "default_copts" in kwargs:
        copts = kwargs["default_copts"] + copts
        kwargs.pop("default_copts", None)
    all_cuda_deps = if_cuda([
        clean_dep("//xla/tsl/cuda:cudart"),
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm([
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
    ])
    all_cuda_copts = if_cuda(["-DGOOGLE_CUDA=1", "-DNV_CUDNN_DISABLE_EXCEPTION"]) + if_rocm(["-DTENSORFLOW_USE_ROCM=1"])
    if not add_gpu_deps_for_oss:
        all_cuda_deps = if_google(all_cuda_deps)
        all_cuda_copts = if_google(all_cuda_copts)
    cc_library(
        deps = deps + all_cuda_deps,
        copts = (copts + all_cuda_copts + if_xla_available(["-DTENSORFLOW_USE_XLA=1"]) + if_mkl(["-DINTEL_MKL=1"]) + if_enable_mkl(["-DENABLE_MKL"]) + if_tensorrt(["-DGOOGLE_TENSORRT=1"])),
        **kwargs
    )

register_extension_info(extension = tsl_gpu_library, label_regex_for_dep = "{extension_name}")

# Traverse the dependency graph along the "deps" attribute of the
# target and return a struct with one field called 'tf_collected_deps'.
# tf_collected_deps will be the union of the deps of the current target
# and the tf_collected_deps of the dependencies of this target.
def _collect_deps_aspect_impl(target, ctx):  # buildifier: disable=unused-variable
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
                    _dep_label(input_dep) + " cannot depend on " +
                    _dep_label(disallowed_dep),
                )
        for required_dep in required_deps:
            if not sets.contains(collected_deps, required_dep.label):
                fail(
                    _dep_label(input_dep) + " must depend on " +
                    _dep_label(required_dep),
                )
    return []

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

def get_compatible_with_portable():
    return []

def get_compatible_with_libtpu_portable():
    return []

def filegroup(**kwargs):
    native.filegroup(**kwargs)

def internal_hlo_deps():
    return []

# Config setting selector used when building for products
# which requires restricted licenses to be avoided.
def if_not_mobile_or_arm_or_macos_or_lgpl_restricted(a):
    _ = (a,)  # buildifier: disable=unused-variable
    return select({
        "//conditions:default": [],
    })

def tsl_grpc_cc_dependencies():
    return [clean_dep("//xla/tsl:grpc++")]

# Bazel rule for collecting the header files that a target depends on.
def _transitive_hdrs_impl(ctx):
    outputs = _get_transitive_headers([], ctx.attr.deps)
    return DefaultInfo(files = outputs)

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
    native.filegroup(name = name, srcs = [":" + name + "_gather"], **kwargs)

def cc_header_only_library(name, deps = [], includes = [], extra_deps = [], compatible_with = None, **kwargs):
    if use_pywrap_rules():
        cc_library(
            name = name,
            deps = deps + extra_deps,
            compatible_with = compatible_with,
            **kwargs
        )
    else:
        custom_op_cc_header_only_library(
            name,
            deps,
            includes,
            extra_deps,
            compatible_with,
            **kwargs
        )

# Create a header only library that includes all the headers exported by
# the libraries in deps.
#
# **NOTE**: The headers brought in are **NOT** fully transitive; certain
# deep headers may be missing.  If this creates problems, you must find
# a header-only version of the cc_library rule you care about and link it
# *directly* in addition to your use of the cc_header_only_library
# intermediary.
#
# For:
#   * Eigen: it's a header-only library.  Add it directly to your deps.
#   * GRPC: add a direct dep on @com_github_grpc_grpc//:grpc++.
#
def custom_op_cc_header_only_library(name, deps = [], includes = [], extra_deps = [], compatible_with = None, **kwargs):
    _transitive_hdrs(
        name = name + "_gather",
        deps = deps,
        compatible_with = compatible_with,
    )
    _transitive_parameters_library(
        name = name + "_gathered_parameters",
        original_deps = deps,
        compatible_with = compatible_with,
    )
    cc_library(
        name = name,
        hdrs = [":" + name + "_gather"],
        includes = includes,
        compatible_with = compatible_with,
        deps = [":" + name + "_gathered_parameters"] + extra_deps,
        **kwargs
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

# Bazel rule for collecting the transitive parameters from a set of dependencies into a library.
# Propagates defines and includes.
def _transitive_parameters_library_impl(ctx):
    defines = depset(
        transitive = [dep[CcInfo].compilation_context.defines for dep in ctx.attr.original_deps],
    )
    system_includes = depset(
        transitive = [dep[CcInfo].compilation_context.system_includes for dep in ctx.attr.original_deps],
    )
    includes = depset(
        transitive = [dep[CcInfo].compilation_context.includes for dep in ctx.attr.original_deps],
    )
    quote_includes = depset(
        transitive = [dep[CcInfo].compilation_context.quote_includes for dep in ctx.attr.original_deps],
    )
    framework_includes = depset(
        transitive = [dep[CcInfo].compilation_context.framework_includes for dep in ctx.attr.original_deps],
    )
    return CcInfo(
        compilation_context = cc_common.create_compilation_context(
            defines = depset(direct = defines.to_list()),
            system_includes = depset(direct = system_includes.to_list()),
            includes = depset(direct = includes.to_list()),
            quote_includes = depset(direct = quote_includes.to_list()),
            framework_includes = depset(direct = framework_includes.to_list()),
        ),
    )

_transitive_parameters_library = rule(
    attrs = {
        "original_deps": attr.label_list(
            allow_empty = True,
            allow_files = True,
            providers = [CcInfo],
        ),
    },
    implementation = _transitive_parameters_library_impl,
)

# buildozer: disable=function-docstring-args
def tsl_pybind_extension_opensource(
        name,
        srcs,
        module_name = None,  # @unused
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
        enable_stub_generation = False,  # @unused
        features = [],
        licenses = None,
        linkopts = [],
        pytype_deps = [],
        pytype_srcs = [],
        restricted_to = None,
        srcs_version = "PY3",
        testonly = None,
        visibility = None,
        win_def_file = None):  # @unused
    """Builds a generic Python extension module."""
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
                clean_dep("//xla/tsl:windows"): [],
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
            user_link_flags = linkopts + select({
                clean_dep("//xla/tsl:macos"): [
                    # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                    # not being exported.  There should be a better way to deal with this.
                    "-Wl,-w",
                    "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
                ],
                clean_dep("//xla/tsl:windows"): [],
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
        cc_binary(
            name = so_file,
            srcs = srcs + hdrs,
            data = data,
            copts = copts + [
                "-fno-strict-aliasing",
                "-fexceptions",
            ] + select({
                clean_dep("//xla/tsl:windows"): [],
                "//conditions:default": [
                    "-fvisibility=hidden",
                ],
            }),
            linkopts = linkopts + select({
                clean_dep("//xla/tsl:macos"): [
                    # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                    # not being exported.  There should be a better way to deal with this.
                    "-Wl,-w",
                    "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
                ],
                clean_dep("//xla/tsl:windows"): [],
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
            clean_dep("//xla/tsl:windows"): [pyd_file],
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

def nvtx_headers():
    return if_oss(["@nvtx_archive//:headers"], ["@local_config_cuda//cuda:cuda_headers"])

def tsl_google_bzl_deps():
    return []

def tsl_extra_config_settings():
    pass

def tsl_extra_config_settings_targets():
    return []

# TODO(b/356020232): remove after migration is done
tsl_pybind_extension = tsl_pybind_extension_opensource
