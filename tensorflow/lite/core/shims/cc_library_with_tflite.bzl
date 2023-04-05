"""Definitions for targets that use the TFLite shims."""

load("//tensorflow:tensorflow.bzl", "clean_dep")
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_copts_warnings",
    "tflite_custom_c_library",
    "tflite_jni_binary",
)
load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

def _concat(lists):
    """Concatenate a list of lists, without requiring the inner lists to be iterable.

    This allows the inner lists to be obtained by calls to select().
    """
    result = []
    for selected_list in lists:
        result = result + selected_list
    return result

def _is_not_package_relative_label(label):
    """Tests whether a label is not relative to the current package."""
    return label.startswith("//") or label.startswith("@")

def add_suffix(label, suffix):
    """Appends a suffix to the given label.

    Args:
      label: (str) target in relative or absolute form.
      suffix: (str) string to add after the label.

    Returns:
      Suffixed label in relative or absolute form.
    """
    if label.find(":") == -1 and _is_not_package_relative_label(label):
        # Handles labels like @repo//foo/bar, which is equivalent to
        # @repo//foo/bar:bar.
        _, _, target = label.rpartition("/")
        return "%s:%s%s" % (label, target, suffix)
    return label + suffix

def alias_with_tflite(name, actual, generate_opaque_delegate_target = False, **kwargs):
    """Defines an alias for a target that uses the TFLite shims.

    This rule 'alias_with_tflite' should be used instead of the native
    'alias' rule whenever the 'actual' target that is being aliased
    is defined using one of the *_with_tflite build macros.

    Args:
      name: determines the name used for the alias target.
      actual: the target that the alias target is aliased to.
      generate_opaque_delegate_target: (bool) If set, generates an additional
        alias for the _opaque_delegate variant.
      **kwargs: additional alias parameters.
    """
    native.alias(name = name, actual = actual, **kwargs)
    if generate_opaque_delegate_target:
        native.alias(name = name + "_opaque_delegate", actual = actual + "_opaque_delegate", **kwargs)

def android_library_with_tflite(
        name,
        deps = [],
        tflite_deps = [],
        exports = [],
        tflite_exports = [],
        **kwargs):
    """Defines an android_library that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' dependency on one of the "shim"
    library targets from //third_party/tensorflow/lite/core/shims:*.

    Args:
      name: as for android_library.
      deps: as for android_library.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite' / 'android_library_with_tflite'.
      exports: same as for android_library.
      tflite_exports: exported dependencies that are themselves defined using
        'cc_library_with_tflite' / 'android_library_with_tflite'.
      **kwargs: Additional android_library parameters.
    """
    android_library(
        name = name,
        exports = exports + tflite_exports,
        deps = deps + tflite_deps,
        **kwargs
    )

def cc_library_with_tflite(
        name,
        srcs = [],
        tflite_jni_binaries = [],
        deps = [],
        tflite_deps = [],
        tflite_deps_selects = [],
        generate_opaque_delegate_target = False,
        **kwargs):
    """Defines a cc_library that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' dependency on one of the "shim"
    library targets from //tensorflow/lite/core/shims:*.

    Args:
      name: as for cc_library.
      srcs: as for cc_library.
      tflite_jni_binaries: dependencies on shared libraries that are defined
        using 'jni_binary_with_tflite'.
      deps: as for cc_library.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite'.
      tflite_deps_selects: A list of dictionaries that will be converted to dependencies
        with select on rules.
      generate_opaque_delegate_target: (bool) If set, generates an additional
        cc_library target, which has "_opaque_delegate" appended to the name.
        The target depends on
        //third_party/tensorflow/lite/core/shims:tflite_use_opaque_delegate
        which enables the truly opaque delegate type. This macro ensures that
        dependencies listed in 'tflite_deps' use _opaque_delegate variant.
      **kwargs: Additional cc_library parameters.
    """
    native.cc_library(
        name = name,
        srcs = srcs + tflite_jni_binaries,
        deps = deps + tflite_deps + _concat([select(map) for map in tflite_deps_selects]),
        **kwargs
    )

    if generate_opaque_delegate_target:
        tflite_deps_renamed = [add_suffix(tflite_dep, "_opaque_delegate") for tflite_dep in tflite_deps]
        tflite_deps_selects_renamed = [{key: [add_suffix(dep, "_opaque_delegate") for dep in value] for (key, value) in map.items()} for map in tflite_deps_selects]
        native.cc_library(
            name = name + "_opaque_delegate",
            srcs = srcs + tflite_jni_binaries,
            deps = deps + tflite_deps_renamed + _concat([select(map) for map in tflite_deps_selects_renamed]) + [
                clean_dep("//tensorflow/lite/core/shims:tflite_use_opaque_delegate"),
            ],
            **kwargs
        )

def _label(target):
    """Return a Label <https://bazel.build/rules/lib/Label#Label> given a string.

    Args:
      target: (string) a relative or absolute build target.
    """
    if target[0:2] == "//":
        return Label(target)
    if target[0] == ":":
        return Label("//" + native.package_name() + target)
    return Label("//" + native.package_name() + ":" + target)

def cc_library_with_tflite_with_c_headers_test(name, hdrs, **kwargs):
    """Defines a C++ library with C-compatible header files.

    This generates a cc_library rule, but also generates
    build tests that verify that each of the 'hdrs'
    can be successfully built in a C (not C++!) compilation unit
    that directly includes only that header file.

    Args:
      name: (string) as per cc_library.
      hdrs: (list of string) as per cc_library.
      **kwargs: Additional kwargs to pass to cc_library.
    """
    cc_library_with_tflite(name = name, hdrs = hdrs, **kwargs)

    build_tests = []
    for hdr in hdrs:
        label = _label(hdr)
        basename = "%s__test_self_contained_c__%s" % (name, label.name)
        native.genrule(
            name = "%s_gen" % basename,
            outs = ["%s.c" % basename],
            cmd = "echo '#include \"%s/%s\"' > $@" % (label.package, label.name),
            visibility = ["//visibility:private"],
            testonly = True,
        )
        cc_library_with_tflite(
            name = "%s_lib" % basename,
            srcs = ["%s.c" % basename],
            deps = [":" + name],
            copts = kwargs.get("copts", []),
            visibility = ["//visibility:private"],
            testonly = True,
            tags = ["allow_undefined_symbols"],
        )
        build_test(
            name = "%s_build_test" % basename,
            visibility = ["//visibility:private"],
            targets = ["%s_lib" % basename],
        )
        build_tests.append("%s_build_test" % basename)

def cc_library_with_stable_tflite_abi(
        deps = [],
        non_stable_abi_deps = [],
        stable_abi_deps = [],  # @unused
        **kwargs):
    """Defines a cc_library that uses the TFLite shims.

    This is a proxy method for cc_library_with_tflite() for targets that use
    the TFLite shims.

    Args:
      deps: Same as for cc_library_with_tflite.
      non_stable_abi_deps: dependencies that will be enabled only when NOT
        using TFLite with stable ABI.  This should be used for dependencies
        arising from code inside '#if !TFLITE_WITH_STABLE_ABI'.
      stable_abi_deps: dependencies that will be enabled only when using TFLite
        with stable ABI. This should be used for dependencies arising from code
        inside '#if TFLITE_WITH_STABLE_ABI'.
      **kwargs: Additional cc_library_with_tflite parameters.
    """
    cc_library_with_tflite(
        deps = deps + non_stable_abi_deps,
        **kwargs
    )

def cc_test_with_tflite(
        name,
        deps = [],
        tflite_deps = [],
        **kwargs):
    """Defines a cc_test that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' dependency on one of the "shim"
    library targets from //third_party/tensorflow/lite/core/shims:*.

    Args:
      name: as for cc_test.
      deps: as for cc_test.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite'.
      **kwargs: Additional cc_test parameters.
    """
    native.cc_test(
        name = name,
        deps = deps + tflite_deps,
        **kwargs
    )

def java_library_with_tflite(
        name,
        deps = [],
        runtime_deps = [],
        tflite_deps = [],
        tflite_jni_binaries = [],
        exports = [],
        tflite_exports = [],
        **kwargs):
    """Defines a java_library that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' or 'tflite_jni_binaries' dependency
    on one of the "shim" library targets from
    //third_party/tensorflow/lite/core/shims:*.

    Args:
      name: as for java_library.
      deps: as for java_library.
      runtime_deps: as for java_library.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite' / 'java_library_with_tflite'.
      tflite_jni_binaries: dependencies on shared libraries that are defined
        using 'jni_binary_with_tflite'.
      exports: same as for java_library.
      tflite_exports: exported dependencies that are themselves defined using
        'cc_library_with_tflite' / 'java_library_with_tflite'.
      **kwargs: Additional java_library parameters.
    """
    native.java_library(
        name = name,
        exports = exports + tflite_exports,
        deps = deps + tflite_deps + tflite_jni_binaries,
        **kwargs
    )

def java_test_with_tflite(
        name,
        deps = [],
        runtime_deps = [],
        tflite_deps = [],
        tflite_jni_binaries = [],
        **kwargs):
    """Defines a java_library that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' or 'tflite_jni_binaries' dependency
    on one of the "shim" library targets from
    //third_party/tensorflow/lite/core/shims:*.

    Args:
      name: as for java_library.
      deps: as for java_library.
      runtime_deps: as for java_library.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite' / 'java_library_with_tflite'.
      tflite_jni_binaries: dependencies on shared libraries that are defined
        using 'jni_binary_with_tflite'.
      **kwargs: Additional java_library parameters.
    """
    native.java_test(
        name = name,
        deps = deps + tflite_deps,
        runtime_deps = deps + tflite_jni_binaries,
        **kwargs
    )

def jni_binary_with_tflite(
        name,
        deps = [],
        tflite_deps = [],
        **kwargs):
    """Defines a tflite_jni_binary that uses the TFLite shims.

    This is a hook to allow applying different build flags (etc.)
    for targets that use the TFLite shims.

    Note that this build rule doesn't itself add any dependencies on
    TF Lite; this macro should normally be used in conjunction with a
    direct or indirect 'tflite_deps' dependency on one of the "shim"
    library targets from //third_party/tensorflow/lite/core/shims:*.

    Args:
      name: as for tflite_jni_binary.
      deps: as for tflite_jni_binary.
      tflite_deps: dependencies on rules that are themselves defined using
        'cc_library_with_tflite'.
      **kwargs: Additional tflite_jni_binary parameters.
    """
    tflite_jni_binary(
        name = name,
        deps = deps + tflite_deps,
        **kwargs
    )

def custom_c_library_with_tflite(
        name,
        models = [],
        experimental = False,
        **kwargs):
    """Generates a tflite c library, stripping off unused operators.

    This library includes the C API and the op kernels used in the given models.

    Args:
        name: Str, name of the target.
        models: List of models. This TFLite build will only include
            operators used in these models. If the list is empty, all builtin
            operators are included.
        experimental: Whether to include experimental APIs or not.
       **kwargs: kwargs to cc_library_with_tflite.
    """
    tflite_custom_c_library(
        name = "%s_c_api" % name,
        models = models,
        experimental = experimental,
    )

    if experimental:
        hdrs = [
            clean_dep("//tensorflow/lite/core/shims:c/c_api.h"),
            clean_dep("//tensorflow/lite/core/shims:c/c_api_experimental.h"),
            clean_dep("//tensorflow/lite/core/shims:c/c_api_opaque.h"),
        ]
    else:
        hdrs = [
            clean_dep("//tensorflow/lite/core/shims:c/c_api.h"),
        ]

    cc_library_with_tflite(
        name = name,
        hdrs = hdrs,
        copts = tflite_copts_warnings(),
        deps = [
            ":%s_c_api" % name,
        ],
        **kwargs
    )
