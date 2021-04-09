load(
    "@rules_cc//cc:defs.bzl",
    "cc_binary",
    "cc_test",
)

# version for the shared libraries, can
# not contain rc or alpha, only numbers.
# Also update tensorflow/core/public/version.h
# and tensorflow/tools/pip_package/setup.py
VERSION = "2.5.0"
VERSION_MAJOR = VERSION.split(".")[0]

tf_cc_test = cc_test

def py_test(deps = [], data = [], kernels = [], exec_properties = None, **kwargs):
    pass

def if_not_windows(a):
    return select({
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": a,
    })

def transitive_hdrs(name, deps = [], **kwargs):
    pass

def clean_dep(dep):
    return str(Label(dep))

def get_compatible_with_portable():
    return []

def get_compatible_with_cloud():
    return []

def tf_opts_nortti_if_android():
    return []

def tf_binary_additional_srcs(fullversion = False):
    if fullversion:
        suffix = "." + VERSION
    else:
        suffix = "." + VERSION_MAJOR

    return []

def tf_cc_shared_object(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = [],
        framework_so = [],
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

    names = [(
        name,
        name + suffix,
        name + longsuffix,
    )]

    for name_os, name_os_major, name_os_full in names:
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

        data_extra = []

        cc_binary(
            name = name_os_full,
            srcs = srcs + framework_so,
            deps = deps,
            linkshared = 1,
            data = data + data_extra,
            linkopts = linkopts,
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
