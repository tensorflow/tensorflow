"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("@python_version_repo//:py_version.bzl", "REQUIREMENTS_WITH_LOCAL_WHEELS")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")
load(
    "//third_party/py:python_init_toolchains.bzl",
    "get_toolchain_name_per_python_version",
)

def python_init_pip():
    numpy_annotations = {
        "numpy": package_annotation(
            additive_build_content = """\
cc_library(
    name = "numpy_headers_2",
    hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
    strip_include_prefix="site-packages/numpy/_core/include/",
)
cc_library(
    name = "numpy_headers_1",
    hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
    strip_include_prefix="site-packages/numpy/core/include/",
)
cc_library(
    name = "numpy_headers",
    deps = [":numpy_headers_2", ":numpy_headers_1"],
    # For the layering check to work we need to re-export the headers from the
    # dependencies.
    hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]) +
           glob(["site-packages/numpy/core/include/**/*.h"]),
)
""",
        ),
    }

    pip_parse(
        name = "pypi",
        annotations = numpy_annotations,
        python_interpreter_target = "@{}_host//:python".format(
            get_toolchain_name_per_python_version("python"),
        ),
        extra_hub_aliases = {
            "numpy": ["numpy_headers"],
        },
        requirements_lock = REQUIREMENTS_WITH_LOCAL_WHEELS,
    )

python_init_pip_ext = module_extension(
    implementation = lambda mctx: python_init_pip(),
)
