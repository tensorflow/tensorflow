"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load(
    "@python_version_repo//:py_version.bzl",
    "HERMETIC_PYTHON_VERSION",
    "HERMETIC_PYTHON_VERSION_KIND",
    "REQUIREMENTS_WITH_LOCAL_WHEELS",
)
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

    # NOTE: According to rules_python 1.8.4
    # rules_python doesn't distinguish between freethreaded and
    # non-freethreaded, it is a 1:1 mapping. This can cause issue with
    # HERMETIC_PYTHON_VERSION == 3.14-ft or any <version>-ft.
    # Which causes bazel to run python 3.14-ft but rules_python would
    # lazyly download requirements with non-freethreaded from requirements.txt.
    #
    # see spefici code below:
    # https://github.com/bazel-contrib/rules_python/blob/1.8.4
    # /python/private/pypi/pip_repository.bzl#L111

    extra_pip_args = []
    is_download_only = False
    if "ft" in HERMETIC_PYTHON_VERSION_KIND:
        version_num = HERMETIC_PYTHON_VERSION.split("-")[0].replace(".", "")
        extra_pip_args = ["--abi", "cp{}t".format(version_num)]
        is_download_only = True

    pip_parse(
        name = "pypi",
        annotations = numpy_annotations,
        python_interpreter_target = "@{}_host//:python".format(
            get_toolchain_name_per_python_version("python"),
        ),
        extra_hub_aliases = {
            "numpy": ["numpy_headers"],
        },
        # NOTE: (Required for rules_python >= 1.7.0)
        # pipstar flag default has been flipped to be on by default.
        # It can be disabled through RULES_PYTHON_ENABLE_PIPSTAR=0
        # environment variable.
        envsubst = ["RULES_PYTHON_ENABLE_PIPSTAR"],
        requirements_lock = REQUIREMENTS_WITH_LOCAL_WHEELS,
        extra_pip_args = extra_pip_args,
        download_only = is_download_only,
    )
