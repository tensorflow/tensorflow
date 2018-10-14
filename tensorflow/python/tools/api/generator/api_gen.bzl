"""Targets for generating TensorFlow Python API __init__.py files."""

load("//tensorflow/python/tools/api/generator:api_init_files.bzl", "TENSORFLOW_API_INIT_FILES")

# keep sorted
ESTIMATOR_API_INIT_FILES = [
    # BEGIN GENERATED ESTIMATOR FILES
    "__init__.py",
    "estimator/__init__.py",
    "estimator/export/__init__.py",
    "estimator/inputs/__init__.py",
    # END GENERATED ESTIMATOR FILES
]

def get_compat_files(
        file_paths,
        compat_api_version):
    """Prepends compat/v<compat_api_version> to file_paths."""
    return ["compat/v%d/%s" % (compat_api_version, f) for f in file_paths]

def gen_api_init_files(
        name,
        output_files = TENSORFLOW_API_INIT_FILES,
        root_init_template = None,
        srcs = [],
        api_name = "tensorflow",
        api_version = 2,
        compat_api_versions = [],
        package = "tensorflow.python",
        package_dep = "//tensorflow/python:no_contrib",
        output_package = "tensorflow",
        output_dir = ""):
    """Creates API directory structure and __init__.py files.

    Creates a genrule that generates a directory structure with __init__.py
    files that import all exported modules (i.e. modules with tf_export
    decorators).

    Args:
      name: name of genrule to create.
      output_files: List of __init__.py files that should be generated.
        This list should include file name for every module exported using
        tf_export. For e.g. if an op is decorated with
        @tf_export('module1.module2', 'module3'). Then, output_files should
        include module1/module2/__init__.py and module3/__init__.py.
      root_init_template: Python init file that should be used as template for
        root __init__.py file. "# API IMPORTS PLACEHOLDER" comment inside this
        template will be replaced with root imports collected by this genrule.
      srcs: genrule sources. If passing root_init_template, the template file
        must be included in sources.
      api_name: Name of the project that you want to generate API files for
        (e.g. "tensorflow" or "estimator").
      api_version: TensorFlow API version to generate. Must be either 1 or 2.
      compat_api_versions: Older TensorFlow API versions to generate under
        compat/ directory.
      package: Python package containing the @tf_export decorators you want to
        process
      package_dep: Python library target containing your package.
      output_package: Package where generated API will be added to.
      output_dir: Subdirectory to output API to.
        If non-empty, must end with '/'.
    """
    root_init_template_flag = ""
    if root_init_template:
        root_init_template_flag = "--root_init_template=$(location " + root_init_template + ")"

    api_gen_binary_target = ("create_" + package + "_api_%d") % api_version
    native.py_binary(
        name = api_gen_binary_target,
        srcs = ["//tensorflow/python/tools/api/generator:create_python_api.py"],
        main = "//tensorflow/python/tools/api/generator:create_python_api.py",
        srcs_version = "PY2AND3",
        visibility = ["//visibility:public"],
        deps = [
            package_dep,
            "//tensorflow/python:util",
            "//tensorflow/python/tools/api/generator:doc_srcs",
        ],
    )

    all_output_files = ["%s%s" % (output_dir, f) for f in output_files]
    compat_api_version_flags = ""
    for compat_api_version in compat_api_versions:
        compat_api_version_flags += " --compat_apiversion=%d" % compat_api_version

    native.genrule(
        name = name,
        outs = all_output_files,
        cmd = (
            "$(location :" + api_gen_binary_target + ") " +
            root_init_template_flag + " --apidir=$(@D)" + output_dir +
            " --apiname=" + api_name + " --apiversion=" + str(api_version) +
            compat_api_version_flags + " --package=" + package +
            " --output_package=" + output_package + " $(OUTS)"
        ),
        srcs = srcs,
        tools = [":" + api_gen_binary_target],
        visibility = [
            "//tensorflow:__pkg__",
            "//tensorflow/tools/api/tests:__pkg__",
        ],
    )
