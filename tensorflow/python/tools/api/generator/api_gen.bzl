"""Targets for generating TensorFlow Python API __init__.py files."""

load("//tensorflow:tensorflow.bzl", "if_indexing_source_code")
load("//tensorflow/python/tools/api/generator:api_init_files.bzl", "TENSORFLOW_API_INIT_FILES")

def get_compat_files(
        file_paths,
        compat_api_version):
    """Prepends compat/v<compat_api_version> to file_paths."""
    return ["compat/v%d/%s" % (compat_api_version, f) for f in file_paths]

def get_nested_compat_files(compat_api_versions):
    """Return __init__.py file paths for files under nested compat modules.

    A nested compat module contains two __init__.py files:
      1. compat/vN/compat/vK/__init__.py
      2. compat/vN/compat/vK/compat/__init__.py

    Args:
      compat_api_versions: list of compat versions.

    Returns:
      List of __init__.py file paths to include under nested compat modules.
    """
    files = []
    for v in compat_api_versions:
        files.extend([
            "compat/v%d/compat/v%d/__init__.py" % (v, sv)
            for sv in compat_api_versions
        ])
        files.extend([
            "compat/v%d/compat/v%d/compat/__init__.py" % (v, sv)
            for sv in compat_api_versions
        ])
    return files

def gen_api_init_files(
        name,
        output_files = TENSORFLOW_API_INIT_FILES,
        root_init_template = None,
        srcs = [],
        api_name = "tensorflow",
        api_version = 2,
        compat_api_versions = [],
        compat_init_templates = [],
        packages = [
            "tensorflow.python",
            "tensorflow.dtensor.python.api",
            "tensorflow.dtensor.python.d_checkpoint",
            "tensorflow.dtensor.python.d_variable",
            "tensorflow.dtensor.python.layout",
            "tensorflow.dtensor.python.mesh_util",
            "tensorflow.dtensor.python.save_restore",
            "tensorflow.dtensor.python.tpu_util",
            "tensorflow.lite.python.analyzer",
            "tensorflow.lite.python.lite",
            "tensorflow.lite.python.authoring.authoring",
            "tensorflow.python.modules_with_exports",
        ],
        package_deps = [
            "//tensorflow/python:no_contrib",
            "//tensorflow/python:modules_with_exports",
        ],
        output_package = "tensorflow",
        output_dir = "",
        root_file_name = "__init__.py"):
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
      compat_init_templates: Python init file that should be used as template
        for top level __init__.py files under compat/vN directories.
        "# API IMPORTS PLACEHOLDER" comment inside this
        template will be replaced with root imports collected by this genrule.
      packages: Python packages containing the @tf_export decorators you want to
        process
      package_deps: Python library target containing your packages.
      output_package: Package where generated API will be added to.
      output_dir: Subdirectory to output API to.
        If non-empty, must end with '/'.
      root_file_name: Name of the root file with all the root imports.
    """
    root_init_template_flag = ""
    if root_init_template:
        root_init_template_flag = "--root_init_template=" + root_init_template

    primary_package = packages[0]
    api_gen_binary_target = ("create_" + primary_package + "_api_%s") % name
    native.py_binary(
        name = api_gen_binary_target,
        srcs = ["//tensorflow/python/tools/api/generator:create_python_api.py"],
        main = "//tensorflow/python/tools/api/generator:create_python_api.py",
        python_version = "PY3",
        srcs_version = "PY3",
        visibility = ["//visibility:public"],
        deps = package_deps + [
            "//tensorflow/python:util",
            "//tensorflow/python/tools/api/generator:doc_srcs",
        ],
    )

    # Replace name of root file with root_file_name.
    output_files = [
        root_file_name if f == "__init__.py" else f
        for f in output_files
    ]
    all_output_files = ["%s%s" % (output_dir, f) for f in output_files]
    compat_api_version_flags = ""
    for compat_api_version in compat_api_versions:
        compat_api_version_flags += " --compat_apiversion=%d" % compat_api_version

    compat_init_template_flags = ""
    for compat_init_template in compat_init_templates:
        compat_init_template_flags += (
            " --compat_init_template=$(location %s)" % compat_init_template
        )

    flags = [
        root_init_template_flag,
        "--apidir=$(@D)" + output_dir,
        "--apiname=" + api_name,
        "--apiversion=" + str(api_version),
        compat_api_version_flags,
        compat_init_template_flags,
        "--packages=" + ",".join(packages),
        "--output_package=" + output_package,
        "--use_relative_imports=True",
    ]

    # copybara:uncomment_begin(configurable API loading)
    # native.vardef("TF_API_INIT_LOADING", "default")
    # loading_value = "$(TF_API_INIT_LOADING)"
    # copybara:uncomment_end_and_comment_begin
    loading_value = "default"
    # copybara:comment_end

    native.genrule(
        name = name,
        outs = all_output_files,
        cmd = if_indexing_source_code(
            _make_cmd(api_gen_binary_target, flags, loading = "static"),
            _make_cmd(api_gen_binary_target, flags, loading = loading_value),
        ),
        srcs = srcs,
        tools = [":" + api_gen_binary_target],
        visibility = [
            "//tensorflow:__pkg__",
            "//tensorflow/tools/api/tests:__pkg__",
        ],
    )

def _make_cmd(api_gen_binary_target, flags, loading):
    binary = "$(location :" + api_gen_binary_target + ")"
    flags.append("--loading=" + loading)
    return " ".join([binary] + flags + ["$(OUTS)"])
