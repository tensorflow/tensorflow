"""Targets for generating TensorFlow Python API __init__.py files."""

load("//tensorflow:py.default.bzl", "py_binary")
load("//tensorflow:tensorflow.bzl", "if_oss")
load("//tensorflow:tensorflow.default.bzl", "if_indexing_source_code")
load("//tensorflow/python/tools/api/generator:api_init_files.bzl", "TENSORFLOW_API_INIT_FILES")

TENSORFLOW_API_GEN_PACKAGES = [
    "tensorflow.python",
    "tensorflow.compiler.mlir.quantization.tensorflow.python.quantize_model",
    "tensorflow.compiler.mlir.quantization.tensorflow.python.representative_dataset",
    "tensorflow.dtensor.python.accelerator_util",
    "tensorflow.dtensor.python.api",
    "tensorflow.dtensor.python.config",
    "tensorflow.dtensor.python.d_checkpoint",
    "tensorflow.dtensor.python.d_variable",
    "tensorflow.dtensor.python.input_util",
    "tensorflow.dtensor.python.layout",
    "tensorflow.dtensor.python.mesh_util",
    "tensorflow.dtensor.python.tpu_util",
    "tensorflow.dtensor.python.save_restore",
    "tensorflow.lite.python.analyzer",
    "tensorflow.lite.python.lite",
    "tensorflow.lite.python.authoring.authoring",
    "tensorflow.python.modules_with_exports",
]

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
        packages = TENSORFLOW_API_GEN_PACKAGES,
        package_deps = [
            "//tensorflow/python:no_contrib",
            "//tensorflow/python:modules_with_exports",
            "//tensorflow/lite/python:analyzer",
            "//tensorflow/lite/python:lite",
            "//tensorflow/lite/python/authoring",
        ],
        output_package = "tensorflow",
        output_dir = "",
        root_file_name = "__init__.py",
        proxy_module_root = None,
        api_packages_file_name = None,
        visibility = []):
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
        (e.g. "tensorflow").
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
      proxy_module_root: Module root for proxy-import format. If specified, proxy files with content
        like `from proxy_module_root.proxy_module import *` will be created to enable import
        resolution under TensorFlow.
      api_packages_file_name: Name of the file with the list of all API packages. Stores in output_dir.
      visibility: Visibility of the rule.
    """
    root_init_template_flag = ""
    if root_init_template:
        root_init_template_flag = "--root_init_template=" + root_init_template

    primary_package = packages[0]
    api_gen_binary_target = ("create_" + primary_package + "_api_%s") % name
    py_binary(
        name = api_gen_binary_target,
        srcs = ["//tensorflow/python/tools/api/generator:create_python_api.py"],
        main = "//tensorflow/python/tools/api/generator:create_python_api.py",
        python_version = "PY3",
        srcs_version = "PY3",
        visibility = ["//visibility:public"],
        deps = package_deps + [
            "//tensorflow/python/util:tf_decorator_py",
            "//tensorflow/python/util:tf_export",
            "//tensorflow/python/util:module_wrapper",
            "//tensorflow/python/tools/api/generator:doc_srcs",
        ],
    )
    if proxy_module_root != None:
        # Avoid conflicts between the __init__.py file of TensorFlow and proxy module.
        output_files = [f for f in output_files if f != "__init__.py"]

    # Replace name of root file with root_file_name.
    output_files = [
        root_file_name if f == "__init__.py" else f
        for f in output_files
    ]
    all_output_files = ["%s%s" % (output_dir, f) for f in output_files]
    compat_api_version_flags = ""
    for compat_api_version in compat_api_versions:
        compat_api_version_flags += " --compat_apiversion=%d" % compat_api_version

    if api_packages_file_name:
        api_packages_path = "%s%s" % (output_dir, api_packages_file_name)
    else:
        api_packages_path = None

    compat_init_template_flags = ""
    for compat_init_template in compat_init_templates:
        compat_init_template_flags += (
            " --compat_init_template=$(location %s)" % compat_init_template
        )

    flags = [
        root_init_template_flag,
        "--apidir=$(@D)/" + output_dir,
        "--apiname=" + api_name,
        "--apiversion=" + str(api_version),
        compat_api_version_flags,
        compat_init_template_flags,
        "--packages=" + ",".join(packages),
        "--output_package=" + output_package,
        "--use_relative_imports=True",
    ]
    if proxy_module_root != None:
        flags.append("--proxy_module_root=" + proxy_module_root)

    # copybara:uncomment_begin(configurable API loading)
    # native.vardef("TF_API_INIT_LOADING", "default")
    # loading_value = "$(TF_API_INIT_LOADING)"
    # copybara:uncomment_end_and_comment_begin
    loading_value = "default"
    # copybara:comment_end

    api_gen_rule(
        name = name,
        outs = all_output_files,
        srcs = srcs,
        flags = flags,
        api_gen_binary_target = ":" + api_gen_binary_target,
        api_packages_path = api_packages_path,
        loading_value = if_indexing_source_code("static", loading_value),
        visibility = visibility,
    )

def _get_module_by_path(dir_path, output_dir):
    """Get module that corresponds to the path.

    bazel-out/k8-opt/bin/tensorflow/_api/v2/compat/v2/compat/v2/compat/__init__.py
    to
    tensorflow._api.v2.compat.v2.compat.v2.compat

    Args:
    dir_path: Path to the directory.
    output_dir: Path to the directory.

    Returns:
    Name of module that corresponds to the given directory.
    """
    dir_path = dir_path.split(output_dir)[1]
    dir_path = dir_path.replace("__init__.py", "")

    #    dir_path = "tensorflow/%s" % dir_path
    return dir_path.replace("/", ".").strip(".")

def _api_gen_rule_impl(ctx):
    api_gen_binary_target = ctx.attr.api_gen_binary_target[DefaultInfo].files_to_run.executable
    flags = [ctx.expand_location(flag) for flag in ctx.attr.flags]
    variables = {"@D": ctx.genfiles_dir.path + "/" + ctx.label.package}
    flags = [ctx.expand_make_variables("tf_api_version", flag, variables) for flag in flags]
    loading = ctx.expand_make_variables("TF_API_INIT_LOADING", ctx.attr.loading_value, {})
    output_paths = [f.path for f in ctx.outputs.outs]

    # Generate file containing the list of outputs
    # Without this, the command will be too long (even when executed in a shell script)
    params = ctx.actions.declare_file(ctx.attr.name + ".params")
    ctx.actions.write(params, ";".join(output_paths))

    # Convert output_paths to the list of corresponding modules for the further testing
    if ctx.attr.api_packages_path:
        output_modules = sorted([
            _get_module_by_path(path, ctx.genfiles_dir.path)
            for path in output_paths
            if "__init__.py" in path
        ])
        api_packages = ctx.actions.declare_file(ctx.attr.api_packages_path.name)
        ctx.actions.write(api_packages, "\n".join(output_modules))

    cmd = _make_cmd(api_gen_binary_target, flags, loading, [params.path])
    ctx.actions.run_shell(
        inputs = ctx.files.srcs + [params],
        outputs = ctx.outputs.outs,
        tools = [api_gen_binary_target],
        use_default_shell_env = True,
        command = cmd,
    )

# Note: if only one output_paths is provided, api_gen_binary_target assumes it is a file to be read
def _make_cmd(api_gen_binary_target, flags, loading, output_paths):
    binary = api_gen_binary_target.path
    flags = flags + ["--loading=" + loading]
    return " ".join([binary] + flags + output_paths)

# To prevent compiling the C++ code twice, we only want to build `api_gen_binary_target`
# for the target platform and not the execution platform.
# To achieve this without causing confusion with source dependencies (e.g. putting api_gen_binary_target in srcs of the genrule),
# we use a custom rule to execute the command line for generating the API files.
# See https://github.com/tensorflow/tensorflow/issues/60167
# To not break internal cross-platform builds, we only set `cfg` to `target` for the OSS build.
api_gen_rule = rule(
    implementation = _api_gen_rule_impl,
    attrs = {
        "outs": attr.output_list(mandatory = True),
        "srcs": attr.label_list(allow_files = True),
        "flags": attr.string_list(),
        "api_gen_binary_target": attr.label(executable = True, cfg = if_oss("target", "exec"), mandatory = True),
        "loading_value": attr.string(mandatory = True),
        "api_packages_path": attr.output(),
    },
)
