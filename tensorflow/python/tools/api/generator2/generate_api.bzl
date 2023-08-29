"""Rules to generate the TensorFlow public API from annotated files."""

load(":apis.bzl", _APIS = "APIS")
load(":patterns.bzl", "any_match")
load("//tensorflow/python/tools/api/generator:api_init_files.bzl", "TENSORFLOW_API_INIT_FILES")
load("@bazel_skylib//lib:paths.bzl", "paths")

APIS = _APIS.keys()

_MODULE_PREFIX = ""

def _api_info_init(*, transitive_api):
    if type(transitive_api) != type(depset()):
        fail("ApiInfo.transitive_api must be a depset")
    return {"transitive_api": transitive_api}

ApiInfo, _new_api_info = provider(
    doc = "Provider for API symbols and docstrings extracted from Python files.",
    fields = {
        "transitive_api": "depset of files with extracted API.",
    },
    init = _api_info_init,
)

def _py_files(f):
    if f.basename.endswith(".py") or f.basename.endswith(".py3"):
        return f.path
    return None

def _merge_py_info(
        deps,
        direct_sources = None,
        direct_imports = None,
        has_py2_only_sources = False,
        has_py3_only_sources = False,
        uses_shared_libraries = False):
    transitive_sources = []
    transitive_imports = []
    for dep in deps:
        if PyInfo in dep:
            transitive_sources.append(dep[PyInfo].transitive_sources)
            transitive_imports.append(dep[PyInfo].imports)
            has_py2_only_sources = has_py2_only_sources or dep[PyInfo].has_py2_only_sources
            has_py3_only_sources = has_py3_only_sources or dep[PyInfo].has_py3_only_sources
            uses_shared_libraries = uses_shared_libraries or dep[PyInfo].uses_shared_libraries

    return PyInfo(
        transitive_sources = depset(direct = direct_sources, transitive = transitive_sources),
        imports = depset(direct = direct_imports, transitive = transitive_imports),
        has_py2_only_sources = has_py2_only_sources,
        has_py3_only_sources = has_py3_only_sources,
        uses_shared_libraries = uses_shared_libraries,
    )

def _merge_api_info(
        deps,
        direct_api = None):
    transitive_api = []
    for dep in deps:
        if ApiInfo in dep:
            transitive_api.append(dep[ApiInfo].transitive_api)
    return ApiInfo(transitive_api = depset(direct = direct_api, transitive = transitive_api))

def _api_extractor_impl(target, ctx):
    api = ctx.attr.api
    config = _APIS[api]
    direct_api = []

    # Make sure the rule has a non-empty srcs attribute.
    if (
        any_match(config["target_patterns"], target.label) and
        hasattr(ctx.rule.attr, "srcs") and
        ctx.rule.attr.srcs
    ):
        output = ctx.actions.declare_file("_".join([
            target.label.name,
            "extracted",
            api,
            "api.json",
        ]))

        args = ctx.actions.args()
        args.set_param_file_format("multiline")
        args.use_param_file("--flagfile=%s")

        args.add("--output", output)
        args.add("--decorator", config["decorator"])
        args.add("--api_name", api)
        args.add_all(ctx.rule.files.srcs, expand_directories = True, map_each = _py_files)

        ctx.actions.run(
            mnemonic = "ExtractAPI",
            executable = ctx.executable._extractor_bin,
            inputs = ctx.rule.files.srcs,
            outputs = [output],
            arguments = [args],
            progress_message = "Extracting " + api + " APIs for %{label} to %{output}.",
        )

        direct_api.append(output)

    return [
        _merge_api_info(ctx.rule.attr.deps if hasattr(ctx.rule.attr, "deps") else [], direct_api = direct_api),
    ]

api_extractor = aspect(
    doc = "Extracts the exported API for the given target and its dependencies.",
    implementation = _api_extractor_impl,
    attr_aspects = ["deps"],
    provides = [ApiInfo],
    # Currently the Python rules do not correctly advertise their providers.
    # required_providers = [PyInfo],
    attrs = {
        "_extractor_bin": attr.label(
            default = Label("//tensorflow/python/tools/api/generator2/extractor:extractor"),
            executable = True,
            cfg = "exec",
        ),
        "api": attr.string(
            doc = "API to extract from dependencies.",
            mandatory = True,
            values = APIS,
        ),
    },
)

def _extract_api_impl(ctx):
    return [
        _merge_api_info(ctx.attr.deps),
        _merge_py_info(ctx.attr.deps),
    ]

extract_api = rule(
    doc = "Extract Python API for all targets in transitive dependencies.",
    implementation = _extract_api_impl,
    attrs = {
        "deps": attr.label_list(
            doc = "Targets to extract API from.",
            allow_empty = False,
            aspects = [api_extractor],
            providers = [PyInfo],
            mandatory = True,
        ),
        "api": attr.string(
            doc = "API to extract from dependencies.",
            mandatory = True,
            values = APIS,
        ),
    },
    provides = [ApiInfo, PyInfo],
)

def _generate_api_impl(ctx):
    args = ctx.actions.args()
    args.set_param_file_format("multiline")
    args.use_param_file("--flagfile=%s")

    args.add_joined("--output_files", ctx.outputs.output_files, join_with = ",")
    args.add("--output_dir", paths.join(ctx.bin_dir.path, ctx.label.package, ctx.attr.output_dir))
    if ctx.file.root_init_template:
        args.add("--root_init_template", ctx.file.root_init_template)
    args.add("--apiversion", ctx.attr.api_version)
    args.add_joined("--compat_api_versions", ctx.attr.compat_api_versions, join_with = ",")
    args.add_joined("--compat_init_templates", ctx.files.compat_init_templates, join_with = ",")
    args.add("--output_package", ctx.attr.output_package)
    args.add_joined("--packages_to_ignore", ctx.attr.packages_to_ignore, join_with = ",")
    if _MODULE_PREFIX:
        args.add("--module_prefix", _MODULE_PREFIX)
    if ctx.attr.use_lazy_loading:
        args.add("--use_lazy_loading")
    else:
        args.add("--nouse_lazy_loading")
    if ctx.attr.proxy_module_root:
        args.add("--proxy_module_root", ctx.attr.proxy_module_root)
    args.add_joined("--file_prefixes_to_strip", [ctx.bin_dir.path, ctx.genfiles_dir.path], join_with = ",")

    inputs = depset(transitive = [
        dep[ApiInfo].transitive_api
        for dep in ctx.attr.deps
    ])
    args.add_all(
        inputs,
        expand_directories = True,
    )

    transitive_inputs = [inputs]
    if ctx.attr.root_init_template:
        transitive_inputs.append(ctx.attr.root_init_template.files)

    ctx.actions.run(
        mnemonic = "GenerateAPI",
        executable = ctx.executable._generator_bin,
        inputs = depset(
            direct = ctx.files.compat_init_templates,
            transitive = transitive_inputs,
        ),
        outputs = ctx.outputs.output_files,
        arguments = [args],
        progress_message = "Generating APIs for %{label} to %{output}.",
    )

generate_api = rule(
    doc = "Generate Python API for all targets in transitive dependencies.",
    implementation = _generate_api_impl,
    attrs = {
        "deps": attr.label_list(
            doc = "extract_api targets to generate API from.",
            allow_empty = True,
            providers = [ApiInfo, PyInfo],
            mandatory = True,
        ),
        "root_init_template": attr.label(
            doc = "Template for the top level __init__.py file",
            allow_single_file = True,
        ),
        "api_version": attr.int(
            doc = "The API version to generate (1 or 2)",
            values = [1, 2],
        ),
        "compat_api_versions": attr.int_list(
            doc = "Additional versions to generate in compat/ subdirectory.",
        ),
        "compat_init_templates": attr.label_list(
            doc = "Template for top-level __init__files under compat modules. This list must be " +
                  "in the same order as the list of versions in compat_apiversions",
            allow_files = True,
        ),
        "output_package": attr.string(
            doc = "Root output package.",
        ),
        "output_dir": attr.string(
            doc = "Subdirectory to output API to. If non-empty, must end with '/'.",
        ),
        "proxy_module_root": attr.string(
            doc = "Module root for proxy-import format. If specified, proxy files with " +
                  "`from proxy_module_root.proxy_module import *` will be created to enable " +
                  "import resolution under TensorFlow.",
        ),
        "output_files": attr.output_list(
            doc = "List of __init__.py files that should be generated. This list should include " +
                  "file name for every module exported using tf_export. For e.g. if an op is " +
                  "decorated with @tf_export('module1.module2', 'module3'). Then, output_files " +
                  "should include module1/module2/__init__.py and module3/__init__.py.",
        ),
        "use_lazy_loading": attr.bool(
            doc = "If true, lazy load imports in the generated API rather then imporing them all statically.",
        ),
        "packages_to_ignore": attr.string_list(
            doc = "List of packages to ignore tf_exports from.",
        ),
        "_generator_bin": attr.label(
            default = Label("//tensorflow/python/tools/api/generator2/generator:main"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def generate_apis(
        name,
        apis = [],
        deps = [
            "//tensorflow/python:no_contrib",
            "//tensorflow/python:modules_with_exports",
            "//tensorflow/lite/python:analyzer",
            "//tensorflow/lite/python:lite",
            "//tensorflow/lite/python/authoring",
        ],
        output_files = TENSORFLOW_API_INIT_FILES,
        root_init_template = None,
        api_version = 2,
        compat_api_versions = [],
        compat_init_templates = [],
        output_package = "tensorflow",
        output_dir = "",
        proxy_module_root = None,
        packages_to_ignore = [],
        visibility = ["//visibility:private"]):
    """Generate TensorFlow APIs for a set of libraries.

    Args:
        name: name of generate_api target.
        apis: APIs to extract. See APIS constant for allowed values.
        deps: python_library targets to serve as roots for extracting APIs.
        output_files: The list of files that the API generator is exected to create.
        root_init_template: The template for the top level __init__.py file generated.
            "#API IMPORTS PLACEHOLDER" comment will be replaced with imports.
        api_version: THhe API version to generate. (1 or 2)
        compat_api_versions: Additional versions to generate in compat/ subdirectory.
        compat_init_templates: Template for top level __init__.py files under the compat modules.
            The list must be in the same order as the list of versions in 'compat_api_versions'
        output_package: Root output package.
        output_dir: Directory where the generated output files are placed. This should be a prefix
            of every directory in 'output_files'
        proxy_module_root: Module root for proxy-import format. If specified, proxy files with
            `from proxy_module_root.proxy_module import *` will be created to enable import
            resolution under TensorFlow.
        packages_to_ignore: List of packages to ignore tf_exports from.
        visibility: Visibility of the target containing the generated files.
    """
    extract_api_targets = []
    for api in apis:
        extract_name = name + ".extract-" + api
        extract_api(
            name = extract_name,
            api = api,
            deps = deps,
            visibility = ["//visibility:private"],
        )
        extract_api_targets.append(extract_name)

    if proxy_module_root != None:
        # Avoid conflicts between the __init__.py file of TensorFlow and proxy module.
        output_files = [f for f in output_files if f != "__init__.py"]

    all_output_files = [paths.join(output_dir, f) for f in output_files]

    generate_api(
        name = name,
        deps = extract_api_targets,
        output_files = all_output_files,
        output_dir = output_dir,
        root_init_template = root_init_template,
        compat_api_versions = compat_api_versions,
        compat_init_templates = compat_init_templates,
        api_version = api_version,
        proxy_module_root = proxy_module_root,
        visibility = visibility,
        packages_to_ignore = packages_to_ignore,
        # copybara:uncomment_begin(configurable API loading)
        # use_lazy_loading = select({
        # "//tensorflow/python/tools/api/generator2:static_gen": False,
        # "//tensorflow:api_indexable": False,
        # "//conditions:default": True,
        # }),
        # copybara:uncomment_end_and_comment_begin
        use_lazy_loading = False,
        # copybara:comment_end
        output_package = output_package,
    )
