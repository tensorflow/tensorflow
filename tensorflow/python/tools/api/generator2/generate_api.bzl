"""Rules to generate the TensorFlow public API from annotated files."""

load(":apis.bzl", _APIS = "APIS")
load(":patterns.bzl", "any_match")

APIS = _APIS.keys()

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
    output = ctx.actions.declare_file("_".join([
        ctx.label.name,
        "merged-api.json",
    ]))

    args = ctx.actions.args()
    args.set_param_file_format("multiline")
    args.use_param_file("--flagfile=%s")

    args.add("--output", output)
    inputs = depset(transitive = [
        dep[ApiInfo].transitive_api
        for dep in ctx.attr.deps
    ])
    args.add_all(
        inputs,
        expand_directories = True,
    )

    ctx.actions.run(
        mnemonic = "GenerateAPI",
        executable = ctx.executable._generator_bin,
        inputs = inputs,
        outputs = [output],
        arguments = [args],
        progress_message = "Generating APIs for %{label} to %{output}.",
    )

    return [
        DefaultInfo(files = depset([output])),  # TODO -- remove, for testing only
        _merge_py_info(ctx.attr.deps),  # TODO -- include generated files in direct_sources
    ]

generate_api = rule(
    doc = "Generate Python API for all targets in transitive dependencies.",
    implementation = _generate_api_impl,
    attrs = {
        "deps": attr.label_list(
            doc = "extract_api targets to generate API from.",
            allow_empty = False,
            providers = [ApiInfo, PyInfo],
            mandatory = True,
        ),
        # "root_init_template": attr.label(
        #     doc = "Template for the top level __init__.py file",
        #     allow_single_file = True,
        # ),
        # "api_version": attr.int(
        #     doc = "The API version to generate (1 or 2)",
        #     values = [1, 2],
        # ),
        # "compat_api_versions": attr.int_list(
        #     doc = "Additional versions to generate in compat/ subdirectory.",
        # ),
        # "compat_init_templates": attr.label_list(
        #     doc = "Template for top-level __init__files under compat modules. This list must be " +
        #           "in the same order as the list of versions in compat_apiversions",
        #     allow_files = True,
        # ),
        # "output_package": attr.string(
        #     doc = "Root output package.",
        # ),
        # "output_dir": attr.string(
        #     doc = "Subdirectory to output API to. If non-empty, must end with '/'.",
        # ),
        # "proxy_module_root": attr.string(
        #     doc = "Module root for proxy-import format. If specified, proxy files with " +
        #           "`from proxy_module_root.proxy_module import *` will be created to enable " +
        #           "import resolution under TensorFlow.",
        # ),
        # "output_files": attr.output_list(
        #     doc = "List of __init__.py files that should be generated. This list should include " +
        #           "file name for every module exported using tf_export. For e.g. if an op is " +
        #           "decorated with @tf_export('module1.module2', 'module3'). Then, output_files " +
        #           "should include module1/module2/__init__.py and module3/__init__.py.",
        # ),
        "_generator_bin": attr.label(
            default = Label("//tensorflow/python/tools/api/generator2/generator:generator"),
            executable = True,
            cfg = "exec",
        ),
    },
    provides = [PyInfo],
)

def generate_apis(
        *,
        name,
        apis,
        deps,
        **kwargs):
    """Generate TensorFlow APIs for a set of libraries.

    Args:
        name: name of generate_api target.
        apis: APIs to extract. See APIS constant for allowed values.
        deps: python_library targets to serve as roots for extracting APIs.
        **kwargs: additional arguments to pass to generate_api rule.
    """
    extract_api_targets = []
    for api in apis:
        extract_name = name + ".extract-" + api
        extract_api(
            name = extract_name,
            api = api,
            deps = deps,
            visibility = ["//visibility:private"],
            testonly = kwargs.get("testonly"),
        )
        extract_api_targets.append(extract_name)

    generate_api(
        name = name,
        deps = extract_api_targets,
        **kwargs
    )
