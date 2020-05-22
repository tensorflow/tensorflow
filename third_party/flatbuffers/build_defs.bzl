"""BUILD rules for generating flatbuffer files."""

load("@build_bazel_rules_android//android:rules.bzl", "android_library")

flatc_path = "@flatbuffers//:flatc"
zip_files = "//tensorflow/lite/tools:zip_files"

DEFAULT_INCLUDE_PATHS = [
    "./",
    "$(GENDIR)",
    "$(BINDIR)",
]

DEFAULT_FLATC_ARGS = [
    "--no-union-value-namespacing",
    "--gen-object-api",
]

def flatbuffer_library_public(
        name,
        srcs,
        outs,
        language_flag,
        out_prefix = "",
        includes = [],
        include_paths = [],
        flatc_args = DEFAULT_FLATC_ARGS,
        reflection_name = "",
        reflection_visibility = None,
        output_to_bindir = False):
    """Generates code files for reading/writing the given flatbuffers in the requested language using the public compiler.

    Outs:
      filegroup(name): all generated source files.
      Fileset([reflection_name]): (Optional) all generated reflection binaries.

    Args:
      name: Rule name.
      srcs: Source .fbs files. Sent in order to the compiler.
      outs: Output files from flatc.
      language_flag: Target language flag. One of [-c, -j, -js].
      out_prefix: Prepend this path to the front of all generated files except on
          single source targets. Usually is a directory name.
      includes: Optional, list of filegroups of schemas that the srcs depend on.
      include_paths: Optional, list of paths the includes files can be found in.
      flatc_args: Optional, list of additional arguments to pass to flatc.
      reflection_name: Optional, if set this will generate the flatbuffer
        reflection binaries for the schemas.
      reflection_visibility: The visibility of the generated reflection Fileset.
      output_to_bindir: Passed to genrule for output to bin directory.
    """
    include_paths_cmd = ["-I %s" % (s) for s in include_paths]

    # '$(@D)' when given a single source target will give the appropriate
    # directory. Appending 'out_prefix' is only necessary when given a build
    # target with multiple sources.
    output_directory = (
        ("-o $(@D)/%s" % (out_prefix)) if len(srcs) > 1 else ("-o $(@D)")
    )
    genrule_cmd = " ".join([
        "for f in $(SRCS); do",
        "$(location %s)" % (flatc_path),
        " ".join(flatc_args),
        " ".join(include_paths_cmd),
        language_flag,
        output_directory,
        "$$f;",
        "done",
    ])
    native.genrule(
        name = name,
        srcs = srcs,
        outs = outs,
        output_to_bindir = output_to_bindir,
        tools = includes + [flatc_path],
        cmd = genrule_cmd,
        message = "Generating flatbuffer files for %s:" % (name),
    )
    if reflection_name:
        reflection_genrule_cmd = " ".join([
            "for f in $(SRCS); do",
            "$(location %s)" % (flatc_path),
            "-b --schema",
            " ".join(flatc_args),
            " ".join(include_paths_cmd),
            language_flag,
            output_directory,
            "$$f;",
            "done",
        ])
        reflection_outs = [
            (out_prefix + "%s.bfbs") % (s.replace(".fbs", "").split("/")[-1])
            for s in srcs
        ]
        native.genrule(
            name = "%s_srcs" % reflection_name,
            srcs = srcs,
            outs = reflection_outs,
            output_to_bindir = output_to_bindir,
            tools = includes + [flatc_path],
            cmd = reflection_genrule_cmd,
            message = "Generating flatbuffer reflection binary for %s:" % (name),
        )
        # TODO(b/114456773): Make bazel rules proper and supported by flatbuffer
        # Have to comment this since FilesetEntry is not supported in bazel
        # skylark.
        # native.Fileset(
        #     name = reflection_name,
        #     out = "%s_out" % reflection_name,
        #     entries = [
        #         native.FilesetEntry(files = reflection_outs),
        #     ],
        #     visibility = reflection_visibility,
        # )

def flatbuffer_cc_library(
        name,
        srcs,
        srcs_filegroup_name = "",
        out_prefix = "",
        includes = [],
        include_paths = [],
        flatc_args = DEFAULT_FLATC_ARGS,
        visibility = None,
        srcs_filegroup_visibility = None,
        gen_reflections = False):
    '''A cc_library with the generated reader/writers for the given flatbuffer definitions.

    Outs:
      filegroup([name]_srcs): all generated .h files.
      filegroup(srcs_filegroup_name if specified, or [name]_includes if not):
          Other flatbuffer_cc_library's can pass this in for their `includes`
          parameter, if they depend on the schemas in this library.
      Fileset([name]_reflection): (Optional) all generated reflection binaries.
      cc_library([name]): library with sources and flatbuffers deps.

    Remarks:
      ** Because the genrule used to call flatc does not have any trivial way of
        computing the output list of files transitively generated by includes and
        --gen-includes (the default) being defined for flatc, the --gen-includes
        flag will not work as expected. The way around this is to add a dependency
        to the flatbuffer_cc_library defined alongside the flatc included Fileset.
        For example you might define:

        flatbuffer_cc_library(
            name = "my_fbs",
            srcs = [ "schemas/foo.fbs" ],
            includes = [ "//third_party/bazz:bazz_fbs_includes" ],
        )

        In which foo.fbs includes a few files from the Fileset defined at
        //third_party/bazz:bazz_fbs_includes. When compiling the library that
        includes foo_generated.h, and therefore has my_fbs as a dependency, it
        will fail to find any of the bazz *_generated.h files unless you also
        add bazz's flatbuffer_cc_library to your own dependency list, e.g.:

        cc_library(
            name = "my_lib",
            deps = [
                ":my_fbs",
                "//third_party/bazz:bazz_fbs"
            ],
        )

        Happy dependent Flatbuffering!

    Args:
      name: Rule name.
      srcs: Source .fbs files. Sent in order to the compiler.
      srcs_filegroup_name: Name of the output filegroup that holds srcs. Pass this
          filegroup into the `includes` parameter of any other
          flatbuffer_cc_library that depends on this one's schemas.
      out_prefix: Prepend this path to the front of all generated files. Usually
          is a directory name.
      includes: Optional, list of filegroups of schemas that the srcs depend on.
          ** SEE REMARKS BELOW **
      include_paths: Optional, list of paths the includes files can be found in.
      flatc_args: Optional list of additional arguments to pass to flatc
          (e.g. --gen-mutable).
      visibility: The visibility of the generated cc_library. By default, use the
          default visibility of the project.
      srcs_filegroup_visibility: The visibility of the generated srcs filegroup.
          By default, use the value of the visibility parameter above.
      gen_reflections: Optional, if true this will generate the flatbuffer
        reflection binaries for the schemas.
    '''
    output_headers = [
        (out_prefix + "%s_generated.h") % (s.replace(".fbs", "").split("/")[-1])
        for s in srcs
    ]
    reflection_name = "%s_reflection" % name if gen_reflections else ""

    flatbuffer_library_public(
        name = "%s_srcs" % (name),
        srcs = srcs,
        outs = output_headers,
        language_flag = "-c",
        out_prefix = out_prefix,
        includes = includes,
        include_paths = include_paths,
        flatc_args = flatc_args,
        reflection_name = reflection_name,
        reflection_visibility = visibility,
    )
    native.cc_library(
        name = name,
        hdrs = output_headers,
        srcs = output_headers,
        features = [
            "-parse_headers",
        ],
        deps = [
            "@flatbuffers//:runtime_cc",
        ],
        includes = ["."],
        linkstatic = 1,
        visibility = visibility,
    )

    # A filegroup for the `srcs`. That is, all the schema files for this
    # Flatbuffer set.
    native.filegroup(
        name = srcs_filegroup_name if srcs_filegroup_name else "%s_includes" % (name),
        srcs = srcs,
        visibility = srcs_filegroup_visibility if srcs_filegroup_visibility != None else visibility,
    )

# Custom provider to track dependencies transitively.
FlatbufferInfo = provider(
    fields = {
        "transitive_srcs": "flatbuffer schema definitions.",
    },
)

def _flatbuffer_schemas_aspect_impl(target, ctx):
    _ignore = [target]
    transitive_srcs = depset()
    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if FlatbufferInfo in dep:
                transitive_srcs = depset(dep[FlatbufferInfo].transitive_srcs, transitive = [transitive_srcs])
    if hasattr(ctx.rule.attr, "srcs"):
        for src in ctx.rule.attr.srcs:
            if FlatbufferInfo in src:
                transitive_srcs = depset(src[FlatbufferInfo].transitive_srcs, transitive = [transitive_srcs])
            for f in src.files:
                if f.extension == "fbs":
                    transitive_srcs = depset([f], transitive = [transitive_srcs])
    return [FlatbufferInfo(transitive_srcs = transitive_srcs)]

# An aspect that runs over all dependencies and transitively collects
# flatbuffer schema files.
_flatbuffer_schemas_aspect = aspect(
    attr_aspects = [
        "deps",
        "srcs",
    ],
    implementation = _flatbuffer_schemas_aspect_impl,
)

# Rule to invoke the flatbuffer compiler.
def _gen_flatbuffer_srcs_impl(ctx):
    outputs = ctx.attr.outputs
    include_paths = ctx.attr.include_paths
    if ctx.attr.no_includes:
        no_includes_statement = ["--no-includes"]
    else:
        no_includes_statement = []

    # Need to generate all files in a directory.
    if not outputs:
        outputs = [ctx.actions.declare_directory("{}_all".format(ctx.attr.name))]
        output_directory = outputs[0].path
    else:
        outputs = [ctx.actions.declare_file(output) for output in outputs]
        output_directory = outputs[0].dirname

    deps = depset(ctx.files.srcs + ctx.files.deps, transitive = [
        dep[FlatbufferInfo].transitive_srcs
        for dep in ctx.attr.deps
        if FlatbufferInfo in dep
    ])

    include_paths_cmd_line = []
    for s in include_paths:
        include_paths_cmd_line.extend(["-I", s])

    for src in ctx.files.srcs:
        ctx.actions.run(
            inputs = deps,
            outputs = outputs,
            executable = ctx.executable._flatc,
            arguments = [
                            ctx.attr.language_flag,
                            "-o",
                            output_directory,
                            # Allow for absolute imports and referencing of generated files.
                            "-I",
                            "./",
                            "-I",
                            ctx.genfiles_dir.path,
                            "-I",
                            ctx.bin_dir.path,
                        ] + no_includes_statement +
                        include_paths_cmd_line + [
                "--no-union-value-namespacing",
                "--gen-object-api",
                src.path,
            ],
            progress_message = "Generating flatbuffer files for {}:".format(src),
        )
    return [
        DefaultInfo(files = depset(outputs)),
    ]

_gen_flatbuffer_srcs = rule(
    _gen_flatbuffer_srcs_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".fbs"],
            mandatory = True,
        ),
        "outputs": attr.string_list(
            default = [],
            mandatory = False,
        ),
        "deps": attr.label_list(
            default = [],
            mandatory = False,
            aspects = [_flatbuffer_schemas_aspect],
        ),
        "include_paths": attr.string_list(
            default = [],
            mandatory = False,
        ),
        "language_flag": attr.string(
            mandatory = True,
        ),
        "no_includes": attr.bool(
            default = False,
            mandatory = False,
        ),
        "_flatc": attr.label(
            default = Label("@flatbuffers//:flatc"),
            executable = True,
            cfg = "host",
        ),
    },
    output_to_genfiles = True,
)

def _concat_flatbuffer_py_srcs_impl(ctx):
    # Merge all generated python files. The files are concatenated and the
    # import statements are removed. Finally we import the flatbuffer runtime
    # library.
    ctx.actions.run_shell(
        inputs = ctx.attr.deps[0].files,
        outputs = [ctx.outputs.out],
        command = (
            "find '%s' -name '*.py' -exec cat {} + |" +
            "sed '/import flatbuffers/d' |" +
            "sed 's/from flatbuffers." +
            "/from flatbuffers.python.flatbuffers./' |" +
            "sed '1s/^/from flatbuffers.python " +
            "import flatbuffers\\n/' > %s"
        ) % (
            ctx.attr.deps[0].files.to_list()[0].path,
            ctx.outputs.out.path,
        ),
    )

_concat_flatbuffer_py_srcs = rule(
    _concat_flatbuffer_py_srcs_impl,
    attrs = {
        "deps": attr.label_list(mandatory = True),
    },
    output_to_genfiles = True,
    outputs = {"out": "%{name}.py"},
)

def flatbuffer_py_library(
        name,
        srcs,
        deps = [],
        include_paths = []):
    """A py_library with the generated reader/writers for the given schema.

    This rule assumes that the schema files define non-conflicting names, so that
    they can be merged in a single file. This is e.g. the case if only a single
    namespace is used.
    The rule call the flatbuffer compiler for all schema files and merges the
    generated python files into a single file that is wrapped in a py_library.

    Args:
      name: Rule name. (required)
      srcs: List of source .fbs files. (required)
      deps: List of dependencies.
      include_paths: Optional, list of paths the includes files can be found in.
    """
    all_srcs = "{}_srcs".format(name)
    _gen_flatbuffer_srcs(
        name = all_srcs,
        srcs = srcs,
        language_flag = "--python",
        deps = deps,
        include_paths = include_paths,
    )
    all_srcs_no_include = "{}_srcs_no_include".format(name)
    _gen_flatbuffer_srcs(
        name = all_srcs_no_include,
        srcs = srcs,
        language_flag = "--python",
        deps = deps,
        no_includes = True,
        include_paths = include_paths,
    )
    concat_py_srcs = "{}_generated".format(name)
    _concat_flatbuffer_py_srcs(
        name = concat_py_srcs,
        deps = [
            ":{}".format(all_srcs_no_include),
        ],
    )
    native.py_library(
        name = name,
        srcs = [
            ":{}".format(concat_py_srcs),
        ],
        srcs_version = "PY2AND3",
        deps = deps + [
            "@flatbuffers//:runtime_py",
        ],
    )

def flatbuffer_java_library(
        name,
        srcs,
        custom_package = "",
        package_prefix = "",
        include_paths = DEFAULT_INCLUDE_PATHS,
        flatc_args = DEFAULT_FLATC_ARGS,
        visibility = None):
    """A java library with the generated reader/writers for the given flatbuffer definitions.

    Args:
      name: Rule name. (required)
      srcs: List of source .fbs files including all includes. (required)
      custom_package: Package name of generated Java files. If not specified
          namespace in the schema files will be used. (optional)
      package_prefix: like custom_package, but prefixes to the existing
          namespace. (optional)
      include_paths: List of paths that includes files can be found in. (optional)
      flatc_args: List of additional arguments to pass to flatc. (optional)
      visibility: Visibility setting for the java_library rule. (optional)
    """
    out_srcjar = "java_%s_all.srcjar" % name
    flatbuffer_java_srcjar(
        name = "%s_srcjar" % name,
        srcs = srcs,
        out = out_srcjar,
        custom_package = custom_package,
        flatc_args = flatc_args,
        include_paths = include_paths,
        package_prefix = package_prefix,
    )

    native.filegroup(
        name = "%s.srcjar" % name,
        srcs = [out_srcjar],
    )

    native.java_library(
        name = name,
        srcs = [out_srcjar],
        javacopts = ["-source 7 -target 7"],
        deps = [
            "@flatbuffers//:runtime_java",
        ],
        visibility = visibility,
    )

def flatbuffer_java_srcjar(
        name,
        srcs,
        out,
        custom_package = "",
        package_prefix = "",
        include_paths = DEFAULT_INCLUDE_PATHS,
        flatc_args = DEFAULT_FLATC_ARGS):
    """Generate flatbuffer Java source files.

    Args:
      name: Rule name. (required)
      srcs: List of source .fbs files including all includes. (required)
      out: Output file name. (required)
      custom_package: Package name of generated Java files. If not specified
          namespace in the schema files will be used. (optional)
      package_prefix: like custom_package, but prefixes to the existing
          namespace. (optional)
      include_paths: List of paths that includes files can be found in. (optional)
      flatc_args: List of additional arguments to pass to flatc. (optional)
    """
    command_fmt = """set -e
      tmpdir=$(@D)
      schemas=$$tmpdir/schemas
      java_root=$$tmpdir/java
      rm -rf $$schemas
      rm -rf $$java_root
      mkdir -p $$schemas
      mkdir -p $$java_root

      for src in $(SRCS); do
        dest=$$schemas/$$src
        rm -rf $$(dirname $$dest)
        mkdir -p $$(dirname $$dest)
        if [ -z "{custom_package}" ] && [ -z "{package_prefix}" ]; then
          cp -f $$src $$dest
        else
          if [ -z "{package_prefix}" ]; then
            sed -e "s/namespace\\s.*/namespace {custom_package};/" $$src > $$dest
          else
            sed -e "s/namespace \\([^;]\\+\\);/namespace {package_prefix}.\\1;/" $$src > $$dest
          fi
        fi
      done

      flatc_arg_I="-I $$tmpdir/schemas"
      for include_path in {include_paths}; do
        flatc_arg_I="$$flatc_arg_I -I $$schemas/$$include_path"
      done

      flatc_additional_args=
      for arg in {flatc_args}; do
        flatc_additional_args="$$flatc_additional_args $$arg"
      done

      for src in $(SRCS); do
        $(location {flatc_path}) $$flatc_arg_I --java $$flatc_additional_args -o $$java_root  $$schemas/$$src
      done

      $(location {zip_files}) -export_zip_path=$@ -file_directory=$$java_root
      """
    genrule_cmd = command_fmt.format(
        package_name = native.package_name(),
        custom_package = custom_package,
        package_prefix = package_prefix,
        flatc_path = flatc_path,
        zip_files = zip_files,
        include_paths = " ".join(include_paths),
        flatc_args = " ".join(flatc_args),
    )

    native.genrule(
        name = name,
        srcs = srcs,
        outs = [out],
        tools = [flatc_path, zip_files],
        cmd = genrule_cmd,
    )

def flatbuffer_android_library(
        name,
        srcs,
        custom_package = "",
        package_prefix = "",
        include_paths = DEFAULT_INCLUDE_PATHS,
        flatc_args = DEFAULT_FLATC_ARGS,
        visibility = None):
    """An android_library with the generated reader/writers for the given flatbuffer definitions.

    Args:
      name: Rule name. (required)
      srcs: List of source .fbs files including all includes. (required)
      custom_package: Package name of generated Java files. If not specified
          namespace in the schema files will be used. (optional)
      package_prefix: like custom_package, but prefixes to the existing
          namespace. (optional)
      include_paths: List of paths that includes files can be found in. (optional)
      flatc_args: List of additional arguments to pass to flatc. (optional)
      visibility: Visibility setting for the android_library rule. (optional)
    """
    out_srcjar = "android_%s_all.srcjar" % name
    flatbuffer_java_srcjar(
        name = "%s_srcjar" % name,
        srcs = srcs,
        out = out_srcjar,
        custom_package = custom_package,
        flatc_args = flatc_args,
        include_paths = include_paths,
        package_prefix = package_prefix,
    )

    native.filegroup(
        name = "%s.srcjar" % name,
        srcs = [out_srcjar],
    )

    # To support org.checkerframework.dataflow.qual.Pure.
    checkerframework_annotations = [
        "@org_checkerframework_qual",
    ] if "--java-checkerframework" in flatc_args else []

    android_library(
        name = name,
        srcs = [out_srcjar],
        javacopts = ["-source 7 -target 7"],
        visibility = visibility,
        deps = [
            "@flatbuffers//:runtime_android",
        ] + checkerframework_annotations,
    )
