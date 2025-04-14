"""
Repository rule to manage hermetic Python interpreter under Bazel.

Version can be set via build parameter "--repo_env=HERMETIC_PYTHON_VERSION=3.11"

To set wheel name, add "--repo_env=WHEEL_NAME=tensorflow_cpu"
"""

DEFAULT_VERSION = "3.11"

def _python_repository_impl(ctx):
    version, py_kind = _get_python_version(ctx)
    version_and_kind = "%s-%s" % (version, py_kind) if py_kind else version

    ctx.file("BUILD", "")
    wheel_name = ctx.os.environ.get("WHEEL_NAME", "tensorflow")
    wheel_collab = ctx.os.environ.get("WHEEL_COLLAB", False)
    macos_deployment_target = ctx.os.environ.get("MACOSX_DEPLOYMENT_TARGET", "")
    hermetic_url = ctx.os.environ.get("HERMETIC_PYTHON_URL", "")
    hermetic_sha256 = ctx.os.environ.get("HERMETIC_PYTHON_SHA256", "")
    hermetic_prefix = ctx.os.environ.get("HERMETIC_PYTHON_PREFIX", "python")
    custom_requirements = ctx.os.environ.get("HERMETIC_REQUIREMENTS_LOCK", None)

    if not (hermetic_url + hermetic_sha256) and (hermetic_url or hermetic_sha256):
        fail("""
Please either specify both HERMETIC_PYTHON_URL and HERMETIC_PYTHON_SHA256
to set up a custom python interpreter, or none of them to rely on default ones.
""")
    requirements = None
    if not requirements:
        for i in range(0, len(ctx.attr.requirements_locks)):
            if ctx.attr.requirements_versions[i] == version_and_kind:
                requirements = ctx.attr.requirements_locks[i]
                break

    if not requirements and not custom_requirements:
        fail("""
Could not find requirements_lock.txt file matching specified Python version.
Specified python version: {version}
Python versions with available requirement_lock.txt files: {versions}
Please check python_init_repositories() in your WORKSPACE file.
""".format(
            version = version,
            versions = ", ".join(ctx.attr.requirements_versions),
        ))

    if custom_requirements:
        custom_requirements_path = ctx.path(custom_requirements)
        requirements_with_local_wheels = "@{repo}//:{label}".format(
            repo = ctx.name,
            label = custom_requirements_path.basename,
        )
        ctx.file(
            custom_requirements_path.basename,
            ctx.read(custom_requirements_path),
        )
    elif ctx.attr.local_wheel_workspaces:
        local_wheel_requirements = _get_injected_local_wheels(
            ctx,
            version,
            ctx.attr.local_wheel_workspaces,
        )
        requirements_content = [ctx.read(requirements)] + local_wheel_requirements
        merged_requirements_content = "\n".join(requirements_content)

        requirements_with_local_wheels = "@{repo}//:{label}".format(
            repo = ctx.name,
            label = requirements.name,
        )
        ctx.file(
            requirements.name,
            merged_requirements_content,
        )
    else:
        requirements_with_local_wheels = str(requirements)

    use_pywrap_rules = bool(
        ctx.os.environ.get("USE_PYWRAP_RULES", False),
    )

    if use_pywrap_rules:
        print("!!!Using pywrap rules instead of directly creating .so objects!!!")  # buildifier: disable=print

    interpreter_type = "\"default\" (provided by rules_python)"
    if hermetic_url:
        interpreter_type = "\"custom\" (pulled from %s)" % hermetic_url
    print(
        """
=============================
Hermetic Python configuration:
Version: "{version}"
Kind: "{py_kind}"
Interpreter: {interpreter_type}
Requirements_lock label: "{requirements_lock_label}"
=====================================
""".format(
            version = version,
            py_kind = py_kind,
            interpreter_type = interpreter_type,
            requirements_lock_label = requirements_with_local_wheels,
        ),
    )  # buildifier: disable=print

    ctx.file(
        "py_version.bzl",
        """
TF_PYTHON_VERSION = "{version}"
HERMETIC_PYTHON_VERSION = "{version}"
HERMETIC_PYTHON_VERSION_KIND = "{py_kind}"
WHEEL_NAME = "{wheel_name}"
WHEEL_COLLAB = "{wheel_collab}"
REQUIREMENTS = "{requirements}"
REQUIREMENTS_WITH_LOCAL_WHEELS = "{requirements_with_local_wheels}"
USE_PYWRAP_RULES = {use_pywrap_rules}
MACOSX_DEPLOYMENT_TARGET = "{macos_deployment_target}"
HERMETIC_PYTHON_URL = "{hermetic_url}"
HERMETIC_PYTHON_SHA256 = "{hermetic_sha256}"
HERMETIC_PYTHON_PREFIX = "{hermetic_prefix}"
""".format(
            version = version,
            py_kind = py_kind,
            wheel_name = wheel_name,
            wheel_collab = wheel_collab,
            requirements = str(requirements),
            requirements_with_local_wheels = requirements_with_local_wheels,
            use_pywrap_rules = use_pywrap_rules,
            macos_deployment_target = macos_deployment_target,
            hermetic_url = hermetic_url,
            hermetic_sha256 = hermetic_sha256,
            hermetic_prefix = hermetic_prefix,
        ),
    )

def _get_python_version(ctx):
    print_warning = False

    version = ctx.os.environ.get("HERMETIC_PYTHON_VERSION", "")
    if not version:
        version = ctx.os.environ.get("TF_PYTHON_VERSION", "")
    if not version:
        print_warning = True
        if ctx.attr.default_python_version == "system":
            python_version_result = ctx.execute(["python3", "--version"])
            if python_version_result.return_code == 0:
                version = python_version_result.stdout
            else:
                fail("""
Cannot match hermetic Python version to system Python version.
System Python was not found.""")
        else:
            version = ctx.attr.default_python_version

    version, kind = _parse_python_version(version)

    if print_warning:
        print("""
HERMETIC_PYTHON_VERSION variable was not set correctly, using default version.
Python {} will be used.
To select Python version, either set HERMETIC_PYTHON_VERSION env variable in
your shell:
  export HERMETIC_PYTHON_VERSION=3.12
OR pass it as an argument to bazel command directly or inside your .bazelrc
file:
  --repo_env=HERMETIC_PYTHON_VERSION=3.12
""".format(version))  # buildifier: disable=print
    return version, kind

def _parse_python_version(version_str):
    if version_str.startswith("Python "):
        py_ver_chunks = version_str[7:].split(".")
        return "%s.%s" % (py_ver_chunks[0], py_ver_chunks[1]), ""
    elif "-" in version_str:
        return version_str.split("-")
    return version_str, ""

def _get_injected_local_wheels(
        ctx,
        py_version,
        local_wheel_workspaces):
    local_wheel_requirements = []
    py_ver_marker = "-cp%s-" % py_version.replace(".", "")
    py_major_ver_marker = "-py%s-" % py_version.split(".")[0]
    wheels = {}

    if local_wheel_workspaces:
        for local_wheel_workspace in local_wheel_workspaces:
            local_wheel_workspace_path = ctx.path(local_wheel_workspace)
            dist_folder = ctx.attr.local_wheel_dist_folder
            dist_folder_path = local_wheel_workspace_path.dirname.get_child(dist_folder)
            if dist_folder_path.exists:
                dist_wheels = dist_folder_path.readdir()
                _process_dist_wheels(
                    dist_wheels,
                    wheels,
                    py_ver_marker,
                    py_major_ver_marker,
                    ctx.attr.local_wheel_inclusion_list,
                    ctx.attr.local_wheel_exclusion_list,
                )

    for wheel_name, wheel_path in wheels.items():
        local_wheel_requirements.append(
            "{wheel_name} @ file://{wheel_path}".format(
                wheel_name = wheel_name,
                wheel_path = wheel_path.realpath,
            ),
        )

    return local_wheel_requirements

python_repository = repository_rule(
    implementation = _python_repository_impl,
    attrs = {
        "requirements_versions": attr.string_list(
            mandatory = False,
            default = [],
        ),
        "requirements_locks": attr.label_list(
            mandatory = False,
            default = [],
        ),
        "local_wheel_workspaces": attr.label_list(
            mandatory = False,
            default = [],
        ),
        "local_wheel_dist_folder": attr.string(
            mandatory = False,
            default = "dist",
        ),
        "default_python_version": attr.string(
            mandatory = False,
            default = DEFAULT_VERSION,
        ),
        "local_wheel_inclusion_list": attr.string_list(
            mandatory = False,
            default = ["*"],
        ),
        "local_wheel_exclusion_list": attr.string_list(
            mandatory = False,
            default = [],
        ),
    },
    environ = [
        "TF_PYTHON_VERSION",
        "HERMETIC_PYTHON_VERSION",
        "HERMETIC_PYTHON_URL",
        "HERMETIC_PYTHON_SHA256",
        "HERMETIC_REQUIREMENTS_LOCK",
        "HERMETIC_PYTHON_PREFIX",
        "WHEEL_NAME",
        "WHEEL_COLLAB",
        "USE_PYWRAP_RULES",
        "MACOSX_DEPLOYMENT_TARGET",
    ],
    local = True,
)

def _process_dist_wheels(
        dist_wheels,
        wheels,
        py_ver_marker,
        py_major_ver_marker,
        local_wheel_inclusion_list,
        local_wheel_exclusion_list):
    for wheel in dist_wheels:
        bn = wheel.basename
        if not bn.endswith(".whl") or (bn.find(py_ver_marker) < 0 and bn.find(py_major_ver_marker) < 0):
            continue
        if not _basic_wildcard_match(bn, local_wheel_inclusion_list, True, False):
            continue
        if not _basic_wildcard_match(bn, local_wheel_exclusion_list, False, True):
            continue

        name_components = bn.split("-")
        package_name = name_components[0]
        for name_component in name_components[1:]:
            if name_component[0].isdigit():
                break
            package_name += "-" + name_component

        latest_wheel = wheels.get(package_name, None)

        if not latest_wheel or latest_wheel.basename < wheel.basename:
            wheels[package_name] = wheel

def _basic_wildcard_match(name, patterns, expected_match_result, match_all):
    match = False
    for pattern in patterns:
        match = False
        if pattern.startswith("*") and pattern.endswith("*"):
            match = name.find(pattern[1:-1]) >= 0
        elif pattern.startswith("*"):
            match = name.endswith(pattern[1:])
        elif pattern.endswith("*"):
            match = name.startswith(pattern[:-1])
        else:
            match = name == pattern

        if match_all:
            if match != expected_match_result:
                return False
        elif match == expected_match_result:
            return True

    return match == expected_match_result

def _custom_python_interpreter_impl(ctx):
    version = ctx.attr.version
    version_variant = ctx.attr.version_variant
    strip_prefix = ctx.attr.strip_prefix.format(
        version = version,
        version_variant = version_variant,
    )
    urls = [url.format(version = version, version_variant = version_variant) for url in ctx.attr.urls]
    binary_name = ctx.attr.binary_name
    if not binary_name:
        ver_chunks = version.split(".")
        binary_name = "python%s.%s" % (ver_chunks[0], ver_chunks[1])

    install_dir = "{name}-{version}".format(name = ctx.attr.name, version = version)
    _exec_and_check(ctx, ["mkdir", install_dir])
    install_path = ctx.path(install_dir)
    srcs_dir = "srcs"
    ctx.download_and_extract(
        url = urls,
        stripPrefix = strip_prefix,
        output = srcs_dir,
    )

    configure_params = list(ctx.attr.configure_params)
    if "CC" in ctx.os.environ:
        configure_params.append("CC={}".format(ctx.os.environ["CC"]))
    if "CXX" in ctx.os.environ:
        configure_params.append("CXX={}".format(ctx.os.environ["CXX"]))

    configure_params.append("--prefix=%s" % install_path.realpath)
    _exec_and_check(
        ctx,
        ["./configure"] + configure_params,
        working_directory = srcs_dir,
        quiet = False,
    )
    res = _exec_and_check(ctx, ["nproc"])
    cores = 12 if res.return_code != 0 else max(1, int(res.stdout.strip()) - 1)
    _exec_and_check(ctx, ["make", "-j%s" % cores], working_directory = srcs_dir)
    _exec_and_check(ctx, ["make", "altinstall"], working_directory = srcs_dir)
    _exec_and_check(ctx, ["ln", "-s", binary_name, "python3"], working_directory = install_dir + "/bin")
    tar = "{install_dir}.tgz".format(install_dir = install_dir)
    _exec_and_check(ctx, ["tar", "czpf", tar, install_dir])
    _exec_and_check(ctx, ["rm", "-rf", srcs_dir])
    res = _exec_and_check(ctx, ["sha256sum", tar])

    sha256 = res.stdout.split(" ")[0].strip()
    tar_path = ctx.path(tar)

    example = """\n\n
To use newly built Python interpreter add the following code snippet RIGHT AFTER
python_init_toolchains() in your WORKSPACE file. The code sample should work as
is but it may need some tuning, if you have special requirements.

```
load("@rules_python//python:repositories.bzl", "python_register_toolchains")
python_register_toolchains(
    name = "python",
    # By default assume the interpreter is on the local file system, replace
    # with proper URL if it is not the case.
    base_url = "file://",
    ignore_root_user_error = True,
    python_version = "{version}",
    tool_versions = {{
        "{version}": {{
            # Path to .tar.gz with Python binary. By default it points to .tgz
            # file in cache where it was built originally; replace with proper
            # file location, if you moved it somewhere else.
            "url": "{tar_path}",
            "sha256": {{
                # By default we assume Linux x86_64 architecture, eplace with
                # proper architecture if you were building on a different platform.
                "x86_64-unknown-linux-gnu": "{sha256}",
            }},
            "strip_prefix": "{install_dir}",
        }},
    }},
)
```
\n\n""".format(version = version, tar_path = tar_path, sha256 = sha256, install_dir = install_dir)

    instructions = "INSTRUCTIONS-{version}.md".format(version = version)
    ctx.file(instructions + ".tmpl", example, executable = False)
    ctx.file(
        "BUILD.bazel",
        """
genrule(
    name = "{name}",
    srcs = ["{tar}", "{instructions}.tmpl"],
    outs = ["{install_dir}.tar.gz", "{instructions}"],
    cmd = "cp $(location {tar}) $(location {install_dir}.tar.gz); cp $(location {instructions}.tmpl) $(location {instructions})",
    visibility = ["//visibility:public"],
)
     """.format(
            name = ctx.attr.name,
            tar = tar,
            install_dir = install_dir,
            instructions = instructions,
        ),
        executable = False,
    )

    print(example)  # buildifier: disable=print

custom_python_interpreter = repository_rule(
    implementation = _custom_python_interpreter_impl,
    attrs = {
        "urls": attr.string_list(),
        "strip_prefix": attr.string(),
        "binary_name": attr.string(mandatory = False),
        "version": attr.string(),
        "version_variant": attr.string(),
        "configure_params": attr.string_list(
            mandatory = False,
            default = ["--enable-optimizations"],
        ),
    },
)

def _exec_and_check(ctx, command, fail_on_error = True, quiet = False, **kwargs):
    res = ctx.execute(command, quiet = quiet, **kwargs)
    if fail_on_error and res.return_code != 0:
        fail("""
Failed to execute command: `{command}`
Exit Code: {code}
STDERR: {stderr}
        """.format(
            command = command,
            code = res.return_code,
            stderr = res.stderr,
        ))
    return res
