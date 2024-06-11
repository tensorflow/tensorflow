"""
Repository rule to manage hermetic Python interpreter under Bazel.

Version can be set via build parameter "--repo_env=HERMETIC_PYTHON_VERSION=3.11"
Defaults to 3.11.

To set wheel name, add "--repo_env=WHEEL_NAME=tensorflow_cpu"
"""

DEFAULT_VERSION = "3.11"

def _python_repository_impl(ctx):
    version = _get_python_version(ctx)

    ctx.file("BUILD", "")
    wheel_name = ctx.os.environ.get("WHEEL_NAME", "tensorflow")
    wheel_collab = ctx.os.environ.get("WHEEL_COLLAB", False)

    requirements = None
    for i in range(0, len(ctx.attr.requirements_locks)):
        if ctx.attr.requirements_versions[i] == version:
            requirements = ctx.attr.requirements_locks[i]
            break

    if not requirements:
        fail("""
Could not find requirements_lock.txt file matching specified Python version.
Specified python version: {version}
Python versions with available requirement_lock.txt files: {versions}
Please check python_init_repositories() in your WORKSPACE file.
""".format(
            version = version,
            versions = ", ".join(ctx.attr.requirements_versions),
        ))

    requirements_with_local_wheels = str(requirements)

    local_wheels_dir = ctx.os.environ.get("LOCAL_WHEELS_DIR", "")
    if ctx.attr.local_wheel_workspaces or local_wheels_dir:
        local_wheel_requirements = _get_injected_local_wheels(
            ctx,
            version,
            ctx.attr.local_wheel_workspaces,
            local_wheels_dir,
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

    ctx.file(
        "py_version.bzl",
        """
TF_PYTHON_VERSION = "{version}"
HERMETIC_PYTHON_VERSION = "{version}"
WHEEL_NAME = "{wheel_name}"
WHEEL_COLLAB = "{wheel_collab}"
REQUIREMENTS = "{requirements}"
REQUIREMENTS_WITH_LOCAL_WHEELS = "{requirements_with_local_wheels}"
""".format(
            version = version,
            wheel_name = wheel_name,
            wheel_collab = wheel_collab,
            requirements = str(requirements),
            requirements_with_local_wheels = requirements_with_local_wheels,
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

    version = _parse_python_version(version)

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

    print("Using hermetic Python %s" % version)  # buildifier: disable=print
    return version

def _parse_python_version(version_str):
    if version_str.startswith("Python "):
        py_ver_chunks = version_str[7:].split(".")
        return "%s.%s" % (py_ver_chunks[0], py_ver_chunks[1])
    return version_str

def _get_injected_local_wheels(
        ctx,
        py_version,
        local_wheel_workspaces,
        local_wheels_dir):
    local_wheel_requirements = []
    py_ver_marker = "-cp%s-" % py_version.replace(".", "")
    wheels = {}

    if local_wheel_workspaces:
        for local_wheel_workspace in local_wheel_workspaces:
            local_wheel_workspace_path = ctx.path(local_wheel_workspace)
            dist_folder = ctx.attr.local_wheel_dist_folder
            dist_folder_path = local_wheel_workspace_path.dirname.get_child(dist_folder)
            if dist_folder_path.exists:
                dist_wheels = dist_folder_path.readdir()
                _process_dist_wheels(dist_wheels, wheels, py_ver_marker)
    if local_wheels_dir:
        dist_folder_path = ctx.path(local_wheels_dir)
        if dist_folder_path.exists:
            dist_wheels = dist_folder_path.readdir()
            _process_dist_wheels(dist_wheels, wheels, py_ver_marker)

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
    },
    environ = [
        "TF_PYTHON_VERSION",
        "HERMETIC_PYTHON_VERSION",
        "WHEEL_NAME",
        "WHEEL_COLLAB",
    ],
)

def _process_dist_wheels(dist_wheels, wheels, py_ver_marker):
    for wheel in dist_wheels:
        bn = wheel.basename
        if not bn.endswith(".whl") or bn.find(py_ver_marker) < 0:
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

def _custom_python_interpreter_impl(ctx):
    version = ctx.attr.version
    strip_prefix = ctx.attr.strip_prefix.format(version = version)
    urls = [url.format(version = version) for url in ctx.attr.urls]
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

    configure_params = []
    if "CC" in ctx.os.environ:
        configure_params.append("CC={}".format(ctx.os.environ["CC"]))
    if "CXX" in ctx.os.environ:
        configure_params.append("CXX={}".format(ctx.os.environ["CXX"]))

    configure_params.append("--enable-optimizations")
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
