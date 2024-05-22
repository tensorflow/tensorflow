"""
Repository rule to manage hermetic Python interpreter under Bazel.

Version can be set via build parameter "--repo_env=HERMETIC_PYTHON_VERSION=3.11"
Defaults to 3.11.

To set wheel name, add "--repo_env=WHEEL_NAME=tensorflow_cpu"
"""

DEFAULT_VERSION = "3.11"
WARNING = """
HERMETIC_PYTHON_VERSION variable was not set correctly, using default version.
Python {} will be used.
To select Python version, either set HERMETIC_PYTHON_VERSION env variable in
your shell:
  export HERMETIC_PYTHON_VERSION=3.12
OR pass it as an argument to bazel command directly or inside your .bazelrc
file:
  --repo_env=HERMETIC_PYTHON_VERSION=3.12
""".format(DEFAULT_VERSION)

content = """TF_PYTHON_VERSION = "{version}"
HERMETIC_PYTHON_VERSION = "{version}"
WHEEL_NAME = "{wheel_name}"
WHEEL_COLLAB = "{wheel_collab}"
REQUIREMENTS = "{requirements}"
"""

def _python_repository_impl(ctx):
    ctx.file("BUILD", "")
    version_legacy = ctx.os.environ.get("TF_PYTHON_VERSION", "")
    version = ctx.os.environ.get("HERMETIC_PYTHON_VERSION", "")
    if not version:
        version = version_legacy
    else:
        version_legacy = version

    wheel_name = ctx.os.environ.get("WHEEL_NAME", "tensorflow")
    wheel_collab = ctx.os.environ.get("WHEEL_COLLAB", False)
    if not version:
        print(WARNING)  # buildifier: disable=print
        version = DEFAULT_VERSION
    else:
        print("Using hermetic Python %s" % version)  # buildifier: disable=print

    requirements = ""
    for i in range(0, len(ctx.attr.requirements_locks)):
        if ctx.attr.requirements_versions[i] == version:
            requirements = ctx.attr.requirements_locks[i]
            break

    ctx.file(
        "py_version.bzl",
        content.format(
            version = version,
            wheel_name = wheel_name,
            wheel_collab = wheel_collab,
            requirements = str(requirements),
        ),
    )

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
    },
    environ = [
        "TF_PYTHON_VERSION",
        "HERMETIC_PYTHON_VERSION",
        "WHEEL_NAME",
        "WHEEL_COLLAB",
    ],
)

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
