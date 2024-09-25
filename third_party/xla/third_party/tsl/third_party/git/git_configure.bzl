"""Repository rule for Git autoconfiguration.

`git_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: location of python binary.
"""

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sGit Configuration Error:%s %s\n" % (red, no_color, msg))

def _get_python_bin(repository_ctx):
    """Gets the python bin path."""
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH)
    if python_bin != None:
        return python_bin
    python_bin_path = repository_ctx.which("python3")
    if python_bin_path != None:
        return str(python_bin_path)
    python_bin_path = repository_ctx.which("python")
    if python_bin_path != None:
        return str(python_bin_path)
    _fail("Cannot find python in PATH, please make sure " +
          "python is installed and add its directory in PATH, or --define " +
          "%s='/something/else'.\nPATH=%s" % (
              _PYTHON_BIN_PATH,
              repository_ctx.os.environ.get("PATH", ""),
          ))

def _git_conf_impl(repository_ctx):
    repository_ctx.template(
        "BUILD",
        Label("//third_party/git:BUILD.tpl"),
    )

    tensorflow_root_path = str(repository_ctx.path(
        Label("@org_tensorflow//:BUILD"),
    ))[:-len("BUILD")]
    python_script_path = repository_ctx.path(
        Label("@org_tensorflow//tensorflow/tools/git:gen_git_source.py"),
    )
    generated_files_path = repository_ctx.path("gen")

    r = repository_ctx.execute(
        ["test", "-f", "%s/.git/logs/HEAD" % tensorflow_root_path],
    )
    if r.return_code == 0:
        unused_var = repository_ctx.path(Label("//:.git/HEAD"))  # pylint: disable=unused-variable

    result = repository_ctx.execute([
        _get_python_bin(repository_ctx),
        python_script_path,
        "--configure",
        tensorflow_root_path,
        "--gen_root_path",
        generated_files_path,
    ], quiet = False)

    if not result.return_code == 0:
        _fail(result.stderr)

git_configure = repository_rule(
    implementation = _git_conf_impl,
    environ = [
        _PYTHON_BIN_PATH,
    ],
)
