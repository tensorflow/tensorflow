"""Repository rule for remote GPU autoconfiguration.

This rule creates the starlark file
//third_party/toolchains/remote:execution.bzl
providing the function `gpu_test_tags`.

`gpu_test_tags` will return:

  * `local`: if `REMOTE_GPU_TESTING` is false, allowing CPU tests to run
    remotely and GPU tests to run locally in the same bazel invocation.
  * `remote-gpu`: if `REMOTE_GPU_TESTING` is true; this allows rules to
    set an execution requirement that enables a GPU-enabled remote platform.
"""

_REMOTE_GPU_TESTING = "REMOTE_GPU_TESTING"

def _flag_enabled(repository_ctx, flag_name):
    if flag_name not in repository_ctx.os.environ:
        return False
    return repository_ctx.os.environ[flag_name].strip() == "1"

def _remote_execution_configure(repository_ctx):
    # If we do not support remote gpu test execution, mark them as local, so we
    # can combine remote builds with local gpu tests.
    gpu_test_tags = "\"local\""
    if _flag_enabled(repository_ctx, _REMOTE_GPU_TESTING):
        gpu_test_tags = "\"remote-gpu\""
    repository_ctx.template(
        "remote_execution.bzl",
        Label("//third_party/toolchains/remote:execution.bzl.tpl"),
        {
            "%{gpu_test_tags}": gpu_test_tags,
        },
    )
    repository_ctx.template(
        "BUILD",
        Label("//third_party/toolchains/remote:BUILD.tpl"),
    )

remote_execution_configure = repository_rule(
    implementation = _remote_execution_configure,
    environ = [_REMOTE_GPU_TESTING],
)
