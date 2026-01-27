"""If we're building a nightly, we use this to pass a timestamp for the wheel version."""

def _nightly_timestamp_impl(rctx):
    timestamp_val = rctx.getenv("XLA_NIGHTLY_TIMESTAMP", "")  # Default to ""

    # Smuggle the value via a new .bzl file
    if timestamp_val:
        rctx.file(
            "timestamp.bzl",
            content = 'XLA_NIGHTLY_TIMESTAMP = ".dev{}"'.format(timestamp_val),
        )
    else:
        rctx.file(
            "timestamp.bzl",
            content = 'XLA_NIGHTLY_TIMESTAMP = ""',
        )

    # Create a BUILD file to make timestamp.bzl addressable
    rctx.file("BUILD.bazel", content = "")

nightly_timestamp_repo = repository_rule(
    implementation = _nightly_timestamp_impl,
    environ = ["XLA_NIGHTLY_TIMESTAMP"],
)

# bzlmod implementation
def _nightly_timestamp_ext_impl(mctx):  # @unused
    nightly_timestamp_repo(
        name = "nightly_timestamp",
    )

nightly_timestamp_repo_bzlmod = module_extension(
    implementation = _nightly_timestamp_ext_impl,
    environ = ["XLA_NIGHTLY_TIMESTAMP"],
)
