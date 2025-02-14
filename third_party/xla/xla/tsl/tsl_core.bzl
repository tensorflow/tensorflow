"""Core utilities for tsl.

This file must not load any other bzl files.

We put xla_bzl_visibility() here instead of tsl.bzl to avoid circular dependencies:
tsl.bzl loads other .bzl files (e.g. @local_config_tensorrt//:build_defs.bzl),
and those .bzl files need to use xla_bzl_visibility(). If we put xla_bzl_visibility() in tsl.bzl,
then we'd get a circular dependency: tsl.bzl -> tensorrt_build_defs.bzl -> tsl.bzl.
"""

def _if_google(google_value, oss_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    _ = (google_value, oss_value)  # buildifier: disable=unused-variable
    return oss_value  # copybara:comment_replace return google_value

def _make_abs_targets(targets):
    """Turns a list of targets into absolute labels.
    """

    return ["//" + target for target in targets]

def xla_bzl_visibility(targets):
    """To be used in a .bzl file's visibility declaration:

    visibility(xla_bzl_visibility(["foo/bar"]))

    restricts the current .bzl file's visibility inside Google, but leaves it public in OSS.

    This is needed for avoiding false alarms from the leakr checker. Note that we cannot
    simplify the usage to xla_bzl_visibility(["foo/bar"]) because the visibility declaration
    must be at the top level of a .bzl file - it cannot be inside a function.
    """

    return _if_google(_make_abs_targets(targets), "public")

# Ideally, this should be at the top of the file, but that's impossible because we need to
# define the xla_bzl_visibility function first.
visibility(xla_bzl_visibility([
    "platforms/xla/...",
    "third_party/tensorflow/...",
]))
