"""OSS variant of xla_js.bzl for JavaScript and TypeScript rules."""

load("@aspect_rules_esbuild//esbuild:defs.bzl", "esbuild")
load("@aspect_rules_ts//ts:defs.bzl", "ts_project")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_XLA_USERS",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_XLA_USERS)

def xla_ts_library(**kwargs):
    """Macro wrapping ts_project for XLA TypeScript libraries.

    Args:
        **kwargs: Keyword arguments passed to ts_project.
    """
    kwargs.pop("compatible_with", None)
    if "tsconfig" not in kwargs:
        kwargs["tsconfig"] = {}
    ts_project(**kwargs)

def xla_js_binary(**kwargs):
    """Macro wrapping esbuild for XLA JavaScript binaries.

    Args:
        **kwargs: Keyword arguments passed to esbuild.
    """
    kwargs.pop("compatible_with", None)
    if "entry_point" not in kwargs:
        if "deps" in kwargs and len(kwargs["deps"]) > 0:
            entry = kwargs["deps"][0]
            if entry.startswith(":"):
                entry = entry[1:]
            kwargs["entry_point"] = entry + ".js"
    esbuild(output = kwargs["name"] + ".js", bundle = True, **kwargs)

def xla_js_bzl_library(name = "xla_js_bzl_library"):
    """Macro wrapping bzl_library for xla_js.bzl.

    Args:
        name: Name of the bzl_library target.
    """
    bzl_library(
        name = "xla_js_bzl",
        srcs = ["xla_js.default.bzl"],
        deps = [
            "@bazel_skylib//:bzl_library",
        ],
    )
