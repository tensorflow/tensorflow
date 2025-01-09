"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load(
    "@python_version_repo//:py_version.bzl",
    "HERMETIC_PYTHON_PREFIX",
    "HERMETIC_PYTHON_SHA256",
    "HERMETIC_PYTHON_URL",
    "HERMETIC_PYTHON_VERSION",
)
load("@rules_python//python:repositories.bzl", "python_register_toolchains")
load("@rules_python//python:versions.bzl", "MINOR_MAPPING", "PLATFORMS")

def python_init_toolchains(name = "python", python_version = None, **kwargs):
    """Register hermetic python toolchains.

    Args:
        name: name of the toolchain, "python" by default (it is strongly
          recommended to rely on the default).
        python_version: version of the python to register; if set it will bypass
          kwargs to underlying python_register_toolchains as is (manual
          configuration), otherwise it will automatically configure toolchains
          based on HERMETIC_PYTHON_URL and/or based on HERMETIC_PYTHON_VERSION
          repo_env values.
        **kwargs: additional arguments to pass to python_register_toolchains.
    """

    if python_version:
        python_register_toolchains(
            name = name,
            python_version = python_version,
            **kwargs
        )
    elif HERMETIC_PYTHON_URL:
        tool_version = MINOR_MAPPING.get(HERMETIC_PYTHON_VERSION)
        if not tool_version:
            tool_version = HERMETIC_PYTHON_VERSION + ".0"
        url_components = HERMETIC_PYTHON_URL.split("://", 1)

        sha256s = {}
        for platform in PLATFORMS.keys():
            # Avoid obscure platforms for now just in case
            if "aarch64" in platform or "x86_64" in platform:
                sha256s[platform] = HERMETIC_PYTHON_SHA256

        python_register_toolchains(
            name = name,
            base_url = url_components[0] + "://",
            ignore_root_user_error = True,
            python_version = tool_version,
            tool_versions = {
                tool_version: {
                    "url": url_components[1],
                    "sha256": sha256s,
                    "strip_prefix": HERMETIC_PYTHON_PREFIX,
                },
            },
            # minor_mapping = {HERMETIC_PYTHON_VERSION: tool_version}
        )
    elif HERMETIC_PYTHON_VERSION in MINOR_MAPPING:
        python_register_toolchains(
            name = name,
            ignore_root_user_error = True,
            python_version = HERMETIC_PYTHON_VERSION,
        )
