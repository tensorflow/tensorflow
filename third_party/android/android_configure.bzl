"""Repository rule for Android SDK and NDK autoconfiguration.

`android_configure` depends on the following environment variables:

  * `ANDROID_NDK_HOME`: Location of Android NDK root.
  * `ANDROID_SDK_HOME`: Location of Android SDK root.
  * `ANDROID_SDK_API_LEVEL`: Desired Android SDK API version.
  * `ANDROID_NDK_API_LEVEL`: Desired Android NDK API version.
  * `ANDROID_BUILD_TOOLS_VERSION`: Desired Android build tools version.
"""

# TODO(mikecase): Move logic for getting default values for the env variables
# from configure.py script into this rule.

_ANDROID_NDK_HOME = "ANDROID_NDK_HOME"
_ANDROID_SDK_HOME = "ANDROID_SDK_HOME"
_ANDROID_NDK_VERSION = "ANDROID_NDK_VERSION"
_ANDROID_NDK_API_LEVEL = "ANDROID_NDK_API_LEVEL"
_ANDROID_SDK_API_LEVEL = "ANDROID_SDK_API_LEVEL"
_ANDROID_BUILD_TOOLS_VERSION = "ANDROID_BUILD_TOOLS_VERSION"

_ANDROID_SDK_REPO_TEMPLATE = """
    native.android_sdk_repository(
        name="androidsdk",
        path="%s",
        api_level=%s,
        build_tools_version="%s",
    )
"""

_ANDROID_NDK_REPO_TEMPLATE_INTERNAL = """
    native.android_ndk_repository(
        name="androidndk",
        path="%s",
        api_level=%s,
    )
"""

_ANDROID_NDK_REPO_TEMPLATE_STARLARK = """
    android_ndk_repository(
        name="androidndk",
        path="%s",
        api_level=%s,
    )

    # Bind android/crosstool to support legacy select()
    # https://github.com/bazelbuild/rules_android_ndk/issues/31#issuecomment-1396182185
    native.bind(
        name = "android/crosstool",
        actual = "@androidndk//:toolchain",
    )
"""

_ANDROID_NDK_VERION_FOR_STARLARK_RULES = 25

# Import NDK Starlark rules. Shouldn't have any indentation.
_ANDROID_NDK_STARLARK_RULES = """
load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")
"""

def _android_autoconf_impl(repository_ctx):
    """Implementation of the android_autoconf repository rule."""
    sdk_home = repository_ctx.os.environ.get(_ANDROID_SDK_HOME)
    sdk_api_level = repository_ctx.os.environ.get(_ANDROID_SDK_API_LEVEL)
    build_tools_version = repository_ctx.os.environ.get(
        _ANDROID_BUILD_TOOLS_VERSION,
    )
    ndk_home = repository_ctx.os.environ.get(_ANDROID_NDK_HOME)
    ndk_api_level = repository_ctx.os.environ.get(_ANDROID_NDK_API_LEVEL)
    ndk_version_str = repository_ctx.os.environ.get(_ANDROID_NDK_VERSION)
    ndk_version = int(ndk_version_str) if ndk_version_str else 0

    sdk_rule = ""
    if all([sdk_home, sdk_api_level, build_tools_version]):
        sdk_rule = _ANDROID_SDK_REPO_TEMPLATE % (
            sdk_home,
            sdk_api_level,
            build_tools_version,
        )

    ndk_rule = ""
    ndk_starlark_rules = ""
    if all([ndk_home, ndk_api_level]):
        if ndk_version >= _ANDROID_NDK_VERION_FOR_STARLARK_RULES:
            ndk_starlark_rules = _ANDROID_NDK_STARLARK_RULES
            ndk_rule = _ANDROID_NDK_REPO_TEMPLATE_STARLARK % (ndk_home, ndk_api_level)
        else:
            ndk_rule = _ANDROID_NDK_REPO_TEMPLATE_INTERNAL % (ndk_home, ndk_api_level)

    if ndk_rule == "" and sdk_rule == "":
        sdk_rule = "pass"

    repository_ctx.template(
        "BUILD",
        Label("//third_party/android:android_configure.BUILD.tpl"),
    )
    repository_ctx.template(
        "android.bzl",
        Label("//third_party/android:android.bzl.tpl"),
        substitutions = {
            "MAYBE_ANDROID_NDK_STARLARK_RULES": ndk_starlark_rules,
            "MAYBE_ANDROID_SDK_REPOSITORY": sdk_rule,
            "MAYBE_ANDROID_NDK_REPOSITORY": ndk_rule,
        },
    )

android_configure = repository_rule(
    implementation = _android_autoconf_impl,
    environ = [
        _ANDROID_SDK_API_LEVEL,
        _ANDROID_NDK_VERSION,
        _ANDROID_NDK_API_LEVEL,
        _ANDROID_BUILD_TOOLS_VERSION,
        _ANDROID_NDK_HOME,
        _ANDROID_SDK_HOME,
    ],
)
"""Writes Android SDK and NDK rules.

Add the following to your WORKSPACE FILE:

```python
android_configure(name = "local_config_android")
```

Args:
  name: A unique name for this workspace rule.
"""
