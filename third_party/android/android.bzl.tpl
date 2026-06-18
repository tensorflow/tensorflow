MAYBE_ANDROID_NDK_STARLARK_RULES

"""Set up configurable Android SDK and NDK dependencies."""

def android_workspace():
  # String for replacement in Bazel template.
  # These will either be replaced by android_sdk_repository if various ENV
  # variables are set when `local_config_android` repo_rule is run, or they
  # will be replaced by noops otherwise.
  MAYBE_ANDROID_SDK_REPOSITORY
  MAYBE_ANDROID_NDK_REPOSITORY
