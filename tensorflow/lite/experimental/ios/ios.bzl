"""TensorFlow Lite Build Configurations for iOS"""

# Current version of the TensorFlow Lite iOS libraries.
TFL_IOS_BUILD_VERSION = "0.2.0"

TFL_MINIMUM_OS_VERSION = "9.0"

# Default tags for filtering iOS targets. Targets are restricted to Apple platforms.
TFL_DEFAULT_TAGS = [
    "apple",
]

# Following sanitizer tests are not supported by iOS test targets.
TFL_DISABLED_SANITIZER_TAGS = [
    "noasan",
    "nomsan",
    "notsan",
]
