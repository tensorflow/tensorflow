"""build_defs for service/cpu."""


def runtime_copts():
  """Returns copts used for CPU runtime libraries."""
  return (["-DEIGEN_AVOID_STL_ARRAY"] + select({
      "//tensorflow:android_arm": ["-mfpu=neon"],
      "//conditions:default": []
  }) + select({
      "//tensorflow:android": ["-O2"],
      "//conditions:default": []
  }))


def runtime_logging_deps():
  """Returns deps for building CPU runtime libraries with logging functions."""
  return select({
      "//tensorflow:android": [
          # This dependency is smaller than :android_tensorflow_lib
          "//tensorflow/core:android_tensorflow_lib_selective_registration",
      ],
      "//conditions:default": [
          "//tensorflow/core:lib",
      ],
  })
