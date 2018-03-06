
licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_library(
    name = "ipu",
    srcs = [
      "__init__.py",
      "python/ops.py",
      "python/utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops",
    ],
)