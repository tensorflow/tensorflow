# Description:
#   tf.llm — High-level API for loading and fine-tuning open-source LLMs.
#   Provides tf.llm.from_pretrained(), model.compile(), model.fit(),
#   and model.generate() — a Keras-compatible interface over HuggingFace.

load("@xla//third_party/rules_python/python:defs.bzl", "py_library", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# ---------------------------------------------------------------------------
# Main library target
# ---------------------------------------------------------------------------

py_library(
    name = "llm",
    srcs = [
        "__init__.py",
        "llm_model.py",
    ],
    srcs_version = "PY3",
    deps = [
        # HuggingFace — declared as optional; the module does lazy imports
        # so TF itself stays importable even without these packages.
        # "@pypi//transformers",  # uncomment to make hard dependency
        # "@pypi//peft",          # uncomment to make LoRA a hard dependency
        # "@pypi//torch",         # uncomment to make PyTorch a hard dependency
    ],
)

# ---------------------------------------------------------------------------
# Unit-test target
# ---------------------------------------------------------------------------

py_test(
    name = "llm_model_test",
    size = "small",
    srcs = ["llm_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":llm",
        "//tensorflow/python/platform:client_testlib",
    ],
)

# ---------------------------------------------------------------------------
# Filegroup — for source packaging / distribution
# ---------------------------------------------------------------------------

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/llm:__pkg__"],
)
