"""SavedModel testing harness macros."""

def tf_saved_model_test(name):
    """Create a SavedModel test and return data dep for lit."""
    native.py_binary(
        name = name,
        testonly = 1,
        python_version = "PY3",
        srcs = [name + ".py"],
        deps = [
            "//tensorflow/compiler/mlir/tensorflow/tests/tf_saved_model:common",
        ],
    )
    return name
