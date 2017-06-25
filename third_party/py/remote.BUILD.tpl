licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

alias(
    name = "python_headers",
    actual = "%{REMOTE_PYTHON_REPO}:python_headers",
)

alias(
    name = "numpy_headers",
    actual = "%{REMOTE_PYTHON_REPO}:numpy_headers",
)
