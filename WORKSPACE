# buildifier: disable=load-on-top

workspace(name = "org_tensorflow")

# buildifier: disable=load-on-top

# We must initialize hermetic python first.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

http_archive(
    name = "rules_java",
    sha256 = "c73336802d0b4882e40770666ad055212df4ea62cfa6edf9cb0f9d29828a0934",
    url = "https://github.com/bazelbuild/rules_java/releases/download/5.3.5/rules_java-5.3.5.tar.gz",
)

http_archive(
    name = "rules_python",
    sha256 = "9d04041ac92a0985e344235f5d946f71ac543f1b1565f2cdbc9a2aaee8adf55b",
    strip_prefix = "rules_python-0.26.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.26.0/rules_python-0.26.0.tar.gz",
)

# buildifier: disable=same-origin-load
load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python:repositories.bzl", "python_register_toolchains")  # buildifier: disable=same-origin-load
load(
    "//tensorflow/tools/toolchains/python:python_repo.bzl",
    "python_repository",
)

python_repository(name = "python_version_repo")

load("@python_version_repo//:py_version.bzl", "TF_PYTHON_VERSION")

python_register_toolchains(
    name = "python",
    ignore_root_user_error = True,
    python_version = TF_PYTHON_VERSION,
)

load("@python//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

NUMPY_ANNOTATIONS = {
    "numpy": package_annotation(
        additive_build_content = """\
filegroup(
    name = "includes",
    srcs = glob(["site-packages/numpy/core/include/**/*.h"]),
)
cc_library(
    name = "numpy_headers",
    hdrs = [":includes"],
    strip_include_prefix="site-packages/numpy/core/include/",
)
""",
    ),
}

pip_parse(
    name = "pypi",
    annotations = NUMPY_ANNOTATIONS,
    python_interpreter_target = interpreter,
    requirements = "//:requirements_lock_" + TF_PYTHON_VERSION.replace(".", "_") + ".txt",
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()
