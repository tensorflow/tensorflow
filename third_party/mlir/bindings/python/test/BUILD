# Description:
#   BUILD file for the Python wrappers for EDSCs

licenses(["notice"])  # Apache 2.0

# Export the BUILD file so automated tooling can check licenses
exports_files(["BUILD"])

load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@local_config_mlir//:run_lit.sh",
    test_file_exts = ["py"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":test_edsc",
        "//third_party/llvm/llvm:FileCheck",
    ],
)

py_binary(
    name = "test_edsc",
    srcs = ["test_py2and3.py"],
    main = "test_py2and3.py",
    python_version = "PY2",
    deps = [
        "//testing/pybase",
        "@local_config_mlir//bindings/python:_pybind",
    ],
)
