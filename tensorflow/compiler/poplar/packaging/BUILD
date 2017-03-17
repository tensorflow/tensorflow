package(default_visibility = ["//visibility:private"])

sh_binary(
    name = "build_pip_package",
    srcs = ["pip_package.sh"],
    data = [
            "README.rst",
            "setup.py",
            "//tensorflow/compiler/poplar:poplar_release",
    ],
)
