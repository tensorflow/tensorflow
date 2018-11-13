licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "common_transformers",
    srcs = [
        "anf.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "@gast_archive//:gast",
        "@six_archive//:six",
        # TODO(aqj) Revisit this dependency direction when pyct is more
        # modularized
        "//tensorflow/contrib/autograph/pyct",
    ],
)

py_test(
    name = "anf_test",
    srcs = ["anf_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common_transformers",
        "//tensorflow/python:client_testlib",
    ],
)
