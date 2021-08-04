package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_binary(
    name = "generate_v2_renames_map",
    srcs = ["generate_v2_renames_map.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:lib",
        "//tensorflow/python:no_contrib",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/compatibility:all_renames_v2",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "generate_v2_reorders_map",
    srcs = ["generate_v2_reorders_map.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:lib",
        "//tensorflow/python:no_contrib",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/compatibility:tf_upgrade_v2_lib",
    ],
)
