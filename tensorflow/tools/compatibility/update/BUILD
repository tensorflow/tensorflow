package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_binary(
    name = "generate_v2_renames_map",
    srcs = ["generate_v2_renames_map.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@six_archive//:six",
        "//tensorflow:tensorflow_py",
        # copybara:uncomment "//third_party/py/tensorflow:tensorflow_compat_v1_estimator",
        # copybara:uncomment "//third_party/py/tensorflow:tensorflow_compat_v2_estimator",
        "//tensorflow/python:lib",
        "//tensorflow/python:modules_with_exports",
        "//tensorflow/python:no_contrib",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/compatibility:all_renames_v2",
    ],
)

py_binary(
    name = "generate_v2_reorders_map",
    srcs = ["generate_v2_reorders_map.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        # copybara:uncomment "//third_party/py/tensorflow:tensorflow_compat_v1_estimator",
        # copybara:uncomment "//third_party/py/tensorflow:tensorflow_compat_v2_estimator",
        "//tensorflow/python:lib",
        "//tensorflow/python:no_contrib",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/compatibility:tf_upgrade_v2_lib",
    ],
)
