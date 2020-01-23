load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_cc_test",  # @unused
    "tf_copts",  # @unused
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "ipynb",
    srcs = ["ipynb.py"],
    srcs_version = "PY2AND3",
    deps = ["@six_archive//:six"],
)

py_library(
    name = "ast_edits",
    srcs = ["ast_edits.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@pasta",
        "@six_archive//:six",
    ],
)

py_test(
    name = "ast_edits_test",
    srcs = ["ast_edits_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":ast_edits",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "tf_upgrade",
    srcs = ["tf_upgrade.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":tf_upgrade_lib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "tf_upgrade_lib",
    srcs = ["tf_upgrade.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ast_edits",
        "@six_archive//:six",
    ],
)

py_test(
    name = "tf_upgrade_test",
    srcs = ["tf_upgrade_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":tf_upgrade_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "renames_v2",
    srcs = ["renames_v2.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "reorders_v2",
    srcs = ["reorders_v2.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "all_renames_v2",
    srcs = ["all_renames_v2.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow:__subpackages__",
    ],
    deps = [":renames_v2"],
)

py_test(
    name = "all_renames_v2_test",
    srcs = ["all_renames_v2_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":all_renames_v2",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "module_deprecations_v2",
    srcs = ["module_deprecations_v2.py"],
    srcs_version = "PY2AND3",
    deps = [":ast_edits"],
)

py_library(
    name = "tf_upgrade_v2_lib",
    srcs = ["tf_upgrade_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_renames_v2",
        ":ast_edits",
        ":module_deprecations_v2",
        ":reorders_v2",
        "@six_archive//:six",
    ],
)

py_library(
    name = "tf_upgrade_v2_safety_lib",
    srcs = ["tf_upgrade_v2_safety.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_renames_v2",
        ":ast_edits",
        ":module_deprecations_v2",
    ],
)

py_binary(
    name = "tf_upgrade_v2",
    srcs = ["tf_upgrade_v2_main.py"],
    main = "tf_upgrade_v2_main.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":ast_edits",
        ":ipynb",
        ":tf_upgrade_v2_lib",
        ":tf_upgrade_v2_safety_lib",
        "@six_archive//:six",
    ],
)

py_test(
    name = "tf_upgrade_v2_test",
    srcs = ["tf_upgrade_v2_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "v1only",
    ],
    deps = [
        ":tf_upgrade_v2_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

py_test(
    name = "tf_upgrade_v2_safety_test",
    srcs = ["tf_upgrade_v2_safety_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":tf_upgrade_v2_safety_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
)

# Keep for reference, this test will succeed in 0.11 but fail in 1.0
# py_test(
#     name = "test_file_v0_11",
#     size = "small",
#     srcs = ["testdata/test_file_v0_11.py"],
#     srcs_version = "PY2AND3",
#     deps = [
#         "//tensorflow:tensorflow_py",
#     ],
# )

genrule(
    name = "generate_upgraded_file",
    testonly = 1,
    srcs = ["testdata/test_file_v0_11.py"],
    outs = [
        "test_file_v1_0.py",
        "report.txt",
    ],
    cmd = ("$(location :tf_upgrade)" +
           " --infile $(location testdata/test_file_v0_11.py)" +
           " --outfile $(location test_file_v1_0.py)" +
           " --reportfile $(location report.txt)"),
    tools = [":tf_upgrade"],
)

py_test(
    name = "test_file_v1_0",
    size = "small",
    srcs = ["test_file_v1_0.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

genrule(
    name = "generate_upgraded_file_v2",
    testonly = 1,
    srcs = ["testdata/test_file_v1_12.py"],
    outs = [
        "test_file_v2_0.py",
        "report_v2.txt",
    ],
    cmd = ("$(location :tf_upgrade_v2)" +
           " --infile $(location testdata/test_file_v1_12.py)" +
           " --outfile $(location test_file_v2_0.py)" +
           " --reportfile $(location report_v2.txt) && " +
           "sed -i'.original' 's/_TEST_VERSION = 1/_TEST_VERSION = 2/g' $(location test_file_v2_0.py)"),
    tools = [":tf_upgrade_v2"],
)

py_test(
    name = "test_file_v1_12",
    size = "small",
    srcs = ["testdata/test_file_v1_12.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "v1only",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "test_file_v2_0",
    size = "small",
    srcs = ["test_file_v2_0.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

exports_files(
    [
        "all_renames_v2.py",
        "ast_edits.py",
        "tf_upgrade.py",
        "renames_v2.py",
        "testdata/test_file_v0_11.py",
        "testdata/test_file_v1_12.py",
    ],
)
