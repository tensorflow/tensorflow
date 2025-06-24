load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//learning/brain/testing/tf2_migration_tools:__pkg__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "ipynb",
    srcs = ["ipynb.py"],
)

py_strict_library(
    name = "ast_edits",
    srcs = ["ast_edits.py"],
    deps = [
        "@pasta",
        "@six_archive//:six",  # buildcleaner: keep
    ],
)

py_strict_test(
    name = "ast_edits_test",
    srcs = ["ast_edits_test.py"],
    deps = [
        ":ast_edits",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_binary(
    name = "tf_upgrade",
    srcs = ["tf_upgrade.py"],
    deps = [
        ":tf_upgrade_lib",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "tf_upgrade_lib",
    srcs = ["tf_upgrade.py"],
    deps = [":ast_edits"],
)

py_strict_test(
    name = "tf_upgrade_test",
    srcs = ["tf_upgrade_test.py"],
    tags = [
        "no_pip",
    ],
    deps = [
        ":ast_edits",
        ":tf_upgrade_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "renames_v2",
    srcs = ["renames_v2.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow:internal"],
)

py_strict_library(
    name = "reorders_v2",
    srcs = ["reorders_v2.py"],
)

py_strict_library(
    name = "all_renames_v2",
    srcs = ["all_renames_v2.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [":renames_v2"],
)

py_strict_test(
    name = "all_renames_v2_test",
    srcs = ["all_renames_v2_test.py"],
    deps = [
        ":all_renames_v2",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "module_deprecations_v2",
    srcs = ["module_deprecations_v2.py"],
    deps = [":ast_edits"],
)

py_strict_library(
    name = "tf_upgrade_v2_lib",
    srcs = ["tf_upgrade_v2.py"],
    deps = [
        ":all_renames_v2",
        ":ast_edits",
        ":module_deprecations_v2",
        ":reorders_v2",
        "@pasta",
    ],
)

py_strict_library(
    name = "tf_upgrade_v2_safety_lib",
    srcs = ["tf_upgrade_v2_safety.py"],
    deps = [
        ":all_renames_v2",
        ":ast_edits",
        ":module_deprecations_v2",
    ],
)

py_strict_binary(
    name = "tf_upgrade_v2",
    srcs = ["tf_upgrade_v2_main.py"],
    main = "tf_upgrade_v2_main.py",
    deps = [
        ":ast_edits",
        ":ipynb",
        ":tf_upgrade_v2_lib",
        ":tf_upgrade_v2_safety_lib",
    ],
)

py_strict_test(
    name = "tf_upgrade_v2_test",
    srcs = ["tf_upgrade_v2_test.py"],
    shard_count = 5,
    tags = [
        "no_windows",
        "v1only",
    ],
    deps = [
        ":ast_edits",
        ":tf_upgrade_v2_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "@six_archive//:six",
        # copybara:uncomment "//third_party/py/tensorflow:tensorflow_compat_v1_estimator",
        # copybara:uncomment "//third_party/py/tensorflow:tensorflow_compat_v2_estimator",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "tf_upgrade_v2_safety_test",
    srcs = ["tf_upgrade_v2_safety_test.py"],
    deps = [
        ":ast_edits",
        ":tf_upgrade_v2_safety_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

# Keep for reference, this test will succeed in 0.11 but fail in 1.0
# py_test(
#     name = "test_file_v0_11",
#     size = "small",
#     srcs = ["testdata/test_file_v0_11.py"],
#     srcs_version = "PY3",
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

py_strict_test(
    name = "test_file_v1_0",
    size = "medium",
    srcs = ["test_file_v1_0.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
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

py_strict_test(
    name = "test_file_v1_12",
    size = "medium",
    srcs = ["testdata/test_file_v1_12.py"],
    tags = [
        "no_windows",
        "v1only",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "test_file_v2_0",
    size = "medium",
    srcs = ["test_file_v2_0.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
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
