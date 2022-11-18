load("//tensorflow:strict.default.bzl", "py_strict_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

py_strict_test(
    name = "multiple_results_test",
    srcs = ["multiple_results_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_acos_test",
    srcs = ["tf_acos_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_binary_bcast_test",
    srcs = ["tf_binary_bcast_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",  # TODO(b/201803253): TFRT pybindings not in OSS.
        "nomsan",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_broadcast_to_test",
    srcs = ["tf_broadcast_to_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",  # TODO(b/201803253): TFRT pybindings not in OSS.
        "nomsan",  # TODO(b/210849019)
    ],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_cast_test",
    srcs = ["tf_cast_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_const_test",
    srcs = ["tf_const_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_controlflow_test",
    srcs = ["tf_controlflow_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_function_test",
    srcs = ["tf_function_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_log1p_test",
    srcs = ["tf_log1p_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_logical_ops_test",
    srcs = ["tf_logical_ops_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_math_ops_test",
    srcs = ["tf_math_ops_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "tf_matmul_test",
    srcs = ["tf_matmul_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_mean_test",
    srcs = ["tf_mean_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_metadata_ops_test",
    srcs = ["tf_metadata_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_pack_test",
    srcs = ["tf_pack_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_reshape_test",
    srcs = ["tf_reshape_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_select_test",
    srcs = ["tf_select_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_strided_slice_test",
    srcs = ["tf_strided_slice_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_transpose_test",
    srcs = ["tf_transpose_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss",
        "no_pip",  # TODO(b/201803253): TFRT pybindings not in OSS.
    ],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_reduction_test",
    srcs = ["tf_reduction_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_softmax_test",
    srcs = ["tf_softmax_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/201803253): TFRT pybindings not in OSS.
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_jitrt",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

td_library(
    name = "python_test_attrs_td_files",
    srcs = ["python_test_attrs.td"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "python_test_attrs_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "python_test_attrs.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "python_test_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "python_test_attrs.td",
    deps = [":python_test_attrs_td_files"],
)

cc_library(
    name = "python_test_attrs",
    srcs = [
        "python_test_attrs.cc",
    ],
    hdrs = [
        "python_test_attrs.h",
    ],
    deps = [
        ":python_test_attrs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "python_test_attrs_registration",
    srcs = ["python_test_attrs_registration.cc"],
    hdrs = ["python_test_attrs_registration.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":python_test_attrs",
        "@llvm-project//mlir:IR",
    ],
)
