# python/lib/core package

load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "tf_external_workspace_visible", "tf_python_pybind_extension")

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//tensorflow:internal",
    "//tensorflow/lite/toco/python:__pkg__",
    "//tensorflow_models:__subpackages__",
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/py/launchpad:__subpackages__",
    "//third_party/py/reverb:__subpackages__",
    "//third_party/py/neural_structured_learning:__subpackages__",
    "//third_party/py/tensorflow_examples:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",  # For benchmarks.
    "//third_party/py/tf_slim:__subpackages__",
    "//third_party/py/tensorflow_docs:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

cc_library(
    name = "ndarray_tensor_bridge",
    srcs = ["ndarray_tensor_bridge.cc"],
    hdrs = ["ndarray_tensor_bridge.h"],
    visibility = tf_external_workspace_visible(visibility + [
        "//tensorflow:ndarray_tensor_allow_list",
    ]),
    deps = [
        ":py_util",
        "//tensorflow/c:c_api_no_xla",
        "//tensorflow/c:tf_datatype_hdrs",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@local_xla//xla/tsl/python/lib/core:ml_dtypes_lib",
        "@local_xla//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "py_exception_registry",
    srcs = ["py_exception_registry.cc"],
    hdrs = ["py_exception_registry.h"],
    deps = [
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/python_runtime:headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pybind11_absl",
    hdrs = ["pybind11_absl.h"],
    features = ["-parse_headers"],
    visibility = tf_external_workspace_visible(visibility),
    deps = [
        "//tensorflow/core/platform:stringpiece",
        "@pybind11",
    ],
)

cc_library(
    name = "pybind11_lib",
    hdrs = ["pybind11_lib.h"],
    compatible_with = get_compatible_with_portable(),
    features = ["-parse_headers"],
    visibility = tf_external_workspace_visible(visibility),
    deps = [
        "@pybind11",
    ],
)

cc_library(
    name = "pybind11_status_headers",
    hdrs = [
        "py_exception_registry.h",
        "pybind11_status.h",
        "//tensorflow/c:headers",
        "//tensorflow/c:tf_status_internal_headers",
        "//tensorflow/c/eager:headers",
    ],
    features = [
        "-parse_headers",
    ],
    visibility = tf_external_workspace_visible(visibility),
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "pybind11_status",
    hdrs = [
        "py_exception_registry.h",
        "pybind11_status.h",
    ],
    features = ["-parse_headers"],
    visibility = tf_external_workspace_visible(visibility),
    deps = [
        ":pybind11_status_headers",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "pybind11_proto",
    hdrs = ["pybind11_proto.h"],
    features = ["-parse_headers"],
    visibility = tf_external_workspace_visible(visibility),
    deps = [
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

filegroup(
    name = "py_exception_registry_hdr",
    srcs = [
        "py_exception_registry.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "ndarray_tensor_hdr",
    srcs = ["ndarray_tensor.h"],
)

filegroup(
    name = "basic_hdrs",
    srcs = [
        "ndarray_tensor.h",
        "ndarray_tensor_bridge.h",
        "py_exception_registry.h",
        "pybind11_status.h",
        "safe_pyobject_ptr.h",
    ],
)

cc_library(
    name = "py_func_lib",
    srcs = ["py_func.cc"],
    hdrs = ["py_func.h"],
    deps = [
        ":ndarray_tensor",
        ":ndarray_tensor_bridge",
        ":py_util",
        "//tensorflow/c:safe_ptr",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:script_ops_op_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@local_xla//xla/tsl/python/lib/core:numpy",
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_py_func",
    srcs = ["py_func_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_py_func.pyi",
    ],
    deps = [
        "//tensorflow/python:py_func_headers_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "safe_pyobject_ptr",
    srcs = ["safe_pyobject_ptr.cc"],
    hdrs = ["safe_pyobject_ptr.h"],
    deps = [
        "//third_party/python_runtime:headers",
    ],
)

filegroup(
    name = "safe_pyobject_ptr_required_hdrs",
    srcs = ["safe_pyobject_ptr.h"],
)

cc_library(
    name = "ndarray_tensor_headers",
    hdrs = [
        "ndarray_tensor.h",
        "ndarray_tensor_bridge.h",
        "safe_pyobject_ptr.h",
        "//tensorflow/c:headers",
        "//tensorflow/c:safe_ptr_hdr",
        "//tensorflow/c/eager:headers",
        "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
    ],
    features = [
        "-parse_headers",
    ],
    visibility = tf_external_workspace_visible(visibility + [
        "//tensorflow:ndarray_tensor_allow_list",
    ]),
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core:framework_internal_headers_lib",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@local_xla//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "ndarray_tensor",
    srcs = ["ndarray_tensor.cc"],
    hdrs = ["ndarray_tensor.h"],
    visibility = tf_external_workspace_visible(visibility + [
        "//tensorflow:ndarray_tensor_allow_list",
    ]),
    deps = [
        ":ndarray_tensor_bridge",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:safe_ptr",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "@local_xla//xla/tsl/python/lib/core:ml_dtypes_lib",
        "@local_xla//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "py_seq_tensor",
    srcs = ["py_seq_tensor.cc"],
    hdrs = ["py_seq_tensor.h"],
    features = ["-parse_headers"],
    deps = [
        ":ndarray_tensor",
        ":ndarray_tensor_bridge",
        ":py_util",
        ":safe_pyobject_ptr",
        "//tensorflow/c:safe_ptr",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@local_xla//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "py_util",
    srcs = ["py_util.cc"],
    hdrs = ["py_util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:script_ops_op_lib",
        "//tensorflow/core/platform:logging",
        "//third_party/python_runtime:headers",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "pyclif_tensor",
#     srcs = ["pyclif_tensor.cc"],
#     hdrs = ["pyclif_tensor.h"],
#     visibility = ["//visibility:public"],
#     deps = [
#         ":ndarray_tensor",
#         ":ndarray_tensor_bridge",
#         ":safe_pyobject_ptr",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/status",
#         "//third_party/clif/python:clif",
#         "//third_party/python_runtime:headers",
#         "@local_xla//xla/tsl/python/lib/core:numpy",
#         "//tensorflow/core:framework",
#     ],
# )
# copybara:uncomment_end
