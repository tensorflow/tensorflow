package(
    licenses = ["notice"],
)

py_library(
    name = "monomorphic_function",
    srcs = ["monomorphic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        "//tensorflow/core/function/polymorphism:function_cache",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:graph_to_function_def",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:forwardprop_util",
        "//tensorflow/python/eager:function_context",
        "//tensorflow/python/eager:function_saved_model_utils",
        "//tensorflow/python/eager:function_spec",
        "//tensorflow/python/eager:graph_only_ops",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "polymorphic_function",
    srcs = ["polymorphic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        ":monomorphic_function",
    ],
)

py_library(
    name = "quarantine",
    srcs = ["quarantine.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        ":monomorphic_function",
        ":polymorphic_function",
    ],
)
