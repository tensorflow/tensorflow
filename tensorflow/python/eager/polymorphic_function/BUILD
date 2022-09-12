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
    name = "tracing_compiler",
    srcs = ["tracing_compiler.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        ":monomorphic_function",
    ],
)

py_library(
    name = "polymorphic_function",
    srcs = ["polymorphic_function.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:cond_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:while_v2",  # TODO(b/118513001): Imported via control_flow_ops; remove.
        "//tensorflow/python/distribute/parallel_device",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function_spec",
        "//tensorflow/python/eager:lift_to_graph",
        "//tensorflow/python/eager/polymorphic_function:monomorphic_function",
        "//tensorflow/python/eager/polymorphic_function:tracing_compiler",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/trackable:base",
    ],
)

py_library(
    name = "quarantine",
    srcs = ["quarantine.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/python/eager:__pkg__"],
    deps = [
        ":monomorphic_function",
        ":tracing_compiler",
    ],
)
