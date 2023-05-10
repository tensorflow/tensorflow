# Tests of TensorFlow variables kernels written using the Python API.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "dense_update_ops_no_tsan_test",
    size = "small",
    srcs = ["dense_update_ops_no_tsan_test.py"],
    # TODO (b/140294007): the test fails with XLA.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "dense_update_ops_test",
    size = "small",
    srcs = ["dense_update_ops_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "partitioned_variables_test",
    size = "small",
    srcs = ["partitioned_variables_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/training:saver",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "resource_variable_ops_test",
    size = "medium",
    srcs = ["resource_variable_ops_test.py"],
    # TODO(kkb): CppMemoryChecker is flaky without these two flags, investigate.
    #
    # TODO(b/128347673): Re-enable.
    tags = ["no_windows"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:memory_checker",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:cpp_shape_inference_proto_py",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/training:momentum",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "variable_ops_test",
    size = "small",
    srcs = ["variable_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "variable_scope_test",
    size = "medium",
    srcs = ["variable_scope_test.py"],
    tags = ["no_windows"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_decorator",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "variables_test",
    size = "small",
    srcs = ["variables_test.py"],
    tags = ["no_windows"],  # b/133869052
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
