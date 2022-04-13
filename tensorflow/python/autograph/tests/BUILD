load(":reference_test.bzl", "reference_test")

licenses(["notice"])

py_library(
    name = "reference_tests",
    srcs = ["reference_test_base.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

# # Misc

# ## Boolean expressions, equality testing

reference_test(name = "logical_expression_test")

reference_test(name = "ext_slice_test")

# ## Type annotations

reference_test(name = "type_annotations_test")

# # Scoping and modularity

# ## Composite names

reference_test(name = "composite_names_in_control_flow_test")

# ## Function Calls

reference_test(name = "call_to_builtin_function_test")

reference_test(name = "call_to_lambda_function_test")

reference_test(name = "call_to_named_tuple_test")

reference_test(name = "call_to_numpy_function_test")

reference_test(name = "call_to_print_function_test")

reference_test(name = "call_to_tf_api_test")

reference_test(name = "call_to_user_function_test")

# # Control Flow

reference_test(name = "basic_ifexp_test")

reference_test(name = "cond_basic_test")

reference_test(name = "early_return_test")

reference_test(name = "loop_basic_test")

reference_test(
    name = "loop_control_flow_test",
    shard_count = 20,
)

reference_test(name = "loop_control_flow_illegal_cases_test")

reference_test(name = "loop_created_variables_test")

reference_test(
    name = "loop_distributed_test",
    tags = [
        "no_oss",  # TODO(mdan): Investigate timeout reason.
    ],
)

reference_test(name = "loop_with_variable_type_test")

reference_test(name = "loop_with_variable_type_illegal_cases_test")

reference_test(name = "loop_with_function_call_test")

reference_test(name = "nested_control_flow_test")

# ## Assert

reference_test(name = "assertion_test")

# # Data Structures

# ## Generators

reference_test(name = "generator_test")

# ## Iterators

reference_test(name = "datasets_test")

# ## Lists

reference_test(
    name = "basic_list_test",
    tags = [
        "no_oss",
        "notap",
    ],
)
