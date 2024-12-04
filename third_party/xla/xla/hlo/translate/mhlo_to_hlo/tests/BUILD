load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "add.mlir",
            "attributes.mlir",
            "call.mlir",
            "case.mlir",
            "composite.mlir",
            "dynamic.mlir",
            "export-with-layouts.mlir",
            "export.mlir",
            "export_async.mlir",
            "export_and_check_layouts.mlir",
            "export_entry_computation_layout.mlir",
            "export_large_constants.mlir",
            "export_replicas.mlir",
            "frontend_attributes.mlir",
            "function.mlir",
            "fusion.mlir",
            "if.mlir",
            "input_output_aliasing.mlir",
            "int4.mlir",
            "layouts_and_names.mlir",
            "location_to_op_metadata.mlir",
            "location_to_stacktrace.mlir",
            "missing_main.mlir",
            "module_config.mlir",
            "module_attributes.mlir",
            "multiple_return_tuple.mlir",
            "opaque_elements_attr.mlir",
            "ragged_dot.mlir",
            "rng_get_and_update_state.mlir",
            "sharding.mlir",
            "simple.mlir",
            "unsupported_type.mlir",
            "while.mlir",
            "while_free_vars.mlir",
        ],
        include = [
            "*.mlir",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    data = [":test_utilities"],
    tools = [
        "//xla/hlo/translate:xla-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
)
