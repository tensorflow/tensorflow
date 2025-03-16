load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "attributes.hlo",
            "auto_layout.hlo",
            "bool_compare.hlo",
            "call.hlo",
            "case_conditional.hlo",
            "composite_call.hlo",
            "custom_call.hlo",
            "dynamic_param.hlo",
            "frontend_attributes.hlo",
            "fully_connected_reference_model.hlo",
            "fusion.hlo",
            "if_conditional.hlo",
            "import.hlo",
            "import_locs.hlo",
            "import_async.hlo",
            "import_bounded_dynamism.hlo",
            "import_entry_computation_layout.hlo",
            "layouts_and_names.hlo",
            "location.hlo",
            "module_attributes.hlo",
            "module_config.hlo",
            "operand_convert_for_convolution.hlo",
            "ragged_dot.hlo",
            "simple.hlo",
            "spmd_module_sharding.hlo",
            "stablehlo.hlo",
            "stacktrace_to_location.hlo",
            "types.hlo",
            "while.hlo",
        ],
        include = [
            "*.hlo",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/hlo/tools:hlo-translate",
        "//xla/hlo/translate:xla-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
