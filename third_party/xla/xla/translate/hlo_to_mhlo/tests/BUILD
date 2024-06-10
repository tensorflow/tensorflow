load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "bool_compare.hlo",
            "case_conditional.hlo",
            "custom_call.hlo",
            "dynamic_param.hlo",
            "entry_computation_layout.hlo",
            "frontend_attributes.hlo",
            "fully_connected_reference_model.hlo",
            "fusion.hlo",
            "if_conditional.hlo",
            "import.hlo",
            "import_async.hlo",
            "layouts_and_names.hlo",
            "location.hlo",
            "module_attributes.hlo",
            "module_config.hlo",
            "send_recv.hlo",
            "simple.hlo",
            "spmd_module_sharding.hlo",
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
        "//xla/translate:xla-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
