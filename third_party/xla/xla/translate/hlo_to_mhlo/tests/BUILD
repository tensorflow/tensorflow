load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "bool_compare.hlotxt",
            "case_conditional.hlotxt",
            "custom_call.hlotxt",
            "dynamic_param.hlo",
            "entry_computation_layout.hlotxt",
            "frontend_attributes.hlotxt",
            "fully_connected_reference_model.hlotxt",
            "fusion.hlotxt",
            "if_conditional.hlotxt",
            "import.hlotxt",
            "import_async.hlotxt",
            "layouts_and_names.hlotxt",
            "location.hlotxt",
            "module_attributes.hlo",
            "send_recv.hlotxt",
            "simple.hlo",
            "spmd_module_sharding.hlo",
            "stacktrace_to_location.hlo",
            "types.hlotxt",
            "while.hlotxt",
        ],
        include = [
            "*.hlotxt",
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
